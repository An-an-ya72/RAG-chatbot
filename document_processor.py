import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Disable tokenizers parallelism
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Handle potential PyTorch library conflicts

from typing import List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pickle

class DocumentProcessor:
    def __init__(self, persist_dir: str = "db"):
        self.persist_dir = persist_dir
        # Make text splitter more lenient
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # Smaller chunks
            chunk_overlap=50,  # Less overlap
            length_function=len,
            separators=["\n\n", "\n", " ", ""],  # More separation options
            is_separator_regex=False
        )
        self.vectorizer = TfidfVectorizer(
            stop_words=None,
            lowercase=True,
            min_df=1,
            max_df=1.0,
            token_pattern=r'(?u)\b\w+\b'
        )
        self.chunks = []
        self.vectors = None
        
        # Create persist directory if it doesn't exist
        os.makedirs(persist_dir, exist_ok=True)
        
        # Try to load existing data
        if os.path.exists(os.path.join(persist_dir, "vectors.pkl")):
            try:
                with open(os.path.join(persist_dir, "vectors.pkl"), 'rb') as f:
                    saved_data = pickle.load(f)
                    self.chunks = saved_data['chunks']
                    self.vectorizer = saved_data['vectorizer']
                    self.vectors = saved_data['vectors']
            except Exception as e:
                print(f"Error loading saved data: {str(e)}")
    
    def process_file(self, file_path: str) -> List[str]:
        """Process a single file and return chunks"""
        try:
            if file_path.endswith('.pdf'):
                # Try UnstructuredPDFLoader first
                try:
                    print("Attempting to load PDF with UnstructuredPDFLoader...")
                    loader = UnstructuredPDFLoader(file_path)
                    documents = loader.load()
                    print(f"Successfully loaded PDF with UnstructuredPDFLoader: {len(documents)} sections")
                except Exception as e:
                    print(f"UnstructuredPDFLoader failed: {str(e)}")
                    print("Falling back to PyPDFLoader...")
                    loader = PyPDFLoader(file_path)
                    documents = loader.load()
                    print(f"Successfully loaded PDF with PyPDFLoader: {len(documents)} pages")
            elif file_path.endswith('.docx'):
                loader = Docx2txtLoader(file_path)
                documents = loader.load()
            elif file_path.endswith('.txt'):
                loader = TextLoader(file_path)
                documents = loader.load()
            else:
                raise ValueError(f"Unsupported file type: {file_path}")
            
            if not documents:
                raise ValueError("No content found in document")
            
            # Debug: Print some content from the first document
            print("\nFirst document content preview:")
            print(documents[0].page_content[:200])
            
            print(f"\nAttempting to split {len(documents)} documents...")
            new_chunks = self.text_splitter.split_documents(documents)
            
            if not new_chunks:
                print("\nWarning: Initial splitting produced no chunks. Attempting with raw text...")
                # Try splitting the raw text directly
                raw_text = "\n".join([doc.page_content for doc in documents])
                if raw_text.strip():
                    print(f"Raw text length: {len(raw_text)} characters")
                    new_chunks = self.text_splitter.create_documents([raw_text])
                    if not new_chunks:
                        raise ValueError("Failed to create chunks from raw text")
                else:
                    raise ValueError("No text content found in document")
            
            print(f"\nSuccessfully created {len(new_chunks)} chunks")
            new_texts = [chunk.page_content for chunk in new_chunks]
            
            # Debug: Print first chunk
            if new_texts:
                print("\nFirst chunk preview:")
                print(new_texts[0][:200])
            
            self.chunks.extend(new_texts)
            
            try:
                self.vectors = self.vectorizer.fit_transform(self.chunks)
                print(f"\nVectorizer vocabulary size: {len(self.vectorizer.vocabulary_)}")
                
                with open(os.path.join(self.persist_dir, "vectors.pkl"), 'wb') as f:
                    pickle.dump({
                        'chunks': self.chunks,
                        'vectorizer': self.vectorizer,
                        'vectors': self.vectors
                    }, f)
                
                return new_texts
            except ValueError as ve:
                print(f"\nVectorization error: {str(ve)}")
                print("Document content preview:")
                for i, text in enumerate(new_texts[:2]):
                    print(f"\nChunk {i} preview:")
                    print(text[:200])
                raise
            
        except Exception as e:
            print(f"\nError processing file: {str(e)}")
            raise
    
    def similarity_search(self, query: str, k: int = 4) -> List[str]:
        """Search for similar chunks of text"""
        if not self.chunks:
            return []
        
        try:
            query_vector = self.vectorizer.transform([query])
            similarities = cosine_similarity(query_vector, self.vectors).flatten()
            top_k_indices = similarities.argsort()[-k:][::-1]
            return [self.chunks[i] for i in top_k_indices]
        except Exception as e:
            print(f"Error during similarity search: {str(e)}")
            return [] 