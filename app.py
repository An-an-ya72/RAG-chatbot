import streamlit as st
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import tempfile
import os
from document_processor import DocumentProcessor

# Set page configuration
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")

# Initialize document processor
@st.cache_resource
def get_document_processor():
    return DocumentProcessor()

doc_processor = get_document_processor()

# Initialize Streamlit state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Configure the LLM and prompt
@st.cache_resource
def initialize_chain():
    template = """
    Answer the question below using the provided context. If the question cannot be answered using the context, say that you don't have enough information.

    Context from documents: {context}

    Previous conversation history: {history}

    Question: {question}

    Answer:
    """
    model = OllamaLLM(model="llama3")
    prompt = ChatPromptTemplate.from_template(template)
    return prompt | model

chain = initialize_chain()

# Sidebar for document upload
with st.sidebar:
    st.header("📄 Document Upload")
    uploaded_file = st.file_uploader("Upload a document", type=['pdf', 'txt', 'docx'])
    
    if uploaded_file:
        with st.spinner("Processing document..."):
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                temp_path = tmp_file.name
            
            try:
                # Process the document
                chunks = doc_processor.process_file(temp_path)
                st.success(f"Document processed successfully! ({len(chunks)} chunks created)")
            except Exception as e:
                st.error(f"Error processing document: {str(e)}")
            finally:
                # Clean up temporary file
                os.unlink(temp_path)

# Main chat interface
st.title("🤖 RAG-Enhanced Chatbot")
st.write("Upload documents in the sidebar and chat with the AI. The bot will use the documents to provide informed answers.")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything about your documents..."):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Get relevant document chunks
    relevant_chunks = doc_processor.similarity_search(prompt)
    
    # Get conversation history
    history = "\n".join([
        f"{'User' if msg['role'] == 'user' else 'AI'}: {msg['content']}"
        for msg in st.session_state.messages[:-1]  # Exclude the last user message
    ])
    
    # Generate AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chain.invoke({
                "context": "\n".join(relevant_chunks),
                "history": history,
                "question": prompt
            })
            st.markdown(response)
    
    # Add AI response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response}) 