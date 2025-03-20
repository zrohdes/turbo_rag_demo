import streamlit as st
import os
from pathlib import Path
import tempfile
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI

# Set page configuration
st.set_page_config(
    page_title="Turbine Maintenance Assistant",
    page_icon="🔧",
    layout="wide",
)

# UI Elements
st.title("🔧 Turbine Maintenance Assistant")
st.markdown("""
This chatbot helps with wind and gas turbine maintenance and troubleshooting by using a knowledge base of technical documents.
""")

# Sidebar for document upload
with st.sidebar:
    st.header("Knowledge Base")
    uploaded_files = st.file_uploader(
        "Upload maintenance manuals and technical documents",
        accept_multiple_files=True,
        type=["pdf", "docx", "txt"]
    )

    process_docs = st.button("Process Documents")

    st.markdown("---")
    st.markdown("## About")
    st.markdown("""
    This RAG (Retrieval-Augmented Generation) chatbot uses:
    - Google's Gemini 1.5 Flash for generation
    - FAISS vector database for document retrieval
    - LangChain for the conversation chain
    """)

# Initialize session state variables
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processed_docs" not in st.session_state:
    st.session_state.processed_docs = False


# Function to load and process documents
def process_documents(uploaded_files):
    # Get API key from secrets
    try:
        api_key = st.secrets["api_keys"]["google"]
        if not api_key:
            st.error("Google API key not found in secrets. Please add it to .streamlit/secrets.toml")
            return None
    except Exception as e:
        st.error(f"Error accessing API key from secrets: {e}")
        st.info("Make sure you've added your Google API key to .streamlit/secrets.toml")
        return None

    # Create a temporary directory to store uploaded files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save uploaded files to the temporary directory
        temp_paths = []
        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            temp_paths.append(file_path)

        # Load documents based on file type
        documents = []
        for temp_path in temp_paths:
            file_extension = os.path.splitext(temp_path)[1].lower()
            if file_extension == ".pdf":
                loader = PyPDFLoader(temp_path)
            elif file_extension == ".docx":
                loader = Docx2txtLoader(temp_path)
            elif file_extension == ".txt":
                loader = TextLoader(temp_path)
            else:
                st.warning(f"Unsupported file type: {file_extension}")
                continue

            documents.extend(loader.load())

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200,
        )
        document_chunks = text_splitter.split_documents(documents)

        # Configure API
        os.environ["GOOGLE_API_KEY"] = api_key
        genai.configure(api_key=api_key)

        # Create embeddings and vector store
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001"
        )

        vectorstore = FAISS.from_documents(document_chunks, embeddings)

        return vectorstore


# Process uploaded documents when the button is clicked
if process_docs and uploaded_files:
    with st.spinner("Processing documents..."):
        vectorstore = process_documents(uploaded_files)
        if vectorstore:
            st.session_state.vectorstore = vectorstore
            st.session_state.processed_docs = True
            st.success("Documents processed successfully!")


# Create conversational retrieval chain
def create_conversation_chain(vectorstore):
    try:
        api_key = st.secrets["api_keys"]["google"]
        if not api_key:
            st.error("Google API key not found in secrets")
            return None

        # Configure API
        os.environ["GOOGLE_API_KEY"] = api_key
        genai.configure(api_key=api_key)

        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.2,
            safety_settings={
                "HARASSMENT": "block_none",
                "HATE": "block_none",
                "SEXUALLY_EXPLICIT": "block_none",
                "DANGEROUS": "block_none",
            }
        )

        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
            return_source_documents=True,
            verbose=False,
        )

        return conversation_chain
    except Exception as e:
        st.error(f"Error creating conversation chain: {e}")
        return None


# Initialize conversation chain if documents are processed
if st.session_state.processed_docs and "vectorstore" in st.session_state and not st.session_state.conversation:
    st.session_state.conversation = create_conversation_chain(st.session_state.vectorstore)

# Chat interface
chat_container = st.container()
with chat_container:
    # Display chat history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.chat_message("user").write(message["content"])
        else:
            st.chat_message("assistant").write(message["content"])

    # Input for new questions
    if user_query := st.chat_input("Ask about turbine maintenance and troubleshooting..."):
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        st.chat_message("user").write(user_query)

        if not st.session_state.processed_docs:
            st.warning("Please upload and process documents first.")
        elif not st.session_state.conversation:
            st.error("Conversation chain not initialized. Check API key and try processing documents again.")
        else:
            with st.spinner("Thinking..."):
                # Format chat history for LangChain format
                formatted_chat_history = []
                for message in st.session_state.chat_history[:-1]:  # Exclude the current query
                    if message["role"] == "user":
                        formatted_chat_history.append((message["content"], ""))
                    else:
                        # Update the last tuple with the assistant's response
                        if formatted_chat_history:
                            last_human, _ = formatted_chat_history[-1]
                            formatted_chat_history[-1] = (last_human, message["content"])

                # Get response from conversation chain
                response = st.session_state.conversation({
                    "question": user_query,
                    "chat_history": formatted_chat_history
                })

                # Display response
                ai_response = response["answer"]
                st.chat_message("assistant").write(ai_response)

                # Optional: Display source documents
                with st.expander("Source Documents"):
                    for i, doc in enumerate(response["source_documents"]):
                        st.markdown(f"**Source {i + 1}**")
                        st.markdown(f"*Content:* {doc.page_content}")
                        st.markdown(f"*Source:* {doc.metadata.get('source', 'Unknown')}")
                        st.markdown("---")

                # Add response to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": ai_response})

# Instructions for getting started
if not st.session_state.processed_docs:
    st.info("👈 To get started, upload your turbine maintenance documents in the sidebar and click 'Process Documents'")

# Info note
st.sidebar.markdown("---")
st.sidebar.caption("ℹ️ This application uses the Google API key from .streamlit/secrets.toml")

# Add clear conversation button
if st.sidebar.button("Clear Conversation"):
    st.session_state.chat_history = []
    st.experimental_rerun()

# Deployment instructions in the sidebar
with st.sidebar.expander("Deployment Guide"):
    st.markdown("""
    ### Deployment on Streamlit Cloud

    1. Push this code to a GitHub repository
    2. Create a `.streamlit/secrets.toml` file with:
       ```
       [api_keys]
       google = "your-api-key"  # For testing only
       ```
    3. Go to [streamlit.io](https://streamlit.io)
    4. Create a new app linked to your GitHub repo
    5. Set the API key as a secret in the Streamlit dashboard

    ### Local Development
    ```bash
    pip install -r requirements.txt
    streamlit run app.py
    ```
    """)