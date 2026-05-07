import os
import glob
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.document_loaders import (
    PyPDFLoader,
    CSVLoader,
    TextLoader,
    Docx2txtLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ──────────────────────────────────────────────
# LOAD .env
# ──────────────────────────────────────────────
load_dotenv()

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(page_title="Knowledge Base Chat", page_icon="📚")
st.header("📚 Enterprise Knowledge Base")
st.write("Chat with all your documents — PDFs, CSVs, TXT, DOCX")

# ──────────────────────────────────────────────
# API KEY
# ──────────────────────────────────────────────
api_key = os.environ.get("GROQ_API_KEY", "")

if not api_key:
    api_key = st.sidebar.text_input(
        "Groq API Key", type="password", placeholder="gsk_..."
    )

if not api_key:
    st.info("Please enter your Groq API key to start.")
    st.stop()

# ──────────────────────────────────────────────
# LLM
# ──────────────────────────────────────────────
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    streaming=True,
    api_key=api_key,
)

# ──────────────────────────────────────────────
# EMBEDDING MODEL (runs locally, free)
# ──────────────────────────────────────────────
@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

embeddings = get_embedding_model()

# ──────────────────────────────────────────────
# FILE LOADER - picks the right loader per file type
# ──────────────────────────────────────────────
def load_single_file(file_path):
    """Load a single file based on its extension."""
    ext = os.path.splitext(file_path)[1].lower()

    try:
        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif ext == ".csv":
            loader = CSVLoader(file_path, encoding="utf-8")
        elif ext in [".txt", ".md"]:
            loader = TextLoader(file_path, encoding="utf-8")
        elif ext in [".docx"]:
            loader = Docx2txtLoader(file_path)
        else:
            return []  # skip unsupported files

        return loader.load()
    except Exception as e:
        st.sidebar.warning(f"Could not load {os.path.basename(file_path)}: {e}")
        return []


# ──────────────────────────────────────────────
# PROCESS ALL FILES FROM FOLDER
# ──────────────────────────────────────────────
@st.cache_resource
def build_knowledge_base(folder_path):
    """
    1. Find all supported files in the folder
    2. Load each file
    3. Split into chunks
    4. Embed and store in FAISS
    """

    # Step 1: Find all files
    supported = ["*.pdf", "*.csv", "*.txt", "*.md", "*.docx"]
    all_files = []
    for pattern in supported:
        all_files.extend(glob.glob(os.path.join(folder_path, pattern)))
        # Also search subfolders
        all_files.extend(glob.glob(os.path.join(folder_path, "**", pattern), recursive=True))

    # Remove duplicates
    all_files = list(set(all_files))

    if not all_files:
        return None, []

    # Step 2: Load all files
    all_docs = []
    loaded_files = []
    for file_path in all_files:
        docs = load_single_file(file_path)
        if docs:
            all_docs.extend(docs)
            loaded_files.append(os.path.basename(file_path))

    if not all_docs:
        return None, []

    # Step 3: Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(all_docs)

    # Step 4: Embed and store
    vector_store = FAISS.from_documents(chunks, embeddings)

    return vector_store, loaded_files


# ──────────────────────────────────────────────
# SIDEBAR - FOLDER SELECTION
# ──────────────────────────────────────────────
st.sidebar.header("📁 Document Folder")

# Default folder
default_folder = "data"
folder_path = st.sidebar.text_input(
    "Folder path",
    value=default_folder,
    help="Path to folder containing your documents"
)

# Also allow file uploads as alternative
st.sidebar.divider()
st.sidebar.header("📤 Or Upload Files")
uploaded_files = st.sidebar.file_uploader(
    "Upload documents",
    type=["pdf", "csv", "txt", "docx"],
    accept_multiple_files=True,
)

# Save uploaded files to the data folder
if uploaded_files:
    os.makedirs(folder_path, exist_ok=True)
    for uploaded_file in uploaded_files:
        file_path = os.path.join(folder_path, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    st.sidebar.success(f"Saved {len(uploaded_files)} file(s) to {folder_path}/")

# ──────────────────────────────────────────────
# BUILD KNOWLEDGE BASE
# ──────────────────────────────────────────────
if os.path.exists(folder_path):
    with st.sidebar:
        with st.spinner("Loading documents..."):
            vector_store, loaded_files = build_knowledge_base(folder_path)

    if vector_store and loaded_files:
        # Show loaded files
        st.sidebar.divider()
        st.sidebar.header("✅ Loaded Files")
        for f in loaded_files:
            st.sidebar.write(f"📄 {f}")
        st.sidebar.write(f"**Total: {len(loaded_files)} files**")

        # ──────────────────────────────────────────────
        # RAG CHAIN
        # ──────────────────────────────────────────────
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5},
        )

        rag_prompt = ChatPromptTemplate.from_template(
            """You are a helpful knowledge base assistant. Answer the question based
ONLY on the following context from the loaded documents. If the answer is not
in the context, say "I couldn't find that in the documents."

When possible, mention which document or source the information came from.

Context:
{context}

Chat History:
{chat_history}

Question: {question}

Answer:"""
        )

        def format_docs(docs):
            formatted = []
            for doc in docs:
                source = doc.metadata.get("source", "unknown")
                source = os.path.basename(source)
                formatted.append(f"[From: {source}]\n{doc.page_content}")
            return "\n\n".join(formatted)

        def get_chat_history():
            history = ""
            for msg in st.session_state.get("kb_messages", [])[1:]:
                role = "Human" if msg["role"] == "user" else "Assistant"
                history += f"{role}: {msg['content']}\n"
            return history if history else "No previous conversation."

        rag_chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough(),
                "chat_history": lambda _: get_chat_history(),
            }
            | rag_prompt
            | llm
            | StrOutputParser()
        )

        # ──────────────────────────────────────────────
        # CHAT HISTORY
        # ──────────────────────────────────────────────
        if "kb_messages" not in st.session_state:
            st.session_state.kb_messages = [
                {"role": "assistant", "content": f"Knowledge base ready! I've loaded {len(loaded_files)} documents. Ask me anything about them."}
            ]

        for msg in st.session_state.kb_messages:
            st.chat_message(msg["role"]).write(msg["content"])

        # ──────────────────────────────────────────────
        # HANDLE USER INPUT
        # ──────────────────────────────────────────────
        if user_query := st.chat_input("Ask about your documents..."):

            st.session_state.kb_messages.append({"role": "user", "content": user_query})
            st.chat_message("user").write(user_query)

            with st.chat_message("assistant"):
                try:
                    response = st.write_stream(rag_chain.stream(user_query))
                except Exception as e:
                    response = f"Error: {str(e)[:200]}"
                    st.write(response)

            st.session_state.kb_messages.append({"role": "assistant", "content": response})

    else:
        st.info(f"No supported files found in '{folder_path}/'. Add PDFs, CSVs, TXT, or DOCX files.")
else:
    st.info(f"Folder '{folder_path}' not found. Create it and add your documents, or upload files using the sidebar.")
