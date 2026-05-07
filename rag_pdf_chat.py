import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
 
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage

# LOAD .env FILE
# ──────────────────────────────────────────────
load_dotenv()
 
# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(page_title="Chat with PDFs", page_icon="📄")
st.header("📄 Chat with your PDFs")
st.write("Upload a PDF, then ask questions about it")


llm = ChatGroq(
    model="llama-3.1-8b-instant",
    streaming=True,
    api_key=os.getenv("GROQ_API_KEY")
)

@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
 
embeddings = get_embedding_model()

# PDF UPLOAD & PROCESSING
# ──────────────────────────────────────────────
st.sidebar.header("Upload PDF")
uploaded_file = st.sidebar.file_uploader(
    "Choose a PDF file",
    type="pdf",
    accept_multiple_files=False,
)

# This function does the RAG pipeline:
# PDF → Pages → Chunks → Embeddings → Vector Store
@st.cache_resource
def process_pdf(_uploaded_file):
    """
    Step 1: LOAD   - Read the PDF, extract text from each page
    Step 2: SPLIT  - Break text into small overlapping chunks
    Step 3: EMBED  - Convert each chunk into a vector (numbers)
    Step 4: STORE  - Save vectors in FAISS for fast searching
    """
 
    # Step 1: Load PDF
    # Save uploaded file to a temp location so PyPDFLoader can read it
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(_uploaded_file.getvalue())
        tmp_path = tmp.name
 
    loader = PyPDFLoader(tmp_path)
    pages = loader.load()  # Returns list of Document objects (one per page)
 
    # Step 2: Split into chunks
    # - chunk_size=1000: each chunk is ~1000 characters
    # - chunk_overlap=200: chunks overlap by 200 chars so we don't cut sentences
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(pages)
 
    # Step 3 & 4: Embed chunks and store in FAISS
    vector_store = FAISS.from_documents(chunks, embeddings)
 
    # Clean up temp file
    os.unlink(tmp_path)
 
    return vector_store
if uploaded_file:
    with st.sidebar:
        with st.spinner("Processing PDF..."):
            vector_store = process_pdf(uploaded_file)
        st.success(f"PDF ready! Ask questions below.")
 
    # ──────────────────────────────────────────────
    # RAG CHAIN
    # ──────────────────────────────────────────────
    # The retriever searches the vector store for relevant chunks
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},  # return top 4 matching chunks
    )
 
    # The prompt tells the LLM: answer using ONLY the provided context
    rag_prompt = ChatPromptTemplate.from_template(
        """You are a helpful assistant. Answer the question based ONLY on the
following context from the uploaded PDF. If you cannot find the answer
in the context, say "I couldn't find that information in the PDF."
 
Context:
{context}

Chat History:
{chat_history}

 
Question: {question}
 
Answer:"""
    )

    # Helper: converts Document objects to a single string
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    # Helper: converts chat history to a readable string for the LLM
    def get_chat_history():
        history = ""
        for msg in st.session_state.get("rag_messages", [])[1:]:  # skip welcome msg
            role = "Human" if msg["role"] == "user" else "Assistant"
            history += f"{role}: {msg['content']}\n"
        return history if history else "No previous conversation."

    # The RAG chain: retrieve → format → prompt → LLM → parse output
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough(), "chat_history": lambda _: get_chat_history()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    # CHAT HISTORY
    # ──────────────────────────────────────────────
    if "rag_messages" not in st.session_state:
        st.session_state.rag_messages = [
            {"role": "assistant", "content": "PDF uploaded! Ask me anything about it."}
        ]
 
    for msg in st.session_state.rag_messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # HANDLE USER INPUT
    # ──────────────────────────────────────────────
    if user_query := st.chat_input("Ask about your PDF..."):
 
        # 1. Show user message
        st.session_state.rag_messages.append({"role": "user", "content": user_query})
        st.chat_message("user").write(user_query)
 
        # 2. Get streaming response from RAG chain
        with st.chat_message("assistant"):
            response = st.write_stream(
                rag_chain.stream(user_query)
            )
 
        # 3. Save response
        st.session_state.rag_messages.append({"role": "assistant", "content": response})
 
else:
    st.info("👈 Upload a PDF from the sidebar to get started.")
