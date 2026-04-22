# =============================
# Mini ChatGPT over your documents (RAG)
# =============================

import streamlit as st
import os
import torch

from llama_index.core import VectorStoreIndex, Settings, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.groq import Groq
from llama_index.core import PromptTemplate

# -----------------------------
# GROQ LLM HELPER
# -----------------------------

def get_llm():
    return Groq(
        model="llama-3.1-8b-instant",  # updated from llama3-8b-8192
        api_key=st.secrets["GROQ_API_KEY"]
    )
    

# -----------------------------
# CUSTOM PROMPT
# -----------------------------

CUSTOM_PROMPT = PromptTemplate(
    "You are a helpful assistant that answers questions based on the provided documents.\n"
    "Always answer in clear, concise, well-structured paragraphs.\n"
    "Use plain language. Avoid bullet points unless listing is truly necessary.\n"
    "If the answer is not in the documents, say so honestly.\n\n"
    "Context:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n\n"
    "Question: {query_str}\n\n"
    "Answer:"
)

# -----------------------------
# LOAD + INDEX DOCUMENTS
# -----------------------------

@st.cache_resource(show_spinner="Loading documents...")
def load_index(_version=3):
    Settings.llm = get_llm()

    Settings.embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",
        device="cpu",
        model_kwargs={"torch_dtype": torch.float32}
    )

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "data")

    if not os.path.exists(data_path):
        st.error("❌ 'data' folder not found.")
        return None

    files = os.listdir(data_path)
    if len(files) == 0:
        st.error("❌ No files inside 'data' folder.")
        return None

    pdf_files = [os.path.join(data_path, f) for f in files if f.lower().endswith('.pdf')]
    if len(pdf_files) == 0:
        st.error("❌ No PDF files found in 'data' folder.")
        return None

    documents = SimpleDirectoryReader(input_files=pdf_files).load_data()
    parser = SentenceSplitter(chunk_size=600, chunk_overlap=50)
    nodes = parser.get_nodes_from_documents(documents)
    index = VectorStoreIndex(nodes)
    return index


index = load_index()

Settings.llm = get_llm()

if index is not None:
    query_engine = index.as_query_engine(
        similarity_top_k=3,
        llm=get_llm(),
        text_qa_template=CUSTOM_PROMPT
    )
else:
    query_engine = None


# -----------------------------
# STREAMLIT UI
# -----------------------------

st.title("📄 My RAG")
st.caption("Ask questions about your documents and get clear, concise answers.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("Ask something about your documents...")

if user_input and query_engine:
    st.session_state.chat_history.append(("user", user_input))

    with st.spinner("Thinking..."):
        response = query_engine.query(user_input)
        answer = str(response).strip()

    st.session_state.chat_history.append(("assistant", answer))

for role, message in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(message)


# -----------------------------
# SOURCES DEBUG
# -----------------------------

st.sidebar.title("🔎 Debug / Sources")

if st.sidebar.button("Show last sources") and len(st.session_state.chat_history) >= 2:
    last_query = st.session_state.chat_history[-2][1]
    retriever = index.as_retriever(similarity_top_k=3)
    results = retriever.retrieve(last_query)

    for i, node in enumerate(results, start=1):
        st.sidebar.write(f"**Result {i}:**")
        st.sidebar.write(node.text)
        st.sidebar.write("---") 