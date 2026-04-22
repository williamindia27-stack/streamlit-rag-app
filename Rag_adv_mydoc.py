# =============================
# Mini ChatGPT over your documents (RAG)
# =============================

import streamlit as st
import os

from llama_index.core import VectorStoreIndex, Settings, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.llms import MockLLM

# -----------------------------
# CONFIG
# -----------------------------

Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Use MockLLM since no OpenAI API key
Settings.llm = MockLLM()


# -----------------------------
# LOAD + INDEX DOCUMENTS
# -----------------------------

@st.cache_resource
def load_index():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "data")

    if not os.path.exists(data_path):
        st.error("❌ 'data' folder not found.")
        return None

    files = os.listdir(data_path)
    if len(files) == 0:
        st.error("❌ No files inside 'data' folder.")
        return None

    # Filter for PDF files and create full paths
    pdf_files = [os.path.join(data_path, f) for f in files if f.lower().endswith('.pdf')]
    if len(pdf_files) == 0:
        st.error("❌ No PDF files found in 'data' folder.")
        return None

    print("Loading documents...")
    documents = SimpleDirectoryReader(input_files=pdf_files).load_data()
    print(f"Loaded {len(documents)} documents")

    parser = SentenceSplitter(chunk_size=500, chunk_overlap=50)  # 👈 changed
    nodes = parser.get_nodes_from_documents(documents)
    print(f"Created {len(nodes)} nodes")

    print("Creating index...")
    index = VectorStoreIndex(nodes)
    print("Index created")
    return index


index = load_index()

# 👉 Prevent crash if no index
if index is not None:
    query_engine = index.as_query_engine(similarity_top_k=3)
else:
    query_engine = None


# -----------------------------
# STREAMLIT UI
# -----------------------------

st.title("📄 Mini ChatGPT over your documents")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("Ask something about your documents...")

if user_input and query_engine:
    st.session_state.chat_history.append(("user", user_input))

    response = query_engine.query(user_input)
    # Summarize by taking only the first 500 characters
    summary = str(response)[:500] + "..." if len(str(response)) > 500 else str(response)

    st.session_state.chat_history.append(("assistant", summary))

for role, message in st.session_state.chat_history:
    with st.chat_message(role):
        st.write(message)


# -----------------------------
# SOURCES DEBUG
# -----------------------------

st.sidebar.title("🔎 Debug / Sources")

if st.sidebar.button("Show last sources") and len(st.session_state.chat_history) >= 2:
    last_query = st.session_state.chat_history[-2][1]

    retriever = index.as_retriever(similarity_top_k=3)
    results = retriever.retrieve(last_query)

    for i, node in enumerate(results, start=1):
        st.sidebar.write(f"Result {i}:")
        st.sidebar.write(node.text)
        st.sidebar.write("---")





