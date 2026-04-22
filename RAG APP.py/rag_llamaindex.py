from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Set local embedding model
Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

documents = [
    Document(text="Python is a programming language used for AI and web development."),
    Document(text="Machine learning is a subset of artificial intelligence."),
    Document(text="TensorFlow is a framework used for deep learning."),
    Document(text="Embeddings convert text into vectors."),
    Document(text="RAG means Retrieval Augmented Generation.")
]

index = VectorStoreIndex.from_documents(documents)

retriever = index.as_retriever(similarity_top_k=3)

query = input("Ask your question: ")

results = retriever.retrieve(query)

print("\nTop 3 retrieved chunks:")
for i, node in enumerate(results, start=1):
    print(f"\nResult {i}:")
    print(node.text)