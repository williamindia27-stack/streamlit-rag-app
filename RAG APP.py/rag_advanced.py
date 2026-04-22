from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------
# Long sample documents
# -------------------------
documents = [
    """Python is a popular programming language.
    It is widely used in artificial intelligence,
    web development, and data science.""",

    """Machine learning is a subset of AI.
    It allows systems to learn from data
    without explicit programming.""",

    """TensorFlow is a deep learning framework.
    It is often used to build neural networks
    and train AI models."""
]

# -------------------------
# Chunking function
# -------------------------
def chunk_text(text, chunk_size=80):
    chunks = []
    words = text.split()

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks

# -------------------------
# Create all chunks
# -------------------------
all_chunks = []

for doc in documents:
    chunks = chunk_text(doc, chunk_size=10)
    all_chunks.extend(chunks)

print("Document chunks:")
for chunk in all_chunks:
    print("-", chunk)

# -------------------------
# Create embeddings
# -------------------------
chunk_embeddings = model.encode(all_chunks)

# -------------------------
# User question
# -------------------------
question = input("\nAsk your question: ")

question_embedding = model.encode([question])

# -------------------------
# Similarity
# -------------------------
scores = cosine_similarity(question_embedding, chunk_embeddings)

# -------------------------
# Top 3 retrieval
# -------------------------
top_3_indices = np.argsort(scores[0])[-3:][::-1]

print("\nTop 3 most relevant chunks:")
for idx in top_3_indices:
    print(f"\nChunk: {all_chunks[idx]}")
    print(f"Score: {scores[0][idx]:.4f}")