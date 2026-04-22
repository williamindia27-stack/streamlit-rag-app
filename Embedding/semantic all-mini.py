from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Knowledge base
documents = [
    "Artificial intelligence is transforming technology.",
    "Machine learning is a subset of AI.",
    "The weather is sunny today.",
    "Python is a popular programming language.",
    "Deep learning uses neural networks."
]

# Create embeddings for all documents
doc_embeddings = model.encode(documents)

# User query
query = input("Enter your search query: ")

# Embed the query
query_embedding = model.encode([query])

# Compute similarity
similarities = cosine_similarity(query_embedding, doc_embeddings)

# Find best match
best_index = np.argmax(similarities)

print("\nMost relevant result:")
print(documents[best_index])
print("Similarity score:", similarities[0][best_index])