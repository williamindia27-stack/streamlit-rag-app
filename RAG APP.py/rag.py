from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Sample documents
documents = [
    "Python is a programming language used for AI and web development.",
    "Machine learning is a subset of artificial intelligence.",
    "TensorFlow is a framework used for deep learning.",
    "Embeddings convert text into vectors.",
    "RAG means Retrieval Augmented Generation."
]

# Convert documents into embeddings
doc_embeddings = model.encode(documents)

# Ask question
question = input("Ask a question: ")

# Convert question into embedding
question_embedding = model.encode([question])

# Compute similarity
scores = cosine_similarity(question_embedding, doc_embeddings)

# Find best answer
best_index = np.argmax(scores)

print("\nBest answer:")
print(documents[best_index])

print("Similarity score:", scores[0][best_index])