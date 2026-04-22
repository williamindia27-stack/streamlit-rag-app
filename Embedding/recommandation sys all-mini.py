from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

items = [
    "Python programming course",
    "Machine learning course",
    "Football tactics book",
    "Cryptocurrency investing guide",
    "Weather forecasting article"
]

# Create embeddings for items
item_embeddings = model.encode(items)

# User interest
interest = input("What are you interested in? ")

# Convert interest to vector
interest_embedding = model.encode([interest])

# Compare similarities
scores = cosine_similarity(interest_embedding, item_embeddings)

# Get top 3 recommendations
top_indices = np.argsort(scores[0])[-3:][::-1]

print("\nTop recommendations:")
for i in top_indices:
    print(f"- {items[i]} (score: {scores[0][i]:.4f})")