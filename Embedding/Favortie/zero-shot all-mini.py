from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


model = SentenceTransformer("all-MiniLM-L6-v2")

# Predefined categories
categories = [
    "technology",
    "weather",
    "sports",
    "finance"
]

# Create embeddings for category labels
category_embeddings = model.encode(categories)

# User input
text = input("Enter a sentence to classify: ")

# Embed input sentence
text_embedding = model.encode([text])

# Compare with category embeddings
scores = cosine_similarity(text_embedding, category_embeddings)

# Find best category
best_index = np.argmax(scores)

print("\nPredicted category:")
print(categories[best_index])
print("Similarity score:", scores[0][best_index])