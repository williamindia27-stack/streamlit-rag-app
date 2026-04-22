from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

# Example reference messages
labels = ["spam", "normal"]

examples = [
    "Congratulations! You won $1000, click here now!",
    "Hello, can we meet tomorrow for the project?"
]

# Create embeddings for examples
example_embeddings = model.encode(examples)

# User message
message = input("Enter the message to analyze: ")

# Convert user message
message_embedding = model.encode([message])

# Compute similarity
scores = cosine_similarity(message_embedding, example_embeddings)

# Best label
best_index = np.argmax(scores)

print("\nPrediction:")
print(labels[best_index])
print("Similarity score:", scores[0][best_index])