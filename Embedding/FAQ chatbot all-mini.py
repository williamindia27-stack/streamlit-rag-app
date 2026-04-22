from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

# FAQ database
questions = [
    "What is artificial intelligence?",
    "What is machine learning?",
    "What is Python?",
    "What is the weather today?"
]

answers = [
    "Artificial intelligence is the simulation of human intelligence by machines.",
    "Machine learning is a subset of AI that learns from data.",
    "Python is a popular programming language.",
    "The weather is sunny today."
]

# Create embeddings for questions
question_embeddings = model.encode(questions)

# User asks a question
user_question = input("Ask your question: ")

# Convert user question to embedding
user_embedding = model.encode([user_question])

# Compare similarities
scores = cosine_similarity(user_embedding, question_embeddings)

# Find best answer
best_index = np.argmax(scores)

print("\nBest answer:")
print(answers[best_index])
print("Similarity score:", scores[0][best_index])