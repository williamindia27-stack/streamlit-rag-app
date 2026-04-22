from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Job description
job_description = """
We are looking for a Python developer with knowledge in machine learning,
data analysis, and TensorFlow.
"""

# Candidate profile / CV
candidate_profile = input("Paste candidate profile: ")

# Create embeddings
job_embedding = model.encode([job_description])
candidate_embedding = model.encode([candidate_profile])

# Compare similarity
score = cosine_similarity(job_embedding, candidate_embedding)[0][0]

print(f"\nMatch score: {score:.4f}")

if score > 0.75:
    print("Excellent match")
elif score > 0.5:
    print("Potential match")
else:
    print("Low match")