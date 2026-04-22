from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")

sentence1 = input("Enter first sentence: ")
sentence2 = input("Enter second sentence: ")

embedding1 = model.encode([sentence1])
embedding2 = model.encode([sentence2])

# Compute similarity
score = cosine_similarity(embedding1, embedding2)[0][0]

# Print result
print(f"\nSimilarity score: {score:.4f}")

if score > 0.8:
    print("These sentences are very similar.")
elif score > 0.5:
    print("These sentences are somewhat similar.")
else:
    print("These sentences are different.")