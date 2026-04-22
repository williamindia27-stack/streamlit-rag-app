from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")

text1 = input("Enter first paragraph: ")

text2 = input("Enter second paragraph: ")

embedding1 = model.encode([text1])
embedding2 = model.encode([text2])

# Compute similarity
score = cosine_similarity(embedding1, embedding2)[0][0]

print(f"\nSimilarity score: {score:.4f}")

# Decision
if score > 0.85:
    print("Possible plagiarism / near duplicate")
elif score > 0.6:
    print("Texts are related")
else:
    print("Texts are different")