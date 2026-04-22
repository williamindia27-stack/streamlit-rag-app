from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "I love artificial intelligence",
    "Machine learning is fascinating",
    "Deep learning uses neural networks",
    "The weather is sunny",
    "It might rain tomorrow",
    "The sky is blue"
]

embeddings = model.encode(sentences)

# Cluster into 2 groups
kmeans = KMeans(n_clusters=2, random_state=42)
labels = kmeans.fit_predict(embeddings)

for sentence, label in zip(sentences, labels):
    print(f"Cluster {label}: {sentence}")