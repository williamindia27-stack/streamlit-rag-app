from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "I love artificial intelligence",
    "Machine learning is interesting",
    "The weather is sunny"
]

embeddings = model.encode(sentences)

for sentence, embedding in zip(sentences, embeddings):
    print(f"Sentence: {sentence}")
    print(f"Vector length: {len(embedding)}")
    print(f"First 5 values: {embedding[:5]}")
    print()