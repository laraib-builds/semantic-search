from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

sentences = [
    "My name is Lily and I study computer science in Lahore",
    "I am very interested in cybersecurity and AI",
    "My favourite programming language is Python",
    "Today we are learning about context windows and memory",
    "The weather in Lahore is very hot in summer",
    "I want to build AI applications for Pakistani users",
    "Context windows limit how much text a model can see at once",
]

query = "What are my interests and skills?"

# Encode everything
sentence_embeddings = model.encode(sentences)
query_embedding = model.encode([query])[0]

print("\n── Context Window Experiment ──")
print(f"Query: '{query}'\n")

for i, sentence in enumerate(sentences):
    score = np.dot(query_embedding, sentence_embeddings[i]) / (
        np.linalg.norm(query_embedding) * np.linalg.norm(sentence_embeddings[i])
    )
    print(f"[S{i+1}] {sentence[:55]}")
    print(f"      Score: {round(float(score), 4)}\n")