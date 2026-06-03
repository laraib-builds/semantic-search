from sentence_transformers import SentenceTransformer
import numpy as np

# Load a fee, small embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Our words to compare
words = [
    "king",
    "queen",
    "royal",
    "throne",
    "banana",
    "fruit",
    "Pakistan",
    "Lahore",
    "man",
    "woman",
    "بادشاہ",  # king in Urdu
    "ملکہ",    # queen in Urdu
    "کیلا",    # banana in Urdu
    "پھل"      # fruit in Urdu
]

# Convert words to embeddings
embeddings = model.encode(words)

# Function to measure similarity between two words
def similarity(word1, word2):
    idx1 = words.index(word1)
    idx2 = words.index(word2)
    vec1 = embeddings[idx1]
    vec2 = embeddings[idx2]
    score = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return round(float(score), 4)

# Testing king - man + woman = queen
king = embeddings[words.index("king")]
man = embeddings[words.index("man")]
woman = embeddings[words.index("woman")]
queen = embeddings[words.index("queen")]

# The famous equation
result = king - man + woman

# How similar is the result to queen?
score = np.dot(result, queen) / (np.linalg.norm(result) * np.linalg.norm(queen))
print("\nking - man + woman similarity to queen:", round(float(score), 4))

# Compare pairs
print("king vs queen:", similarity("king", "queen"))
print("king vs banana:", similarity("king", "banana"))
print("king vs royal:", similarity("king", "royal"))
print("banana vs fruit:", similarity("banana", "fruit"))
print("Pakistan vs Lahore:", similarity("Pakistan", "Lahore"))
print("man vs woman:", similarity("man", "woman"))
print("king vs man:", similarity("king", "man"))
print("queen vs woman:", similarity("queen", "woman"))
print("\nبادشاہ vs ملکہ:", similarity("بادشاہ", "ملکہ"))
print("کیلا vs پھل:", similarity("کیلا", "پھل"))
print("\nking vs بادشاہ:", similarity("king", "بادشاہ"))
print("queen vs ملکہ:", similarity("queen", "ملکہ"))
print("banana vs کیلا:", similarity("banana", "کیلا"))