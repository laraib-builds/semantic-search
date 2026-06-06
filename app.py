from sentence_transformers import SentenceTransformer
import numpy as np
import streamlit as st

# Page setup
st.set_page_config(page_title="Semantic Search", page_icon="🔍")
st.title("🔍 Semantic Word Explorer")
st.caption("Built by Lily — Week 2 of AI Mastery Roadmap")

# Load model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Word bank
words = [
    "king", "queen", "royal", "throne",
    "banana", "fruit", "Pakistan", "Lahore",
    "man", "woman",
    "بادشاہ", "ملکہ", "کیلا", "پھل"
]

embeddings = model.encode(words)

# Similarity function
def similarity(w1, w2):
    idx1 = words.index(w1)
    idx2 = words.index(w2)
    vec1 = embeddings[idx1]
    vec2 = embeddings[idx2]
    score = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return round(float(score), 4)

# Section 1 — Compare two words
st.header("Compare Two Words")
col1, col2 = st.columns(2)
with col1:
    word1 = st.selectbox("Word 1", words)
with col2:
    word2 = st.selectbox("Word 2", words, index=1)

if st.button("Compare"):
    score = similarity(word1, word2)
    st.metric(label="Similarity Score", value=score)
    if score > 0.7:
        st.success("Very similar in meaning")
    elif score > 0.4:
        st.warning("Somewhat related")
    else:
        st.error("Very different in meaning")

# Section 2 — king - man + woman = queen
st.divider()
st.header("👑 king − man + woman = ?")
st.write("Testing the famous word equation:")

if st.button("Run Equation"):
    king = embeddings[words.index("king")]
    man = embeddings[words.index("man")]
    woman = embeddings[words.index("woman")]
    queen = embeddings[words.index("queen")]

    result = king - man + woman
    score = np.dot(result, queen) / (np.linalg.norm(result) * np.linalg.norm(queen))
    score = round(float(score), 4)

    st.metric("Similarity of result to 'queen'", score)
    st.info("The closer to 1.0, the more the math worked.")

# Section 3 — Custom word similarity
st.divider()
st.header("🔎 Try Your Own Words")
custom1 = st.text_input("Enter any word or sentence:", "happy")
custom2 = st.text_input("Compare it to:", "joyful")

if st.button("Check Similarity"):
    vecs = model.encode([custom1, custom2])
    score = np.dot(vecs[0], vecs[1]) / (
        np.linalg.norm(vecs[0]) * np.linalg.norm(vecs[1])
    )
    score = round(float(score), 4)
    st.metric("Similarity Score", score)

st.warning("Note: This model measures topic similarity, not sentiment. Opposites may score high if they share topic words.")