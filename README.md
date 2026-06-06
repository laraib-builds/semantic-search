# Semantic Search & Embeddings Explorer

Week 2 project from my 6-month AI learning roadmap.

## What this project does
Uses sentence-transformers to explore how embedding 
models represent meaning — and where they fail.

## Experiments

### 1. Semantic Similarity
Measures how closely related words are in embedding space.
Similar meanings score high. Unrelated meanings score low.

### 2. The Urdu Embedding Gap (Key Finding)
Tested cross-lingual understanding between Urdu and English:

| Pair | Score |
|------|-------|
| banana vs fruit (English) | 0.7261 |
| کیلا vs پھل (Urdu) | 0.6615 |
| banana vs کیلا (cross-lingual) | 0.0844 |

Same concept. Almost zero cross-lingual connection.
Critical finding for anyone building AI apps for Pakistani users.

### 3. Context Window Experiment
Proved that query wording shifts retrieval scores gradually.
Changing the query changes what gets retrieved — 
foundation of how RAG systems work.

## What I Discovered

During testing I found that all-MiniLM-L6-v2 measures 
topic similarity, not sentiment or polarity. Two opposite 
sentences scored higher than two sentences with the same 
meaning — because they shared topic words.

This creates a dangerous failure mode in security systems 
called a false negative — where a threat exists but the 
system says everything is fine. A log line saying 
"login successful" and "login failed" could be treated 
as equally similar to a threat pattern, causing real 
intrusions to go undetected.

False positives waste time. False negatives cost you 
everything.

## Tools
- Python
- sentence-transformers
- numpy
- streamlit

## Run it
pip install sentence-transformers numpy
python semantic_search.py
python context_experiment.py