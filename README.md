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

## Tools
- Python
- sentence-transformers
- numpy

## Run it
pip install sentence-transformers numpy
python semantic_search.py
python context_experiment.py