import faiss
import pickle
import numpy as np
import os
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi


# ---------- Models ----------
embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

# Cross Encoder for reranking
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


# ---------- Paths ----------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

chunks_path = os.path.join(BASE_DIR, "vectorstore", "chunks.pkl")
index_path = os.path.join(BASE_DIR, "vectorstore", "faiss_index.index")


# ---------- Load Chunks ----------
with open(chunks_path, "rb") as f:
    chunks = pickle.load(f)


# ---------- Load FAISS Index ----------
index = faiss.read_index(index_path)


# ---------- BM25 Setup ----------
tokenized_chunks = [chunk.lower().split() for chunk in chunks]
bm25 = BM25Okapi(tokenized_chunks)


# ---------- Utility ----------
def summarize_text(text, max_words=120):
    words = text.split()
    return " ".join(words[:max_words])


# ---------- Retrieval ----------
def retrieve_law(query, k=5):

    query = query.lower()

    # ----- Vector Search (FAISS) -----
    query_embedding = embedding_model.encode([query]).astype("float32")

    distances, indices = index.search(query_embedding, k)

    vector_results = []
    for i in indices[0]:
        if i != -1 and i < len(chunks):
            vector_results.append(chunks[i])


    # ----- BM25 Search -----
    tokenized_query = query.split()

    bm25_scores = bm25.get_scores(tokenized_query)

    keyword_indices = np.argsort(bm25_scores)[::-1][:k]

    keyword_results = [chunks[i] for i in keyword_indices]


    # ----- Combine Results -----
    combined = list(dict.fromkeys(vector_results + keyword_results))


    # ----- Cross Encoder Reranking -----
    pairs = [[query, chunk] for chunk in combined]

    scores = reranker.predict(pairs)

    ranked_results = sorted(
        zip(combined, scores),
        key=lambda x: x[1],
        reverse=True
    )


    # ----- Select Best Chunks -----
    top_chunks = [chunk for chunk, score in ranked_results[:3]]

    combined_text = " ".join(top_chunks)


    # ----- Limit Length -----
    return summarize_text(combined_text, max_words=120)