# vector_memory.py

import pickle
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import os

# -------- Embedding Model --------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# -------- Load FAISS Index & Chunks --------
faiss_index_path = "vectorstore/faiss_index.index"
chunks_path = "vectorstore/chunks.pkl"

if os.path.exists(faiss_index_path) and os.path.exists(chunks_path):
    index = faiss.read_index(faiss_index_path)
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
else:
    index = None
    chunks = []

# -------- Simple Memory Store --------
memory_store = []

def store_memory(user_input: str, response: str):
    memory_store.append((user_input, response))

def retrieve_memory(user_input: str):
    """
    Retrieve the most relevant past answer based on keyword match.
    """
    for question, answer in reversed(memory_store):
        if user_input.lower() in question.lower():
            return answer
    return None

# -------- RAG Retrieval --------
def retrieve_rag(user_input: str, top_k: int = 3):
    """
    Retrieve top relevant chunks from FAISS vectorstore using embeddings.
    Returns summarized text for the prompt.
    """
    if not index or not chunks:
        return None

    # Embed query
    query_embedding = embedding_model.encode([user_input])
    distances, indices = index.search(np.array(query_embedding), top_k)

    # Collect top chunks
    results = [chunks[i] for i in indices[0] if i < len(chunks)]
    return " ".join(results) if results else None