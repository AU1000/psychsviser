import pickle
import faiss
import numpy as np
import pickle
import os
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load text chunks
with open("processed/book_chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# Convert chunks to embeddings
embeddings = model.encode(chunks)

dimension = embeddings.shape[1]

# Create FAISS index
index = faiss.IndexFlatL2(dimension)

index.add(np.array(embeddings))

# Create folder if it doesn't exist
os.makedirs("vectorstore", exist_ok=True)

# Save FAISS index
faiss.write_index(index, "vectorstore/faiss_index.index")

# Save chunks separately
with open("vectorstore/chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

print("Vector database created successfully!")