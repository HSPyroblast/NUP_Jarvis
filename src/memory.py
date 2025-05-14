import os
import faiss
import pickle
import numpy as np

from sentence_transformers import SentenceTransformer
from logger import logger


MEMORY_FILE = "../data/memory.pkl"
model = SentenceTransformer("all-MiniLM-L6-v2")
dimension = 384
index = faiss.IndexFlatL2(dimension)

memory_texts = []
memory_vectors = []


def save_memory():
    with open(MEMORY_FILE, "wb") as f:
        pickle.dump((memory_texts, memory_vectors), f)
    logger.info("Memory saved.")

def load_memory():
    global memory_texts, memory_vectors, index
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "rb") as f:
            memory_texts[:], memory_vectors[:] = pickle.load(f)
            if memory_vectors:
                index.add(np.array(memory_vectors))
        logger.info(f"Loaded {len(memory_texts)} items from memory.")
    else:
        logger.info("No saved memory found.")

def add_to_memory(text: str):
    embedding = model.encode([text])[0]
    memory_texts.append(text)
    memory_vectors.append(embedding)
    index.add(np.array([embedding]))
    logger.info(f"Added to memory: {text}")
    save_memory()

def retrieve_memory(query: str, top_k: int = 3):
    if len(memory_vectors) == 0:
        return []

    embedding = model.encode([query])[0]
    D, I = index.search(np.array([embedding]), top_k)
    return [memory_texts[i] for i in I[0] if i < len(memory_texts)]

def clear_memory():
    global memory_texts, memory_vectors, index
    memory_texts = []
    memory_vectors = []
    index.reset()

    if os.path.exists(MEMORY_FILE):
        os.remove(MEMORY_FILE)
    logger.info("Memory cleared.")
