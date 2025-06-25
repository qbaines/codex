"""Example script showing how to index and search code snippets using
OpenAI embeddings with FAISS."""

from openai import OpenAIEmbeddings
import faiss
import numpy as np


def build_index(texts):
    """Create a FAISS index from a list of strings."""
    embeddings = OpenAIEmbeddings().embed_documents(texts)
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index


def query_index(index, texts, query, k=3):
    """Return the top `k` matching texts for a given query."""
    query_vec = OpenAIEmbeddings().embed_query(query)
    distances, indices = index.search(np.array([query_vec]), k=k)
    return [texts[i] for i in indices[0]]


if __name__ == "__main__":
    snippets = [
        "def connect_db(): ...",
        "def send_email(): ...",
    ]
    idx = build_index(snippets)
    result = query_index(idx, snippets, "database connection")
    print(result)
