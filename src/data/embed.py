"""
Phase 1 — Embedding Pipeline
Chunks processed articles and stores them in a local ChromaDB vector store.
Uses Ollama's nomic-embed-text for 100% local embeddings (no API key needed).
"""

import json
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

PROCESSED_DIR = Path("data/processed")
EMBED_DIR = Path("data/embeddings")

CHUNK_SIZE = 300      # characters per chunk
CHUNK_OVERLAP = 50    # overlap between chunks
EMBED_MODEL = "nomic-embed-text"
OLLAMA_URL = "http://localhost:11434"
COLLECTION_NAME = "knowledge"


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping character-level chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def build_vector_store():
    EMBED_DIR.mkdir(parents=True, exist_ok=True)

    # Local Ollama embedding function — no API key required
    embed_fn = OllamaEmbeddingFunction(
        url=f"{OLLAMA_URL}/api/embeddings",
        model_name=EMBED_MODEL,
    )

    client = chromadb.PersistentClient(path=str(EMBED_DIR))

    # Delete and recreate so re-running is idempotent
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
    )

    articles = json.loads((PROCESSED_DIR / "articles_clean.json").read_text())

    documents, metadatas, ids = [], [], []
    for article in articles:
        chunks = chunk_text(article["content"])
        for i, chunk in enumerate(chunks):
            chunk_id = f"{article['id']}_chunk_{i}"
            documents.append(chunk)
            metadatas.append(
                {
                    "article_id": article["id"],
                    "title": article["title"],
                    "tags": ", ".join(article.get("tags", [])),
                    "chunk_index": i,
                }
            )
            ids.append(chunk_id)

    # Batch upsert
    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    print(f"✅ Embedded {len(documents)} chunks from {len(articles)} articles → {EMBED_DIR}")


def query_vector_store(query: str, n_results: int = 3) -> list[dict]:
    """Utility: search the vector store (used by agent tools)."""
    embed_fn = OllamaEmbeddingFunction(
        url=f"{OLLAMA_URL}/api/embeddings",
        model_name=EMBED_MODEL,
    )
    client = chromadb.PersistentClient(path=str(EMBED_DIR))
    collection = client.get_collection(COLLECTION_NAME, embedding_function=embed_fn)
    results = collection.query(query_texts=[query], n_results=n_results)
    return [
        {"text": doc, "metadata": meta}
        for doc, meta in zip(results["documents"][0], results["metadatas"][0])
    ]


if __name__ == "__main__":
    build_vector_store()

    # Quick smoke test
    print("\n🔍 Smoke test — querying 'What is RAG?':")
    hits = query_vector_store("What is RAG?")
    for h in hits:
        print(f"  [{h['metadata']['title']}] {h['text'][:120]}...")
