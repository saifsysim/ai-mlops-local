"""
Phase 2 — MCP Server
Exposes local tools via the Model Context Protocol (FastMCP).
Any MCP-compatible client (Claude Desktop, Cursor, etc.) can connect to this server.

Run with:  python src/mcp/server.py
"""

import json
from pathlib import Path

from fastmcp import FastMCP

mcp = FastMCP(
    name="AI MLOps Local Knowledge Server",
    version="1.0.0",
    description="Exposes local knowledge base and data tools via MCP",
)


# ── Tool 1: Search the local vector knowledge base ────────────────────────────
@mcp.tool()
def search_knowledge(query: str, n_results: int = 3) -> str:
    """
    Search the local ChromaDB knowledge base for articles relevant to the query.
    Returns the top matching text chunks with their source titles.
    """
    try:
        import chromadb
        from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

        embed_fn = OllamaEmbeddingFunction(
            url="http://localhost:11434/api/embeddings",
            model_name="nomic-embed-text",
        )
        client = chromadb.PersistentClient(path="data/embeddings")
        collection = client.get_collection("knowledge", embedding_function=embed_fn)
        results = collection.query(query_texts=[query], n_results=n_results)

        output_parts = []
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            output_parts.append(f"[{meta['title']}]\n{doc}")
        return "\n\n---\n\n".join(output_parts)
    except Exception as e:
        return f"Error searching knowledge base: {e}. Have you run `dvc repro` to build the embeddings?"


# ── Tool 2: List all articles in the knowledge base ───────────────────────────
@mcp.tool()
def list_articles() -> str:
    """List all articles currently stored in the knowledge base with their tags."""
    articles_path = Path("data/processed/articles_clean.json")
    if not articles_path.exists():
        return "No articles found. Run `dvc repro` first to build the pipeline."
    articles = json.loads(articles_path.read_text())
    lines = [f"- [{a['id']}] {a['title']}  (tags: {', '.join(a.get('tags', []))})" for a in articles]
    return f"Knowledge base contains {len(articles)} articles:\n" + "\n".join(lines)


# ── Tool 3: Get Q&A pairs for a topic ─────────────────────────────────────────
@mcp.tool()
def get_qa_pairs(topic: str = "") -> str:
    """
    Return sample Q&A pairs from the dataset.
    Optionally filter by topic keyword in the question or answer.
    """
    qa_path = Path("data/processed/qa_pairs_clean.json")
    if not qa_path.exists():
        return "No Q&A pairs found. Run `dvc repro` first."
    pairs = json.loads(qa_path.read_text())
    if topic:
        pairs = [
            p for p in pairs
            if topic.lower() in p["question"].lower() or topic.lower() in p["answer"].lower()
        ]
    if not pairs:
        return f"No Q&A pairs found for topic '{topic}'."
    lines = [f"Q: {p['question']}\nA: {p['answer']}" for p in pairs]
    return "\n\n".join(lines)


# ── Resource: Expose the raw articles file as an MCP resource ─────────────────
@mcp.resource("knowledge://articles")
def get_articles_resource() -> str:
    """The full processed articles dataset as JSON."""
    p = Path("data/processed/articles_clean.json")
    return p.read_text() if p.exists() else "[]"


if __name__ == "__main__":
    print("🚀 Starting MCP server...")
    print("   Tools: search_knowledge, list_articles, get_qa_pairs")
    print("   Resources: knowledge://articles")
    mcp.run()
