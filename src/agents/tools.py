"""
Phase 2 — Agent Tools
LangChain-compatible tool definitions that wrap the MCP server's capabilities.
The agent will discover and use these tools autonomously during its reasoning loop.
"""

from langchain.tools import tool
from src.data.embed import query_vector_store


@tool
def search_knowledge(query: str) -> str:
    """
    Search the local knowledge base for information relevant to the query.
    Use this tool whenever you need factual information to answer a question.
    """
    results = query_vector_store(query, n_results=3)
    if not results:
        return "No relevant documents found in the knowledge base."
    parts = [f"[Source: {r['metadata']['title']}]\n{r['text']}" for r in results]
    return "\n\n---\n\n".join(parts)


@tool
def list_available_topics() -> str:
    """
    List all topics and articles currently available in the knowledge base.
    Use this tool when the user asks what you know about or what topics are covered.
    """
    import json
    from pathlib import Path

    path = Path("data/processed/articles_clean.json")
    if not path.exists():
        return "Knowledge base not built yet. Run `dvc repro` first."
    articles = json.loads(path.read_text())
    lines = [f"• {a['title']}  (tags: {', '.join(a.get('tags', []))})" for a in articles]
    return "Available topics in knowledge base:\n" + "\n".join(lines)


@tool
def get_sample_qa(topic: str = "") -> str:
    """
    Retrieve sample question-answer pairs from the dataset.
    Useful when the user wants examples or when you need to validate your understanding.
    Optionally filter by topic keyword.
    """
    import json
    from pathlib import Path

    path = Path("data/processed/qa_pairs_clean.json")
    if not path.exists():
        return "Q&A dataset not built yet. Run `dvc repro` first."
    pairs = json.loads(path.read_text())
    if topic:
        pairs = [
            p for p in pairs
            if topic.lower() in p["question"].lower() or topic.lower() in p["answer"].lower()
        ]
    if not pairs:
        return f"No Q&A pairs found for topic '{topic}'."
    lines = [f"Q: {p['question']}\nA: {p['answer']}" for p in pairs[:3]]
    return "\n\n".join(lines)


# Export a flat list for agent registration
AGENT_TOOLS = [search_knowledge, list_available_topics, get_sample_qa]
