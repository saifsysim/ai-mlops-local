"""
Phase 2 — RAG Agent (LangChain 1.x / LangGraph)
A ReAct agent that searches your local ChromaDB knowledge base and answers
questions grounded in the retrieved context. All LLM calls run via Ollama
(completely local — no API keys needed).

Usage:
    python src/agents/rag_agent.py

Or import into the FastAPI serving layer:
    from src.agents.rag_agent import ask_agent
"""

import os
from dotenv import load_dotenv

load_dotenv()

from langchain_ollama import ChatOllama
from langchain_core.tools import tool as lc_tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from src.data.embed import query_vector_store


# ── Tools ──────────────────────────────────────────────────────────────────────
@lc_tool
def search_knowledge(query: str) -> str:
    """Search the local knowledge base for information relevant to the query.
    Use this whenever you need factual information to answer a question."""
    results = query_vector_store(query, n_results=3)
    if not results:
        return "No relevant documents found in the knowledge base."
    parts = [f"[Source: {r['metadata']['title']}]\n{r['text']}" for r in results]
    return "\n\n---\n\n".join(parts)


@lc_tool
def list_available_topics() -> str:
    """List all topics and articles currently available in the knowledge base.
    Use this when the user asks what topics are covered."""
    import json
    from pathlib import Path
    path = Path("data/processed/articles_clean.json")
    if not path.exists():
        return "Knowledge base not built yet. Run the data pipeline first."
    articles = json.loads(path.read_text())
    lines = [f"• {a['title']}  (tags: {', '.join(a.get('tags', []))})" for a in articles]
    return "Available topics:\n" + "\n".join(lines)


TOOLS = [search_knowledge, list_available_topics]

SYSTEM_PROMPT = """You are an intelligent AI assistant with access to a local knowledge base.

ALWAYS use the search_knowledge tool before answering factual questions.
Base your answers ONLY on retrieved information — do not hallucinate facts.
If the knowledge base does not contain relevant information, say so clearly.
Cite the source article title when referencing retrieved content."""


# ── LangFuse: singleton client (gracefully disabled if not configured) ────────
_langfuse = None

def _get_langfuse():
    global _langfuse
    if _langfuse is not None:
        return _langfuse
    try:
        from langfuse import Langfuse
        client = Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY", ""),
            host=os.getenv("LANGFUSE_HOST", "http://localhost:3000"),
        )
        _langfuse = client
        print("✅ LangFuse tracing enabled")
        return _langfuse
    except Exception as e:
        print(f"⚠️  LangFuse not configured ({e}). Proceeding without tracing.")
        return None


# ── Build agent (singleton) ────────────────────────────────────────────────────
_agent = None

def get_agent():
    global _agent
    if _agent is None:
        llm = ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "llama3.2"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=0.1,
        )
        _agent = create_react_agent(llm, TOOLS, prompt=SYSTEM_PROMPT)
    return _agent


def ask_agent(question: str, session_id: str = "default") -> dict:
    """Run the agent on a question, trace it in LangFuse, and return a structured response."""
    agent = get_agent()
    result = agent.invoke({"messages": [HumanMessage(content=question)]})
    final = result["messages"][-1].content

    # ── Trace to LangFuse (direct SDK — works regardless of langchain version) ──
    lf = _get_langfuse()
    if lf:
        try:
            trace = lf.trace(
                name="rag-agent",
                input={"question": question},
                session_id=session_id,
                tags=["llama3.2", "chromadb", "rag"],
            )
            trace.update(output={"answer": final})
            lf.flush()
        except Exception as e:
            print(f"⚠️  LangFuse trace failed: {e}")

    return {"answer": final, "session_id": session_id}


# ── CLI quick test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    questions = [
        "What is RAG and how does it differ from fine-tuning?",
        "What topics are available in the knowledge base?",
        "How does LangFuse help with LLM debugging?",
    ]
    for q in questions:
        print(f"\n{'='*60}\n❓ {q}\n{'='*60}")
        response = ask_agent(q)
        print(f"\n💬 {response['answer']}")
