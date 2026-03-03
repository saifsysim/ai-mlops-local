"""
Phase 5 — FastAPI Agent Gateway
Production-ready REST API that wraps the LangChain agent.
Includes Prometheus metrics, health check, and structured responses.

Run locally:
    uvicorn src.serving.api:app --reload --port 8000

Then open:
    http://localhost:8000/docs  — Interactive Swagger UI
"""

import time
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest
from starlette.responses import PlainTextResponse

load_dotenv()

# ── Prometheus metrics ─────────────────────────────────────────────────────────
REQUEST_COUNT = Counter(
    "agent_requests_total",
    "Total number of agent API requests",
    ["status"],
)
REQUEST_LATENCY = Histogram(
    "agent_request_latency_seconds",
    "Agent request latency in seconds",
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
)


# ── Pydantic schemas ───────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The question to ask the agent")
    session_id: str = Field(default="default", description="Session ID for LangFuse grouping")


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    latency_ms: float


# ── App lifespan: warm up the agent on startup ────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 AI Agent API starting up — warming agent...")
    try:
        from src.agents.rag_agent import get_agent
        get_agent()
        print("✅ Agent warmed up and ready")
    except Exception as e:
        print(f"⚠️  Agent warm-up failed: {e}. Check Ollama is running.")
    yield
    print("👋 AI Agent API shutting down")


# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AI MLOps Local — Agent API",
    description="LangChain ReAct agent backed by a local Ollama LLM and ChromaDB vector store.",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow the React UI (running on localhost:5173) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
def health_check():
    """Returns 200 OK when the service is ready."""
    return {"status": "ok", "model": os.getenv("OLLAMA_MODEL", "llama3.2")}


@app.get("/metrics", response_class=PlainTextResponse, tags=["System"])
def metrics():
    """Prometheus metrics endpoint — scraped by Prometheus."""
    return generate_latest()


@app.post("/agent/chat", response_model=ChatResponse, tags=["Agent"])
def chat(request: ChatRequest):
    """
    Send a question to the RAG agent.
    The agent will search the knowledge base and return a grounded answer.
    Every call is traced in LangFuse (if configured).
    """
    try:
        from src.agents.rag_agent import ask_agent

        start = time.perf_counter()
        result = ask_agent(request.question, session_id=request.session_id)
        latency = (time.perf_counter() - start) * 1000

        REQUEST_COUNT.labels(status="success").inc()
        REQUEST_LATENCY.observe(latency / 1000)

        return ChatResponse(
            answer=result["answer"],
            session_id=result["session_id"],
            latency_ms=round(latency, 2),
        )
    except Exception as e:
        REQUEST_COUNT.labels(status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/knowledge/topics", tags=["Knowledge"])
def list_topics():
    """List all topics in the knowledge base."""
    try:
        import json
        from pathlib import Path
        path = Path("data/processed/articles_clean.json")
        if not path.exists():
            return {"topics": [], "message": "Run `dvc repro` to build the knowledge base."}
        articles = json.loads(path.read_text())
        return {
            "count": len(articles),
            "topics": [{"id": a["id"], "title": a["title"], "tags": a.get("tags", [])} for a in articles],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
