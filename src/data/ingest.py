"""
Phase 1 — Data Ingestion
Downloads sample AI/ML articles and Q&A pairs to data/raw/.
In a real project, swap these sources for your own data (PDFs, CSVs, databases).
"""

import json
import urllib.request
from pathlib import Path

RAW_DIR = Path("data/raw")

# ── Sample Articles ────────────────────────────────────────────────────────────
# These are representative Q&A topics about AI/ML that the agent will reason over.
# In production: replace with your own PDF loader, database export, or API calls.

ARTICLES = [
    {
        "id": "art_001",
        "title": "What is Machine Learning?",
        "content": (
            "Machine Learning (ML) is a branch of artificial intelligence that enables systems "
            "to learn and improve from experience without being explicitly programmed. ML focuses "
            "on developing computer programs that can access data and use it to learn for themselves. "
            "The process begins with observations or data, such as examples, direct experience, or "
            "instruction, to look for patterns in data and make better decisions in the future."
        ),
        "source": "synthetic",
        "tags": ["ml", "basics", "ai"],
    },
    {
        "id": "art_002",
        "title": "What is RAG (Retrieval-Augmented Generation)?",
        "content": (
            "Retrieval-Augmented Generation (RAG) is a technique that combines information retrieval "
            "with text generation models. Instead of relying solely on the knowledge stored in an LLM's "
            "weights, RAG retrieves relevant documents from an external knowledge base at query time "
            "and feeds them into the LLM as additional context. This reduces hallucinations, allows "
            "the LLM to work with up-to-date information, and makes it possible to ground answers "
            "in specific, verifiable sources."
        ),
        "source": "synthetic",
        "tags": ["rag", "llm", "retrieval"],
    },
    {
        "id": "art_003",
        "title": "What is the Model Context Protocol (MCP)?",
        "content": (
            "The Model Context Protocol (MCP) is an open standard introduced by Anthropic that "
            "defines a uniform way for LLMs to discover and call external tools, data sources, "
            "and APIs. Think of it as a standardized 'plugin system' for AI models. Instead of each "
            "application implementing custom integrations, MCP provides a common interface so any "
            "MCP-compatible model can consume any MCP-compatible tool — similar to how USB-C "
            "standardized device charging. MCP servers expose resources (data) and tools (callable "
            "functions) that agents can list and invoke."
        ),
        "source": "synthetic",
        "tags": ["mcp", "protocol", "tools", "agents"],
    },
    {
        "id": "art_004",
        "title": "What is LangChain and how do agents work?",
        "content": (
            "LangChain is a framework for building applications powered by large language models. "
            "At its core, LangChain provides abstractions for chains (sequences of LLM calls), "
            "tools (external capabilities an LLM can invoke), and agents (autonomous reasoning loops). "
            "A LangChain ReAct agent follows the Thought → Action → Observation cycle: it reasons "
            "about what action to take, calls a tool, observes the result, and repeats until it "
            "has enough information to answer. This makes agents much more capable than single-shot "
            "LLM calls because they can break down complex tasks and use external data."
        ),
        "source": "synthetic",
        "tags": ["langchain", "agents", "react", "tools"],
    },
    {
        "id": "art_005",
        "title": "What is LangFuse?",
        "content": (
            "LangFuse is an open-source LLM observability and analytics platform. It allows "
            "developers to trace every LLM call and agent step, score outputs for quality, "
            "and monitor metrics like latency, token usage, and cost over time. LangFuse integrates "
            "with LangChain via a callback handler, so every chain run and agent step is automatically "
            "captured as a trace. The LangFuse UI shows a full tree of LLM interactions, inputs, "
            "outputs, and metadata — making it easy to debug why an agent gave a wrong answer."
        ),
        "source": "synthetic",
        "tags": ["langfuse", "observability", "tracing", "llm"],
    },
    {
        "id": "art_006",
        "title": "What is fine-tuning and when should you use it?",
        "content": (
            "Fine-tuning is the process of continuing to train a pre-trained language model on a "
            "smaller, domain-specific dataset. The goal is to adapt the model's behavior — its "
            "tone, formatting, or specialized knowledge — without training from scratch. "
            "LoRA (Low-Rank Adaptation) is a popular parameter-efficient fine-tuning technique "
            "that trains a small set of adapter weights instead of modifying all model parameters, "
            "making fine-tuning feasible on consumer hardware. Fine-tune when: (1) prompt engineering "
            "no longer improves quality, (2) you need consistent output structure, or (3) you want "
            "to internalize proprietary domain knowledge."
        ),
        "source": "synthetic",
        "tags": ["fine-tuning", "lora", "training", "llm"],
    },
]

# ── Sample Q&A pairs (used for agent eval + fine-tune dataset) ─────────────────
QA_PAIRS = [
    {
        "id": "qa_001",
        "question": "What is the difference between RAG and fine-tuning?",
        "answer": (
            "RAG retrieves external documents at query time to ground the LLM's answer in "
            "up-to-date information, while fine-tuning bakes domain knowledge directly into the "
            "model's weights during a training phase. Use RAG when your data changes frequently "
            "or when you need source citations. Use fine-tuning when you need consistent style, "
            "formatting, or deeply internalized knowledge that rarely changes."
        ),
        "context_ids": ["art_002", "art_006"],
    },
    {
        "id": "qa_002",
        "question": "How does MCP relate to LangChain tool calling?",
        "answer": (
            "LangChain tool calling is framework-specific — tools are Python classes registered "
            "with an agent. MCP is a protocol-level standard — tools are exposed over a network "
            "interface any MCP-compatible client can consume. MCP tools can be wrapped as LangChain "
            "tools, giving you both the standardisation of MCP and the reasoning power of LangChain agents."
        ),
        "context_ids": ["art_003", "art_004"],
    },
    {
        "id": "qa_003",
        "question": "Why would I use LangFuse instead of just printing logs?",
        "answer": (
            "Print logs are flat and hard to correlate across a multi-step agent run. LangFuse "
            "captures a nested trace tree that shows exactly which tool was called at each reasoning "
            "step, what the inputs/outputs were, how long each step took, and how many tokens were "
            "used. You can also attach human scores to traces and run analytics across hundreds of "
            "agent runs to spot failure patterns."
        ),
        "context_ids": ["art_005"],
    },
]


def ingest():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    articles_path = RAW_DIR / "articles.json"
    with open(articles_path, "w") as f:
        json.dump(ARTICLES, f, indent=2)
    print(f"✅ Saved {len(ARTICLES)} articles → {articles_path}")

    qa_path = RAW_DIR / "qa_pairs.json"
    with open(qa_path, "w") as f:
        json.dump(QA_PAIRS, f, indent=2)
    print(f"✅ Saved {len(QA_PAIRS)} Q&A pairs → {qa_path}")


if __name__ == "__main__":
    ingest()
