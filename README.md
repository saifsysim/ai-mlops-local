# 🧠 AI MLOps Local — End-to-End AI Learning Project

A complete, locally-hosted AI/MLOps pipeline built for learning. No cloud required.

## What's Inside

| Stage | Tech |
|---|---|
| Data versioning | DVC |
| Vector store / RAG | ChromaDB + Ollama embeddings |
| MCP server | FastMCP |
| Agent framework | LangChain ReAct |
| Observability | LangFuse |
| Agent evals | DeepEval + Promptfoo |
| Fine-tuning | Unsloth + LoRA |
| Serving | FastAPI + Docker |
| Orchestration | Prefect |
| UI | React + Vite |

## Quick Start

### 1. Prerequisites
```bash
# Install Ollama (runs LLMs locally)
brew install ollama
ollama pull llama3.2
ollama pull nomic-embed-text

# Install Docker Desktop: https://www.docker.com/products/docker-desktop
```

### 2. Python Environment
```bash
cd ai-mlops-local
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure Environment
```bash
cp .env.example .env
# Edit .env if needed (defaults work out of the box)
```

### 4. Initialize Git + DVC
```bash
git init && dvc init
git add . && git commit -m "Initial scaffold"
```

### 5. Run the Data Pipeline
```bash
# Ingest raw data
python src/data/ingest.py

# Run full DVC pipeline (preprocess → embed)
dvc repro
```

### 6. Start All Services
```bash
docker-compose up -d
```

| Service | URL |
|---|---|
| Agent API | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |
| LangFuse | http://localhost:3000 |
| Prefect UI | http://localhost:4200 |

### 7. Run the MCP Server
```bash
python src/mcp/server.py
```

### 8. Chat via the UI
```bash
cd ui && npm install && npm run dev
# Open http://localhost:5173
```

### 9. Run Agent Evals
```bash
deepeval test run tests/test_agent_evals.py
```

### 10. Run the Full Orchestrated Pipeline
```bash
python pipelines/full_pipeline.py
```

## Project Structure

```
ai-mlops-local/
├── data/               # DVC-tracked datasets & embeddings
├── src/
│   ├── data/           # Ingest, preprocess, embed
│   ├── mcp/            # MCP tool server (FastMCP)
│   ├── agents/         # LangChain agent + tools
│   ├── training/       # Fine-tuning scripts
│   └── serving/        # FastAPI gateway
├── tests/              # DeepEval + Promptfoo agent tests
├── pipelines/          # Prefect orchestration flow
├── ui/                 # React + Vite chat interface
├── docker-compose.yml
├── dvc.yaml
└── requirements.txt
```

## Learning Path

Follow the phases in order:
1. **Week 1** — Data pipeline (ingest → preprocess → embed)
2. **Week 2** — MCP + Agents + LangFuse
3. **Week 3** — Agent testing + fine-tuning
4. **Week 4** — Deployment + React UI
