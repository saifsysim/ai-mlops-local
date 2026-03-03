# 🧠 AI MLOps Local — End-to-End AI Learning Project

> A complete, locally-hosted AI/MLOps pipeline built for learning. **No cloud required.**

---

## 🎯 What You'll Learn

By following this project end-to-end, you'll learn how to:

- **Version your data** with DVC (like git, but for datasets)
- **Build a RAG pipeline** — embed documents and query them with a local LLM
- **Create AI tools** using the Model Context Protocol (MCP)
- **Build an AI Agent** with LangChain that reasons and uses tools
- **Observe & trace** every LLM call with LangFuse (local, no cloud)
- **Evaluate your agent** with DeepEval and Promptfoo
- **Fine-tune a model** locally using Unsloth + LoRA
- **Serve your agent** via FastAPI + Docker
- **Orchestrate pipelines** with Prefect
- **Chat with your agent** through a React UI you'll run locally

---

## 🗺️ Tech Stack

| Stage | Technology |
|---|---|
| Data versioning | DVC |
| Vector store / RAG | ChromaDB + Ollama embeddings |
| MCP server | FastMCP |
| Agent framework | LangChain ReAct |
| Observability | LangFuse (self-hosted) |
| Agent evals | DeepEval + Promptfoo |
| Fine-tuning | Unsloth + LoRA |
| Serving | FastAPI + Docker |
| Orchestration | Prefect |
| UI | React + Vite |

---

## 🚀 Setup (Do This First)

### 1. Prerequisites

Install these before anything else:

```bash
# 1a. Install Ollama (runs LLMs locally on your Mac)
brew install ollama

# 1b. Pull the models you'll use
ollama pull llama3.2           # main chat model
ollama pull nomic-embed-text   # embedding model for RAG
ollama pull mistral            # used during fine-tuning experiments
```

> 💡 **First time with Ollama?** After `brew install ollama`, run `ollama serve` in a separate terminal to start the local server.

Install [Docker Desktop](https://www.docker.com/products/docker-desktop) — needed to run LangFuse and other services.

---

### 2. Clone & Set Up Python

```bash
git clone https://github.com/saifsysim/ai-mlops-local.git
cd ai-mlops-local

# Create and activate a virtual environment
python -m venv .venv && source .venv/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

---

### 3. Configure Environment

```bash
cp .env.example .env
# Open .env in your editor — the defaults work out of the box
# Only change values if you know what you're doing
```

---

### 4. Initialize DVC (Data Version Control)

```bash
dvc init
git add .dvc .gitignore
git commit -m "Initialize DVC"
```

> 💡 DVC is already set up in `dvc.yaml`. This step just links it to your local git repo.

---

### 5. Start All Services

```bash
docker-compose up -d
```

| Service | URL | What it does |
|---|---|---|
| Agent API | http://localhost:8000 | Your FastAPI backend |
| API Docs | http://localhost:8000/docs | Interactive API explorer |
| LangFuse | http://localhost:3000 | Traces every LLM call |
| Prefect UI | http://localhost:4200 | Pipeline orchestration dashboard |

---

## 📚 Learning Path — Follow in Order

### 🟢 Week 1 — Data Pipeline (Ingest → Embed)

**Goal:** Understand how raw data becomes searchable vectors.

```bash
# Step 1: Ingest raw data (creates files in data/raw/)
python src/data/ingest.py

# Step 2: Run the full DVC pipeline (preprocess → embed → store in ChromaDB)
dvc repro
```

**Files to study:**
- `src/data/ingest.py` — how data is loaded
- `src/data/preprocess.py` — cleaning and chunking
- `src/data/embed.py` — turning text into vectors
- `dvc.yaml` — how the pipeline stages connect

---

### 🟡 Week 2 — MCP Server + Agent + Observability

**Goal:** Build tools, wire up an agent, and watch traces in LangFuse.

```bash
# Step 1: Start the MCP tool server
python src/mcp/server.py

# Step 2: In a separate terminal, test the agent
python src/agents/rag_agent.py
```

Then open **LangFuse at http://localhost:3000** to see your agent's trace.

**Files to study:**
- `src/mcp/server.py` — MCP tools exposed to the agent
- `src/agents/rag_agent.py` — the ReAct agent logic
- `src/agents/tools.py` — tool definitions
- `src/agents/prompts.py` — system prompt

---

### 🔴 Week 3 — Evaluation + Fine-Tuning

**Goal:** Test your agent's quality, then improve a model with LoRA fine-tuning.

```bash
# Run agent evals
deepeval test run tests/test_agent_evals.py

# (Optional) Run Promptfoo evals
npx promptfoo eval --config tests/promptfoo.yml

# Prepare fine-tuning dataset
python src/training/prepare_dataset.py

# Run LoRA fine-tune (requires GPU or Apple Silicon)
python src/training/finetune.py
```

**Files to study:**
- `tests/test_agent_evals.py` — DeepEval test cases
- `tests/promptfoo.yml` — Promptfoo eval config
- `src/training/prepare_dataset.py` — dataset formatting
- `src/training/finetune.py` — Unsloth LoRA training
- `src/training/Modelfile` — how to pull the fine-tuned model into Ollama

---

### 🔵 Week 4 — Serving + UI + Full Pipeline

**Goal:** Package everything up and run the full stack.

```bash
# Start the React chat UI
cd ui && npm install && npm run dev
# Open http://localhost:5173

# Run the full orchestrated pipeline (all stages via Prefect)
python pipelines/full_pipeline.py
```

**Files to study:**
- `src/serving/api.py` — FastAPI app
- `ui/src/App.jsx` — React chat interface
- `pipelines/full_pipeline.py` — Prefect flow connecting all stages

---

## 📁 Project Structure

```
ai-mlops-local/
├── data/
│   ├── raw/            # Original input data (DVC tracked)
│   ├── processed/      # Cleaned/chunked data
│   └── embeddings/     # ChromaDB vector store
├── src/
│   ├── data/           # ingest.py, preprocess.py, embed.py
│   ├── mcp/            # MCP tool server
│   ├── agents/         # LangChain ReAct agent + tools + prompts
│   ├── training/       # Fine-tuning scripts + Modelfile
│   └── serving/        # FastAPI gateway
├── tests/              # DeepEval + Promptfoo agent evals
├── pipelines/          # Prefect orchestration flow
├── ui/                 # React + Vite chat interface
├── docker-compose.yml  # Spins up LangFuse + Prefect
├── dvc.yaml            # Pipeline stage definitions
├── .env.example        # Copy to .env to configure
└── requirements.txt
```

---

## 🛠️ Troubleshooting

| Problem | Fix |
|---|---|
| `ollama: command not found` | Run `brew install ollama` first |
| `Connection refused` on port 11434 | Run `ollama serve` in a separate terminal |
| `docker-compose up` fails | Make sure Docker Desktop is running |
| `ModuleNotFoundError` | Make sure your venv is active: `source .venv/bin/activate` |
| LangFuse shows no traces | Check `LANGFUSE_HOST` in `.env` points to `http://localhost:3000` |
| `dvc repro` skips steps | Your data hasn't changed — force with `dvc repro -f` |

---

## 📌 Tips

- Run each week's code **before** moving to the next — each builds on the previous
- LangFuse traces are your best friend for debugging agent behavior
- If fine-tuning is slow, skip Week 3's `finetune.py` and use the pre-built `Modelfile` instead
