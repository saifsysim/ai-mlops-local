# 🧠 AI MLOps Local — Learn AI Engineering From Scratch

> **Who is this for?** Complete beginners. If you've never built an AI pipeline before, you're in exactly the right place. This project teaches you by doing — you'll run real AI systems on your own laptop, step by step.

---

## 🤔 What Even Is This?

This project shows you how **real AI applications actually work** behind the scenes.

When you use ChatGPT or another AI assistant, a lot is happening under the hood:
- The AI reads documents and turns them into searchable data
- A "smart agent" decides what to do based on your question
- The system logs every decision so engineers can debug it
- It's all monitored, tested, and deployed like professional software

**This project does all of that — on your laptop, for free, with no cloud needed.**

You'll learn by building a real AI system piece by piece over 4 weeks.

---

## 🧩 The Tools You'll Use (and why)

Don't worry if these names mean nothing yet — you'll understand each one as you use it.

| Tool | Plain English Explanation |
|---|---|
| **Ollama** | Runs AI language models (like ChatGPT) on your own computer |
| **ChromaDB** | A database that stores text as math, so AI can search it by meaning |
| **LangChain** | A framework that lets you build an AI "agent" that can reason and take actions |
| **DVC** | Version control for data files (like Git, but for large datasets) |
| **LangFuse** | Records every AI interaction so you can see exactly what the AI thought |
| **FastAPI** | Turns your Python code into a web service (an API) |
| **Docker** | Packages apps into containers so they run the same everywhere |
| **Prefect** | Schedules and monitors data pipelines (runs code in the right order) |
| **DeepEval** | Automatically tests whether your AI gives good answers |
| **React + Vite** | Builds the chat interface you'll use to talk to your AI |

---

## ✅ Before You Start — Install These

You only need to do this once.

### Step 1: Install Homebrew (Mac package manager)

If you've never used Homebrew, open your **Terminal** app and run:
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

> 💡 **What is Terminal?** Press `Command + Space`, type "Terminal", hit Enter. It's where you type commands to control your computer.

---

### Step 2: Install Ollama

Ollama lets you run AI models like `llama3.2` directly on your Mac — no internet needed after the download.

```bash
brew install ollama
```

Then download the AI models you'll use:
```bash
# Start Ollama (run this in one Terminal window, keep it open)
ollama serve

# In a NEW Terminal window, download the models:
ollama pull llama3.2           # The main AI model you'll chat with
ollama pull nomic-embed-text   # A model that converts text into searchable math
```

> 💡 These downloads can be large (a few GB). Let them finish before continuing.

---

### Step 3: Install Docker Desktop

Docker runs supporting services like your observability dashboard.

1. Go to https://www.docker.com/products/docker-desktop
2. Download and install it
3. Open Docker Desktop and wait until it says "Docker is running"

---

### Step 4: Install Python

Check if you have Python 3.10+ installed:
```bash
python3 --version
```

If you see something like `Python 3.11.x`, you're good. If not:
```bash
brew install python@3.11
```

---

### Step 5: Install Node.js (for the chat UI)

```bash
brew install node
```

Check it worked:
```bash
node --version   # Should show v18 or higher
npm --version
```

---

## 📦 Get the Project Running

### 1. Clone this repository

"Cloning" means downloading the code to your computer:

```bash
git clone https://github.com/saifsysim/ai-mlops-local.git
cd ai-mlops-local
```

> 💡 **What is `cd`?** It means "change directory" — it moves you into the project folder.

---

### 2. Create a Python virtual environment

A virtual environment is like a sandbox — it keeps this project's dependencies separate from everything else on your computer.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

You'll know it worked when your terminal prompt shows `(.venv)` at the start.

> ⚠️ **Every time you open a new terminal for this project**, run `source .venv/bin/activate` first!

---

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

This installs all the Python libraries this project needs. It may take a few minutes.

---

### 4. Set up your configuration file

```bash
cp .env.example .env
```

This creates a `.env` file with settings. The defaults work out of the box — you don't need to change anything to get started.

---

### 5. Start all services

```bash
docker-compose up -d
```

This starts LangFuse (AI observability) and Prefect (pipeline scheduler) in the background.

Wait about 30 seconds, then visit these in your browser:

| What | URL |
|---|---|
| Your AI agent's API | http://localhost:8000/docs |
| LangFuse (AI traces) | http://localhost:3000 |
| Prefect (pipeline UI) | http://localhost:4200 |

---

## 📅 4-Week Learning Path

Work through these in order. Each week builds on the last.

---

### 🟢 Week 1 — How AI Reads and Searches Data (RAG Pipeline)

**Concept: What is RAG?**
RAG stands for Retrieval Augmented Generation. Instead of the AI guessing an answer from memory, it first *searches* a collection of documents you give it, then *generates* an answer based on what it found — like an open-book exam.

**What you'll do this week:**

```bash
# Step 1: Ingest raw text documents into the system
python src/data/ingest.py
```
→ Look inside `src/data/ingest.py` and read the comments. What kind of data is it loading?

```bash
# Step 2: Clean and chunk the data into smaller pieces
# (This runs automatically as part of the DVC pipeline)

# Step 3: Convert text into vectors (numbers) so AI can search it
dvc repro
```

**What is DVC?** DVC tracks your data files the same way Git tracks your code. When you run `dvc repro`, it runs the whole pipeline from raw data → cleaned data → embeddings.

**Files to read this week:**
- `src/data/ingest.py` — loads raw data
- `src/data/preprocess.py` — cleans and splits text
- `src/data/embed.py` — converts text into vectors using Ollama
- `dvc.yaml` — defines the pipeline steps (read this like a recipe)

**You'll know you succeeded when:** `data/embeddings/` has files in it.

---

### 🟡 Week 2 — Building an AI Agent

**Concept: What is an AI Agent?**
A regular AI just answers questions. An AI *agent* can also take *actions* — like searching your documents, calling an API, or running calculations — and decide *which* action to take based on your question. It's like giving the AI hands.

MCP (Model Context Protocol) is the standard way to give an AI agent access to tools.

**What you'll do this week:**

```bash
# Terminal 1: Start the MCP tool server (gives the agent its tools)
python src/mcp/server.py

# Terminal 2: Run the agent (it will use the tools from Terminal 1)
python src/agents/rag_agent.py
```

Now open **LangFuse at http://localhost:3000** and watch the traces appear. You'll see exactly what the agent searched for and how it decided on its answer.

**Files to read this week:**
- `src/mcp/server.py` — defines the tools the agent can use
- `src/agents/tools.py` — how tools are described to the agent
- `src/agents/prompts.py` — the instructions given to the AI
- `src/agents/rag_agent.py` — the full agent logic

**You'll know you succeeded when:** You ask a question in the terminal, the agent searches your documents, and you see the trace in LangFuse.

---

### 🔴 Week 3 — Testing and Improving Your Agent

**Concept: Why test AI?**
AI can give confident-sounding wrong answers. Agent evaluation tools let you automatically check whether your agent's answers are accurate, relevant, and not hallucinated (made up).

**What you'll do this week:**

```bash
# Run automated AI quality tests
deepeval test run tests/test_agent_evals.py
```

Read the test results — did your agent pass? Look at `tests/test_agent_evals.py` to understand what each test checks.

**Bonus — Fine-tuning (optional, requires a powerful Mac):**
Fine-tuning means training a model on your specific data so it gets better at your task.

```bash
# Prepare a training dataset from your documents
python src/training/prepare_dataset.py

# Run the training (this can take a while)
python src/training/finetune.py
```

**Files to read this week:**
- `tests/test_agent_evals.py` — what makes a "good" agent response?
- `tests/promptfoo.yml` — alternative eval configuration
- `src/training/prepare_dataset.py` — how data is formatted for training
- `src/training/finetune.py` — the actual training code

---

### 🔵 Week 4 — Serving and the Chat UI

**Concept: What is an API?**
An API (Application Programming Interface) is a way for different programs to talk to each other. Your FastAPI server lets your React chat UI talk to the AI agent.

**What you'll do this week:**

```bash
# Start the chat UI
cd ui
npm install    # Download UI dependencies (first time only)
npm run dev    # Start the UI
```

Open http://localhost:5173 and chat with your AI!

```bash
# In a new terminal — run the full pipeline from start to finish
python pipelines/full_pipeline.py
```

**Files to read this week:**
- `src/serving/api.py` — the FastAPI web server that connects UI to agent
- `ui/src/App.jsx` — the React chat interface
- `pipelines/full_pipeline.py` — orchestrates all stages using Prefect

**You'll know you succeeded when:** You can type a question at http://localhost:5173 and get an answer from your local AI.

---

## 📁 Project Map

Here's a bird's-eye view of the project so you don't get lost:

```
ai-mlops-local/
│
├── data/                 ← Your data lives here (tracked by DVC)
│   ├── raw/              ← Original input documents
│   ├── processed/        ← Cleaned and chunked text
│   └── embeddings/       ← ChromaDB vector database
│
├── src/                  ← All the Python source code
│   ├── data/             ← Week 1: ingest → preprocess → embed
│   ├── mcp/              ← Week 2: MCP tool server
│   ├── agents/           ← Week 2: AI agent logic
│   ├── training/         ← Week 3: fine-tuning scripts
│   └── serving/          ← Week 4: FastAPI web server
│
├── tests/                ← Week 3: agent quality tests
├── pipelines/            ← Week 4: Prefect orchestration
├── ui/                   ← Week 4: React chat interface
│
├── docker-compose.yml    ← Starts LangFuse + Prefect
├── dvc.yaml              ← Defines the data pipeline
├── .env.example          ← Copy to .env for configuration
└── requirements.txt      ← Python libraries to install
```

---

## 🔧 When Something Goes Wrong

| Error or Problem | What to Do |
|---|---|
| `command not found: ollama` | You skipped Step 2. Run `brew install ollama` |
| `Connection refused` on port 11434 | Ollama isn't running. Open a terminal and run `ollama serve` |
| `Docker not running` | Open the Docker Desktop app and wait for it to start |
| `ModuleNotFoundError` | Your virtual environment isn't active. Run `source .venv/bin/activate` |
| LangFuse at localhost:3000 is blank | Docker-compose is still starting. Wait 30s and refresh |
| LangFuse shows no traces | Check `.env` — `LANGFUSE_HOST` should be `http://localhost:3000` |
| `dvc repro` says "nothing to reproduce" | Your data hasn't changed. Force it: `dvc repro -f` |
| Fine-tuning is very slow | Skip `finetune.py` for now — it needs Apple Silicon (M1/M2/M3/M4) |

---

## 💡 General Tips for Beginners

- **Read the code, don't just run it.** Every file has comments explaining what it does.
- **One week at a time.** Don't jump ahead — each week assumes you've done the previous one.
- **LangFuse is your best friend.** When something behaves unexpectedly, check the trace.
- **It's okay if things break.** Errors are how you learn. Read the error message carefully — it almost always tells you the fix.
- **Google is allowed.** If you don't know what a term means, look it up. That's how every engineer learns.
