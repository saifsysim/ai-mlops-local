import { useState, useRef, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import axios from 'axios'

const API = 'http://localhost:8000'

const SUGGESTIONS = [
    'What is RAG and how does it work?',
    'Explain LangChain agents with an example',
    'What is MCP and why does it matter?',
    'How is LangFuse used for debugging?',
    'Difference between RAG and fine-tuning?',
]

const PIPELINE_STEPS = [
    { label: 'Ingest raw data', cmd: 'python src/data/ingest.py', done: true },
    { label: 'Preprocess & clean', cmd: 'python src/data/preprocess.py', done: true },
    { label: 'Build vector embeddings', cmd: 'dvc repro embed', done: true },
    { label: 'Run the agent', cmd: 'python src/agents/rag_agent.py', done: true },
    { label: 'Run evals', cmd: 'deepeval test run tests/', done: false },
    { label: 'Fine-tune model', cmd: 'python src/training/finetune.py', done: false },
    { label: 'Deploy API', cmd: 'docker-compose up -d', done: false },
]

// ── Chat View ──────────────────────────────────────────────────────────────────
function ChatView() {
    const [messages, setMessages] = useState([])
    const [input, setInput] = useState('')
    const [loading, setLoading] = useState(false)
    const [sessionId] = useState(() => `session-${Date.now()}`)
    const bottomRef = useRef(null)

    useEffect(() => {
        bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
    }, [messages, loading])

    const send = async (question) => {
        const q = question || input.trim()
        if (!q || loading) return
        setInput('')
        setMessages(prev => [...prev, { role: 'user', text: q, ts: Date.now() }])
        setLoading(true)
        try {
            const res = await axios.post(`${API}/agent/chat`, { question: q, session_id: sessionId })
            setMessages(prev => [...prev, {
                role: 'agent',
                text: res.data.answer,
                ms: res.data.latency_ms,
                ts: Date.now(),
            }])
        } catch {
            setMessages(prev => [...prev, {
                role: 'agent',
                text: '⚠️ Could not reach the API. Make sure the FastAPI server is running on port 8000.',
                ts: Date.now(),
            }])
        } finally {
            setLoading(false)
        }
    }

    const handleKey = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault()
            send()
        }
    }

    return (
        <div className="chat-view">
            <div className="messages">
                {messages.length === 0 && (
                    <div className="welcome">
                        <div className="welcome-icon">🤖</div>
                        <h2>AI MLOps Agent</h2>
                        <p>Ask anything about your local knowledge base. The agent will search, reason, and respond — all running locally on your Mac.</p>
                        <div className="suggestions">
                            {SUGGESTIONS.map(s => (
                                <button key={s} className="suggestion-chip" onClick={() => send(s)}>{s}</button>
                            ))}
                        </div>
                    </div>
                )}

                {messages.map((m, i) => (
                    <div key={i} className={`message ${m.role}`}>
                        <div className="avatar">{m.role === 'user' ? '👤' : '🤖'}</div>
                        <div>
                            <div className="bubble">
                                {m.role === 'agent'
                                    ? <ReactMarkdown>{m.text}</ReactMarkdown>
                                    : <p>{m.text}</p>}
                            </div>
                            {m.ms && <div className="meta">⏱ {m.ms}ms · via Ollama → LangChain → ChromaDB</div>}
                        </div>
                    </div>
                ))}

                {loading && (
                    <div className="message agent">
                        <div className="avatar">🤖</div>
                        <div className="bubble">
                            <div className="typing">
                                <span /><span /><span />
                            </div>
                        </div>
                    </div>
                )}
                <div ref={bottomRef} />
            </div>

            <div className="input-bar">
                <div className="input-row">
                    <textarea
                        className="chat-input"
                        placeholder="Ask the agent anything… (Shift+Enter for new line)"
                        value={input}
                        rows={1}
                        onChange={e => setInput(e.target.value)}
                        onKeyDown={handleKey}
                    />
                    <button className="send-btn" onClick={() => send()} disabled={loading || !input.trim()}>
                        ➤
                    </button>
                </div>
            </div>
        </div>
    )
}

// ── Dashboard View ─────────────────────────────────────────────────────────────
function DashboardView() {
    const [topics, setTopics] = useState([])
    const [apiOk, setApiOk] = useState(null)

    useEffect(() => {
        axios.get(`${API}/health`).then(() => setApiOk(true)).catch(() => setApiOk(false))
        axios.get(`${API}/knowledge/topics`).then(r => setTopics(r.data.topics || [])).catch(() => { })
    }, [])

    return (
        <div className="dashboard-view">
            <div className="dash-title">📊 Dashboard</div>

            <div className="cards-grid">
                <div className="card">
                    <div className="card-label">API Status</div>
                    <div className={`card-value ${apiOk === true ? 'green' : apiOk === false ? '' : 'yellow'}`}>
                        {apiOk === true ? '● Online' : apiOk === false ? '✗ Offline' : '…'}
                    </div>
                </div>
                <div className="card">
                    <div className="card-label">Knowledge Articles</div>
                    <div className="card-value accent">{topics.length}</div>
                </div>
                <div className="card">
                    <div className="card-label">LLM</div>
                    <div className="card-value accent" style={{ fontSize: '16px', marginTop: '6px' }}>LLaMA 3<br /><span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>via Ollama (local)</span></div>
                </div>
                <div className="card">
                    <div className="card-label">Vector Store</div>
                    <div className="card-value accent" style={{ fontSize: '16px', marginTop: '6px' }}>ChromaDB<br /><span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>nomic embeddings</span></div>
                </div>
            </div>

            <div style={{ marginBottom: '32px' }}>
                <div className="card-label" style={{ marginBottom: '12px', fontSize: '12px', textTransform: 'uppercase', letterSpacing: '1px' }}>📚 Knowledge Base Topics</div>
                {topics.length === 0
                    ? <p style={{ color: 'var(--text-muted)', fontSize: '13px' }}>No topics loaded. Run <code>dvc repro</code> to build the pipeline.</p>
                    : <div className="topic-list">
                        {topics.map(t => (
                            <div key={t.id} className="topic-item">
                                <span className="topic-icon">📄</span>
                                <div>
                                    <div className="topic-title">{t.title}</div>
                                    <div className="topic-tags">{t.tags.map(tag => <span key={tag} className="tag">{tag}</span>)}</div>
                                </div>
                            </div>
                        ))}
                    </div>
                }
            </div>

            <div>
                <div className="card-label" style={{ marginBottom: '12px', fontSize: '12px', textTransform: 'uppercase', letterSpacing: '1px' }}>🔄 Pipeline Steps</div>
                <div className="pipeline-steps">
                    {PIPELINE_STEPS.map((s, i) => (
                        <div key={i} className="step-item">
                            <div className={`step-dot ${s.done ? 'done' : 'pending'}`} />
                            <div>
                                <div className="step-label">{s.label}</div>
                                <div className="step-cmd">{s.cmd}</div>
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    )
}

// ── Shell ──────────────────────────────────────────────────────────────────────
export default function App() {
    const [view, setView] = useState('chat')

    return (
        <div className="app">
            <header className="topbar">
                <span className="topbar-logo">🧠 AI MLOps Local</span>
                <span className="topbar-sub">LLaMA 3 · ChromaDB · LangChain · LangFuse</span>
                <div className="status-dot" title="Services running" />
            </header>

            <nav className="sidebar">
                <span className="sidebar-section">Navigation</span>
                {[
                    { id: 'chat', icon: '💬', label: 'Chat Agent' },
                    { id: 'dashboard', icon: '📊', label: 'Dashboard' },
                ].map(n => (
                    <button key={n.id} className={`nav-btn ${view === n.id ? 'active' : ''}`} onClick={() => setView(n.id)}>
                        <span className="icon">{n.icon}</span>{n.label}
                    </button>
                ))}

                <span className="sidebar-section">Observability</span>
                {[
                    { href: 'http://localhost:3000', icon: '🔭', label: 'LangFuse' },
                    { href: 'http://localhost:4200', icon: '⚙️', label: 'Prefect UI' },
                    { href: 'http://localhost:8000/docs', icon: '📖', label: 'API Docs' },
                ].map(l => (
                    <a key={l.href} href={l.href} target="_blank" rel="noopener noreferrer"
                        style={{ textDecoration: 'none' }}>
                        <button className="nav-btn">
                            <span className="icon">{l.icon}</span>{l.label}
                        </button>
                    </a>
                ))}
            </nav>

            <main className="main">
                {view === 'chat' && <ChatView />}
                {view === 'dashboard' && <DashboardView />}
            </main>
        </div>
    )
}
