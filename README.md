# 🧠 Econ Intelligence Agent

> **The AI layer of the portfolio.** Claude-powered macro analyst that answers economic questions by querying your own PostgreSQL data, running ML forecasts from the econ-ml-platform, and doing RAG over Federal Reserve publications, IMF Working Papers, and FOMC meeting minutes.

![Claude](https://img.shields.io/badge/Claude-Opus-orange?logo=anthropic)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?logo=fastapi)
![RAG](https://img.shields.io/badge/RAG-pgvector%20%7C%20Chroma%20%7C%20Pinecone-blue)
![Next.js](https://img.shields.io/badge/Next.js-14-black?logo=next.js)

---

## 🏗️ Architecture

```
                    ┌──────────────────────────────────────────────────────┐
                    │                 CLAUDE OPUS                          │
                    │           (Anthropic Tool Use API)                   │
                    │                                                      │
                    │  Sees: system prompt + conversation history          │
                    │  Decides: which tools to call, in what order         │
                    │  Returns: grounded, cited analysis                   │
                    └───────────────────┬──────────────────────────────────┘
                                        │ tool calls
               ┌────────────────────────┼────────────────────────┐
               ▼                        ▼                        ▼
    ┌──────────────────┐   ┌────────────────────┐   ┌─────────────────────┐
    │  query_macro_data│   │   get_forecast     │   │  search_documents   │
    │                  │   │                    │   │                     │
    │  SELECT * FROM   │   │  POST /predict/    │   │  Semantic search    │
    │  fct_macro_      │   │  gdp_growth        │   │  over corpus:       │
    │  indicators_     │   │  unemployment      │   │  - FOMC minutes     │
    │  monthly         │   │  fed_funds         │   │  - Fed papers       │
    │                  │   │                    │   │  - IMF WPs          │
    │  PostgreSQL mart │   │  econ-ml-platform  │   │                     │
    └──────────────────┘   └────────────────────┘   │  Backends:          │
                                                    │  pgvector / Chroma  │
    ┌──────────────────┐   ┌────────────────────┐   │  / Pinecone         │
    │    calculate     │   │ get_recent_releases│   └─────────────────────┘
    │                  │   │                    │
    │  Python REPL:    │   │  streaming.        │
    │  numpy, pandas,  │   │  releases_enriched │
    │  scipy           │   │  (latest BLS/FRED/ │
    └──────────────────┘   │  BEA releases)     │
                           └────────────────────┘

    ┌─────────────────────────────────────────────────────────────────┐
    │                      INTERFACES                                 │
    │                                                                 │
    │   FastAPI (port 8001)     CLI (Rich terminal)  Next.js (3002)  │
    │   POST /chat (SSE)        python cli/chat.py   /frontend       │
    │   GET  /sessions/:id      slash commands       streaming UI    │
    └─────────────────────────────────────────────────────────────────┘
```

---

## 💬 Example Interactions

```
You: What is the unemployment trend over the last 12 months?

Agent: [calls query_macro_data]
       [calls calculate]

The unemployment rate has declined from 3.9% (Dec 2023) to 4.1% (Dec 2024),
a change of +0.2pp. The 12-month trend shows two distinct phases:
- Jan–Jun 2024: Stable near 3.7–3.9%, consistent with full employment
- Jul–Dec 2024: Gradual rise to 4.1%, reflecting cooling labor demand

MAE vs 12-month moving average: 0.14pp — low volatility period.

---

You: What's the probability the Fed cuts rates next month?

Agent: [calls get_forecast with target=fed_funds_direction]
       [calls search_documents with source_filter=fomc]
       [calls get_recent_releases with high_impact_only=True]

**FED FUNDS DIRECTION FORECAST**
Based on the ML model (XGBoost v4, trained through Nov 2024):
- Direction: **DOWN** (71% probability)
- Implied next rate: 4.25% (from current 4.50%)

**FOMC Context** [FOMC Minutes, Dec 2024]:
The December minutes indicated members were "increasingly confident"
inflation was returning to target, but cautioned that the pace of
cuts would be "data-dependent"...

**Recent Releases** (last 30 days):
- CPI Nov 2024: 2.7% (+3.8% surprise)
- PCE Nov 2024: 2.4% YoY (-0.4pp below Fed target)

**Synthesis:** The combination of below-target PCE and the model's
71% DOWN probability suggests a cut is likely, but the CPI beat
introduces uncertainty. Markets are pricing ~80% probability.
```

---

## 📁 Project Structure

```
econ-intelligence-agent/
├── agent/
│   ├── agent.py              Core: Claude + agentic loop + streaming
│   └── tools/
│       ├── macro_db.py       Tool 1: PostgreSQL SELECT queries
│       ├── forecast.py       Tool 2: ML platform /predict calls
│       ├── rag_search.py     Tool 3: Semantic search (Chroma/pgvector/Pinecone)
│       ├── calculator.py     Tool 4: Python REPL (numpy/pandas/scipy)
│       └── releases.py       Tool 5: Streaming releases feed
│
├── api/
│   └── main.py               FastAPI: /chat (SSE), /sessions, /health
│
├── ingestion/
│   └── ingest.py             FOMC/IMF document ingestion pipeline
│
├── cli/
│   └── chat.py               Rich terminal client with slash commands
│
├── frontend/
│   ├── src/app/
│   │   ├── page.tsx          Next.js chat UI with SSE streaming
│   │   ├── layout.tsx
│   │   └── globals.css
│   ├── package.json
│   └── tailwind.config.ts
│
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
│
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Prerequisites

```bash
# Required: Projects 1–4 running
# - PostgreSQL with fct_macro_indicators_monthly (Project 1)
# - econ-ml-platform at localhost:8000 (Project 4)
# Optional: econ-streaming-pipeline (Project 3) for live releases

# Set environment variables
export ANTHROPIC_API_KEY=sk-ant-...
export POSTGRES_USER=econ_user
export POSTGRES_PASSWORD=...
export POSTGRES_DB=econ_warehouse
export ML_API_URL=http://localhost:8000
export VECTOR_DB=chroma   # or pgvector or pinecone
```

### 2. Install and run API

```bash
cd econ-intelligence-agent
pip install -r requirements.txt

uvicorn api.main:app --port 8001 --reload
# → http://localhost:8001/docs
```

### 3. Ingest documents

```bash
# Ingest FOMC minutes for 2022–2024
python ingestion/ingest.py --source fomc --years 2022 2023 2024

# Add IMF working papers
python ingestion/ingest.py --source imf --query "inflation monetary policy"

# Ingest everything
python ingestion/ingest.py --source all --years 2020 2021 2022 2023 2024
```

### 4. CLI client

```bash
python cli/chat.py
# or
python cli/chat.py --api http://localhost:8001 --new
```

### 5. Next.js frontend

```bash
cd frontend
npm install
NEXT_PUBLIC_AGENT_API_URL=http://localhost:8001 npm run dev
# → http://localhost:3002
```

### 6. Docker

```bash
set -a && source ../.env && set +a
docker compose -f docker/docker-compose.yml up -d
```

---

## 🔧 Vector DB Configuration

Set `VECTOR_DB` env var to switch backends — no code changes needed.

| Backend | Config | Best for |
|---------|--------|----------|
| `chroma` | `CHROMA_PERSIST_DIR=./chroma_db` | Local dev, no infra |
| `pgvector` | Reuses existing PostgreSQL | Production, single DB |
| `pinecone` | `PINECONE_API_KEY`, `PINECONE_INDEX` | Cloud scale |

---

## 🔬 Design Decisions

**Why Anthropic tool use instead of LangChain?**
LangChain adds a heavy abstraction layer over what Anthropic's API does natively. Claude's tool use API is first-class — the model is specifically trained to use tools well. Direct API calls give full control over the agentic loop, streaming behaviour, and error handling without framework magic obscuring what's happening.

**Why support three vector DBs?**
Different portfolio reviewers will run this in different environments. Chroma works out of the box with zero infrastructure. pgvector demonstrates infrastructure reuse (one Postgres does everything). Pinecone shows cloud-native deployment. The `VECTOR_DB` env var switches backends without touching code.

**Why Voyage AI for embeddings?**
Anthropic specifically recommends Voyage AI embeddings when building RAG systems with Claude — they're trained to produce embeddings that work well with Claude's representations. `voyage-3` outperforms OpenAI's `text-embedding-3-small` on retrieval benchmarks, particularly for financial and economic text. OpenAI is supported as a fallback.

**Why SSE streaming instead of WebSockets?**
The agent's response is unidirectional (server → client) and sessions are short-lived. SSE is simpler than WebSockets, works through HTTP/2, and is natively supported by `fetch()` in the browser without a library. The CLI uses plain `httpx.stream()` for the same reason.

---

## 📈 Portfolio Positioning

This project is the **differentiator**. Every data engineering portfolio has pipelines and dashboards. Very few have:
- A working agentic AI that queries their own production data
- RAG over domain-specific documents (Fed/IMF corpus)
- Live ML forecast integration with natural language interface
- Multi-interface deployment (CLI + web + API)

In interviews: run the CLI live, ask "What's the probability the Fed cuts rates next month?" and watch it query the DB, call the ML API, search FOMC minutes, and synthesise a grounded answer with citations. That's a 10-minute demo that no one else has.

---

## 📄 License

MIT
