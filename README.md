# ai-backend-service

A production-pattern FastAPI service that integrates LLM inference and vector similarity search into scalable REST endpoints. Built with FastAPI, OpenAI, and ChromaDB.

> **Live demo cold start:** Deployed on Render free tier. First request after 15 minutes of inactivity takes 30–60 seconds. This is expected behaviour — the service spins down when idle.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    FastAPI Application                   │
│                                                         │
│  ┌──────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ /health  │  │ /api/analyze │  │  /api/documents  │  │
│  │          │  │ /api/rag     │  │  /api/search     │  │
│  └──────────┘  └──────┬───────┘  └────────┬─────────┘  │
│                       │                   │             │
│              ┌────────▼────────┐  ┌───────▼──────────┐  │
│              │   LLM Service   │  │  Vector Service  │  │
│              │                 │  │                  │  │
│              │ - Retry logic   │  │ - Embeddings     │  │
│              │ - Streaming     │  │ - Similarity     │  │
│              │ - JSON output   │  │   search         │  │
│              │ - Token tracking│  │ - CRUD ops       │  │
│              └────────┬────────┘  └────────┬─────────┘  │
│                       │                   │             │
└───────────────────────┼───────────────────┼─────────────┘
                        │                   │
                ┌───────▼───────┐   ┌───────▼────────┐
                │  OpenAI API   │   │   ChromaDB     │
                │ (gpt-4o-mini) │   │ (local/persist)│
                └───────────────┘   └────────────────┘
```

---

## Engineering Decisions

| Decision | Choice | Why |
|---|---|---|
| Async throughout | `async def` on all route handlers | LLM API calls are I/O-bound; blocking handlers would stall the entire event loop under concurrent requests |
| Pydantic on every endpoint | Request + response models in `models.py` | Catches malformed inputs before they reach the LLM; doubles as live API documentation in Swagger |
| ChromaDB local | In-process vector store | No external service dependency for a portfolio project; swap to Pinecone or Qdrant in production with one config change |
| Provider abstraction | LLM calls isolated in `llm_service.py` | Route handlers never import OpenAI directly — swap provider without touching business logic |
| Tenacity for retries | Exponential backoff on LLM calls | OpenAI returns 429 rate-limit errors under load; naive code fails silently; retry logic keeps the pipeline reliable |
| Centralised config | `config.py` with env overrides | Same codebase runs in dev, staging, and production — no code changes, just different `.env` files |
| Request timing middleware | `middleware.py` logs every request duration | LLM calls take 2–10 seconds; without latency tracking you cannot detect when OpenAI is degrading or when you need to scale |
| Docker | `Dockerfile` with layered caching | `requirements.txt` copied before source code — pip install layer is cached unless dependencies change |

---

## What This Demonstrates

| Pattern | Where | Why It Matters |
|---|---|---|
| LLM API integration | `llm_service.py` | Calling inference APIs from a backend service |
| Retry + backoff | `llm_service.py` | Handling rate limits and transient API failures |
| Streaming responses | `llm_service.py` | Real-time output for chat interfaces |
| Structured JSON output | `llm_service.py` | Reliable parsing of LLM responses |
| Vector embeddings | `vector_service.py` | Converting text to searchable vectors |
| Similarity search | `vector_service.py` | Finding documents by meaning, not keywords |
| RAG pipeline | `main.py /api/rag` | Grounding LLM answers in real data |
| Health monitoring | `main.py /health` | Production readiness with dependency checks |
| Batch operations | `vector_service.py` | Efficient bulk document ingestion |
| Pydantic validation | `models.py` | Type-safe request/response contracts |

---

## Project Structure

```
ai-backend-service/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app — all route definitions
│   ├── llm_service.py       # OpenAI integration: retry, streaming, JSON output
│   ├── vector_service.py    # ChromaDB: embed, search, CRUD
│   ├── models.py            # Pydantic schemas for all request/response types
│   ├── config.py            # Centralised settings with env variable overrides
│   └── middleware.py        # Request timing and structured logging
├── seed_data.py             # Load sample documents and run smoke tests
├── inspect_db.py            # Inspect ChromaDB contents during development
├── test_service.py          # Integration tests for all endpoints
├── Dockerfile               # Container build with layered cache optimisation
├── requirements.txt
├── .env.example             # Template for API key configuration
├── .gitignore
└── README.md
```

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/abhijeet-waikar/ai-backend-service.git
cd ai-backend-service
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Add your OPENAI_API_KEY — get one at https://platform.openai.com/api-keys
```

### 3. Run

```bash
uvicorn app.main:app --reload --port 8000
```

API docs: **http://localhost:8000/docs**

### 4. Load sample data

```bash
python seed_data.py
```

### 5. Run with Docker

```bash
docker build -t ai-backend-service .
docker run -p 8000:8000 --env-file .env ai-backend-service
```

---

## API Endpoints

**Health check**
```bash
curl http://localhost:8000/health
```

**Analyze text (LLM)**
```bash
curl -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Revenue grew 13% driven by cross-border volumes.", "instruction": "Summarise the key financial takeaways"}'
```

**Add document**
```bash
curl -X POST http://localhost:8000/api/documents \
  -H "Content-Type: application/json" \
  -d '{"content": "Contactless payments grew 40% year-over-year.", "metadata": {"source": "quarterly_report", "year": 2024}}'
```

**Semantic search**
```bash
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "How is AI used in fraud prevention?", "top_k": 3}'
```

**RAG query (retrieval + LLM answer)**
```bash
curl -X POST http://localhost:8000/api/rag \
  -H "Content-Type: application/json" \
  -d '{"question": "What role does generative AI play in customer service?", "top_k": 3}'
```

---

## Key Concepts

**RAG (Retrieval-Augmented Generation)** solves the core LLM problem of hallucination and knowledge cutoff. Instead of asking the model to recall facts from training, RAG retrieves relevant documents from your own data and injects them into the prompt as grounding context. This is the dominant pattern in production enterprise AI today.

**Vector embeddings** convert text into numerical arrays that encode semantic meaning. Similar meanings produce similar vectors. This enables search by concept rather than exact keyword — "payment fraud" and "transaction security" return the same results even though they share no words.

**Exponential backoff** is not optional for LLM APIs. Under load, OpenAI returns 429 rate-limit errors. A service that does not retry will fail silently and return errors to users. Tenacity handles the retry loop, backoff timing, and max attempt limit so the route handler stays clean.

---

## Tech Stack

| Component | Technology |
|---|---|
| Framework | FastAPI + Uvicorn |
| Language | Python 3.11 |
| LLM | OpenAI gpt-4o-mini |
| Vector DB | ChromaDB (local, persistent) |
| Validation | Pydantic v2 |
| Retry | Tenacity |
| Container | Docker |
| Deployment | Render |

---

## Roadmap

- [ ] Ragas evaluation metrics (faithfulness, answer relevancy scores)
- [ ] Semantic caching — skip LLM call for near-duplicate questions
- [ ] Streaming responses via SSE
- [ ] Swap ChromaDB → Qdrant for production-grade ANN search

---

## Author

**Abhijeet Waikar** — Senior Software Engineer | Java · Python · GCP
11 years building production data platforms. Transitioning into AI backend engineering — building LLM-integrated systems on GitHub.

[LinkedIn](https://linkedin.com/in/abhijeet-waikar-developer) · [GitHub](https://github.com/abhijeet-waikar)
