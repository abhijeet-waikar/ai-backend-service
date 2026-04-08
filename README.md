# AI Backend Service

A production-pattern backend service that integrates **LLM inference APIs** and **vector similarity search** into scalable REST endpoints. Built with **FastAPI**, **OpenAI**, and **ChromaDB**.

This project demonstrates how backend engineers can productionize AI capabilities — the same patterns used by teams building AI-powered features at companies like Mastercard, Google, and Amazon.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    FastAPI Application                    │
│                                                          │
│  ┌──────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │  /health  │  │  /api/analyze │  │  /api/documents   │  │
│  │          │  │  /api/rag     │  │  /api/search      │  │
│  └──────────┘  └──────┬───────┘  └─────────┬─────────┘  │
│                       │                     │            │
│              ┌────────▼────────┐  ┌────────▼─────────┐  │
│              │   LLM Service   │  │  Vector Service   │  │
│              │                 │  │                   │  │
│              │ - Retry logic   │  │ - Embeddings      │  │
│              │ - Streaming     │  │ - Similarity      │  │
│              │ - JSON output   │  │   search          │  │
│              │ - Token tracking│  │ - CRUD ops        │  │
│              └────────┬────────┘  └────────┬─────────┘  │
│                       │                     │            │
└───────────────────────┼─────────────────────┼────────────┘
                        │                     │
                ┌───────▼───────┐    ┌───────▼────────┐
                │  OpenAI API   │    │   ChromaDB      │
                │  (gpt-4o-mini)│    │  (local/persist)│
                └───────────────┘    └────────────────┘
```

---

## What This Demonstrates

| Pattern | Where | Why It Matters |
|---------|-------|----------------|
| **LLM API Integration** | `llm_service.py` | Calling inference APIs from backend services |
| **Retry + Backoff** | `llm_service.py` | Handling rate limits and transient API failures |
| **Streaming Responses** | `llm_service.py` | Real-time output for chat interfaces |
| **Structured JSON Output** | `llm_service.py` | Reliable parsing of LLM responses |
| **Vector Embeddings** | `vector_service.py` | Converting text to searchable vectors |
| **Similarity Search** | `vector_service.py` | Finding documents by meaning, not keywords |
| **RAG Pipeline** | `main.py /api/rag` | Grounding LLM answers in real data |
| **Health Monitoring** | `main.py /health` | Production readiness with dependency checks |
| **Batch Operations** | `vector_service.py` | Efficient bulk document ingestion |
| **Pydantic Validation** | `models.py` | Type-safe request/response contracts |

---

## Project Structure

```
ai-backend-service/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app with all endpoints
│   ├── llm_service.py       # OpenAI integration (retry, streaming, JSON)
│   ├── vector_service.py    # ChromaDB operations (embed, search, CRUD)
│   └── models.py            # Pydantic schemas for all requests/responses
├── seed_data.py             # Load sample documents + run tests
├── requirements.txt         # Python dependencies
├── .env.example             # Template for API key configuration
├── .gitignore
└── README.md
```

---

## Quick Start (Mac)

### 1. Clone and setup

```bash
git clone https://github.com/abhijeet-waikar/ai-backend-service.git
cd ai-backend-service

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API key

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
# Get one at: https://platform.openai.com/api-keys
```

### 3. Start the server

```bash
uvicorn app.main:app --reload --port 8000
```

You should see:
```
INFO | LLM Service started
INFO | Vector Service started
INFO | Application startup complete.
```

### 4. Load sample data and test

In a new terminal:
```bash
source venv/bin/activate
python seed_data.py
```

### 5. Explore the API

Open **http://localhost:8000/docs** in your browser for the interactive Swagger UI.

---

## API Endpoints

### Health Check
```bash
curl http://localhost:8000/health
```

### Analyze Text (LLM)
```bash
curl -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Mastercard reported Q3 revenue growth of 13%, driven by strong cross-border volumes and value-added services expansion.",
    "instruction": "Summarize the key financial takeaways"
  }'
```

### Add Document
```bash
curl -X POST http://localhost:8000/api/documents \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Contactless payments grew 40% year-over-year in emerging markets.",
    "metadata": {"source": "quarterly_report", "year": 2024}
  }'
```

### Semantic Search
```bash
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How is AI used in fraud prevention?",
    "top_k": 3
  }'
```

### RAG Query (Search + LLM Answer)
```bash
curl -X POST http://localhost:8000/api/rag \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What role does generative AI play in customer service for payments?",
    "top_k": 3
  }'
```

---

## Key Concepts Explained

### What is RAG (Retrieval-Augmented Generation)?

RAG solves a fundamental LLM problem: models have a knowledge cutoff and can hallucinate. RAG grounds the model's answers in your actual data:

1. **User asks a question**
2. **Retrieval**: Vector search finds the most relevant documents from your database
3. **Augmentation**: Those documents are injected into the LLM prompt as context
4. **Generation**: The LLM generates an answer using ONLY the provided context

This is the #1 pattern in enterprise AI applications today.

### What are Vector Embeddings?

Text embeddings convert words into numerical vectors (arrays of numbers) that capture semantic meaning. Similar meanings = similar vectors. This enables searching by meaning rather than exact keyword matching.

### Why Retry with Exponential Backoff?

AI API calls can fail due to rate limits, network issues, or service outages. Retry with exponential backoff (wait 1s, then 2s, then 4s) handles transient failures gracefully without overwhelming the API.

---

## Tech Stack

- **Python 3.11+** - Backend language
- **FastAPI** - High-performance async web framework
- **OpenAI API** - LLM inference (gpt-4o-mini)
- **ChromaDB** - Local vector database for embeddings and similarity search
- **Pydantic** - Data validation and serialization
- **Tenacity** - Retry logic with exponential backoff
- **Uvicorn** - ASGI server

---

## Author

**Abhijeet Sandeep Waikar** — Backend Engineer | GCP Certified | Building AI-integrated systems

- [LinkedIn](https://www.linkedin.com/in/abhijeet-waikar-developer)
- [GitHub](https://github.com/abhijeet-waikar)
