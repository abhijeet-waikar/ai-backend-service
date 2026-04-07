import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

from app.llm_service import LLMService
from app.vector_service import VectorService
from app.models import (
    AnalyzeRequest, AnalyzeResponse,
    DocumentInput, BatchDocumentInput,
    SearchRequest, SearchResponse, SearchResult,
    RAGRequest, RAGResponse,
    HealthResponse,
)

# ── Setup ──────────────────────────────────────────────────────
from app.config import settings
#load_dotenv()  # Load .env file for API keys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Service Initialization ─────────────────────────────────────

# Services are initialized once at startup and shared across requests
llm_service: LLMService | None = None
vector_service: VectorService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifecycle management.
    Services are created on startup and cleaned up on shutdown.
    """
    global llm_service, vector_service

    # Startup
    api_key = settings.OPENAI_API_KEY
    if not api_key:
        logger.error("OPENAI_API_KEY not set. LLM features will be unavailable.")
        llm_service = None
    else:
        llm_service = LLMService(api_key=api_key)
        logger.info("LLM Service started")

    vector_service = VectorService(persist_directory="./chroma_data")
    logger.info("Vector Service started")

    yield  # Application runs here

    # Shutdown
    logger.info("Services shutting down")


# ── FastAPI App ────────────────────────────────────────────────

app = FastAPI(
    title="AI Backend Service",
    description=(
        "A production-pattern backend service integrating LLM inference "
        "and vector similarity search. Demonstrates AI integration patterns "
        "for enterprise backend systems."
    ),
    version=settings.API_VERSION,
    lifespan=lifespan,
)


# ── Health Check ───────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """
    Service health endpoint with dependency status.

    Production pattern: Always check downstream dependencies
    so monitoring tools can detect partial failures.
    """
    llm_ok = llm_service.is_available() if llm_service else False
    vector_ok = vector_service.is_available() if vector_service else False
    doc_count = vector_service.get_document_count() if vector_service else 0

    overall = "healthy" if (llm_ok and vector_ok) else "degraded"

    return HealthResponse(
        status=overall,
        llm_service="connected" if llm_ok else "unavailable",
        vector_service="connected" if vector_ok else "unavailable",
        total_documents=doc_count,
        version=settings.API_VERSION,
    )


# ── LLM Analyze Endpoint ──────────────────────────────────────

@app.post("/api/analyze", response_model=AnalyzeResponse, tags=["LLM"])
def analyze_text(request: AnalyzeRequest):
    """
    Analyze text using LLM and return structured JSON output.

    Demonstrates:
    - Calling inference APIs from a backend service
    - Structured output parsing (JSON mode)
    - Retry with exponential backoff on transient failures
    - Token usage tracking for cost monitoring
    """
    if not llm_service:
        raise HTTPException(
            status_code=503,
            detail="LLM service unavailable. Check OPENAI_API_KEY configuration.",
        )

    try:
        result = llm_service.analyze_text(
            text=request.text,
            instruction=request.instruction,
        )
        return AnalyzeResponse(**result)

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


# ── Document Management ───────────────────────────────────────

@app.post("/api/documents", tags=["Vector Store"])
def add_document(doc: DocumentInput):
    """
    Add a single document to the vector store.

    The document text is automatically converted to an embedding vector
    and stored alongside its metadata for future similarity search.
    """
    if not vector_service:
        raise HTTPException(status_code=503, detail="Vector service unavailable")

    doc_id = vector_service.add_document(
        content=doc.content,
        metadata=doc.metadata,
    )
    return {
        "message": "Document added successfully",
        "document_id": doc_id,
        "total_documents": vector_service.get_document_count(),
    }


@app.post("/api/documents/batch", tags=["Vector Store"])
def add_documents_batch(batch: BatchDocumentInput):
    """
    Add multiple documents in a single batch operation.

    More efficient than individual adds for bulk ingestion.
    """
    if not vector_service:
        raise HTTPException(status_code=503, detail="Vector service unavailable")

    docs = [{"content": d.content, "metadata": d.metadata} for d in batch.documents]
    ids = vector_service.add_documents_batch(docs)

    return {
        "message": f"{len(ids)} documents added successfully",
        "document_ids": ids,
        "total_documents": vector_service.get_document_count(),
    }


# ── Semantic Search ────────────────────────────────────────────

@app.post("/api/search", response_model=SearchResponse, tags=["Vector Store"])
def search_documents(request: SearchRequest):
    """
    Search documents by semantic similarity.

    Unlike keyword search (SQL LIKE), this finds documents with similar
    MEANING even if they use completely different words.

    Example:
        Query: "payment failed"
        Finds: "transaction was declined due to insufficient balance"
        (Same meaning, different words)
    """
    if not vector_service:
        raise HTTPException(status_code=503, detail="Vector service unavailable")

    results = vector_service.search(query=request.query, top_k=request.top_k)

    return SearchResponse(
        query=request.query,
        results=[SearchResult(**r) for r in results],
        total_documents=vector_service.get_document_count(),
    )


# ── RAG: Retrieval-Augmented Generation ────────────────────────

@app.post("/api/rag", response_model=RAGResponse, tags=["RAG"])
def rag_query(request: RAGRequest):
    """
    Answer a question using Retrieval-Augmented Generation (RAG).

    RAG combines two steps:
    1. RETRIEVAL: Find relevant documents from the vector store
    2. GENERATION: Send those documents as context to the LLM to generate an answer

    Why RAG matters:
    - LLMs have a knowledge cutoff and can hallucinate
    - RAG grounds the LLM's answer in YOUR actual data
    - The LLM only uses the retrieved documents, reducing hallucination
    - You can update the knowledge base without retraining the model

    This is the #1 pattern used in enterprise AI applications today.
    """
    if not llm_service:
        raise HTTPException(status_code=503, detail="LLM service unavailable")
    if not vector_service:
        raise HTTPException(status_code=503, detail="Vector service unavailable")

    # Step 1: Retrieve relevant documents
    search_results = vector_service.search(
        query=request.question,
        top_k=request.top_k,
    )

    if not search_results:
        raise HTTPException(
            status_code=404,
            detail="No documents found. Add documents first via /api/documents",
        )

    # Step 2: Generate answer using retrieved context
    context_docs = [r["content"] for r in search_results]
    llm_result = llm_service.generate_rag_answer(
        question=request.question,
        context_docs=context_docs,
    )

    return RAGResponse(
        question=request.question,
        answer=llm_result["answer"],
        sources=[SearchResult(**r) for r in search_results],
        model_used=llm_result["model_used"],
    )


# ── Reset (Development Only) ──────────────────────────────────

@app.post("/api/reset", tags=["Admin"])
def reset_collection():
    """Delete all documents. For development/testing only."""
    if not vector_service:
        raise HTTPException(status_code=503, detail="Vector service unavailable")

    vector_service.reset_collection()
    return {"message": "All documents deleted", "total_documents": 0}

#__ Insert data into the applicaiton ___________________________

@app.post("/api/seed")
def seed_data():
    """One-time endpoint to seed initial data into ChromaDB"""
    try:
        from seed_data import run_seed
        run_seed(base_url="http://localhost:8000")  # sync call, no await
        return {"message": "Data seeded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
