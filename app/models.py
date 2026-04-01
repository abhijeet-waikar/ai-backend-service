"""
models.py - Request and Response schemas for the API.
"""

from pydantic import BaseModel, Field
from typing import Optional


# ── LLM Endpoints ──────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    """Request body for the /api/analyze endpoint."""
    text: str = Field(..., min_length=1, max_length=5000, description="Text to analyze")
    instruction: str = Field(
        default="Summarize the key points and provide actionable insights.",
        description="What the LLM should do with the text"
    )

    model_config = {"json_schema_extra": {"examples": [
        {"text": "Mastercard reported Q3 revenue of $7.4 billion...", "instruction": "Summarize key financials"}
    ]}}


class AnalyzeResponse(BaseModel):
    """Structured response from the /api/analyze endpoint."""
    summary: str = Field(..., description="LLM-generated analysis")
    key_points: list[str] = Field(default_factory=list, description="Extracted key points")
    sentiment: str = Field(default="neutral", description="Overall sentiment: positive, negative, or neutral")
    model_used: str = Field(..., description="Which model processed this request")
    tokens_used: int = Field(default=0, description="Total tokens consumed")


# ── Vector Search Endpoints ────────────────────────────────────

class DocumentInput(BaseModel):
    """Request body for adding a document to the vector store."""
    content: str = Field(..., min_length=1, max_length=10000, description="Document text content")
    metadata: dict = Field(default_factory=dict, description="Optional metadata (source, author, date, etc.)")


class BatchDocumentInput(BaseModel):
    """Request body for adding multiple documents at once."""
    documents: list[DocumentInput] = Field(..., min_length=1, max_length=50)


class SearchRequest(BaseModel):
    """Request body for the /api/search endpoint."""
    query: str = Field(..., min_length=1, max_length=1000, description="Natural language search query")
    top_k: int = Field(default=3, ge=1, le=10, description="Number of results to return")


class SearchResult(BaseModel):
    """A single search result with relevance score."""
    content: str
    metadata: dict
    relevance_score: float = Field(..., description="Similarity score (0.0 to 1.0, higher is more relevant)")


class SearchResponse(BaseModel):
    """Response from the /api/search endpoint."""
    query: str
    results: list[SearchResult]
    total_documents: int = Field(..., description="Total documents in the collection")


# ── RAG (Retrieve + Generate) ─────────────────────────────────

class RAGRequest(BaseModel):
    """Request body for the /api/rag endpoint - combines search + LLM."""
    question: str = Field(..., min_length=1, max_length=1000, description="Question to answer using stored documents")
    top_k: int = Field(default=3, ge=1, le=10, description="Number of context documents to retrieve")


class RAGResponse(BaseModel):
    """Response from the /api/rag endpoint."""
    question: str
    answer: str = Field(..., description="LLM-generated answer grounded in retrieved documents")
    sources: list[SearchResult] = Field(..., description="Documents used as context for the answer")
    model_used: str


# ── Health ─────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    """Response from the /health endpoint."""
    status: str
    llm_service: str
    vector_service: str
    total_documents: int
    version: str
