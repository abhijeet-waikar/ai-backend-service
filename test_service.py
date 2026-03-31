"""
test_service.py - Tests for the AI backend service.

Run: python -m pytest test_service.py -v
"""

import os
import pytest
from fastapi.testclient import TestClient


# ── Vector Service Unit Tests (no API key needed) ──────────

class TestVectorService:
    """Tests for vector store operations — runs without any API keys."""

    def setup_method(self):
        from app.vector_service import VectorService
        self.vs = VectorService()  # In-memory mode
        self.vs.reset_collection()  # Clean state for each test

    def test_add_document(self):
        doc_id = self.vs.add_document(
            "Test payment document",
            {"source": "test"}
        )
        assert doc_id is not None
        assert self.vs.get_document_count() == 1

    def test_add_batch(self):
        docs = [
            {"content": "Fraud detection uses ML models", "metadata": {"topic": "fraud"}},
            {"content": "Tokenization secures card data", "metadata": {"topic": "security"}},
        ]
        ids = self.vs.add_documents_batch(docs)
        assert len(ids) == 2
        assert self.vs.get_document_count() == 2

    def test_search_returns_results(self):
        self.vs.add_document(
            "Machine learning detects fraudulent transactions",
            {"topic": "fraud"}
        )
        self.vs.add_document(
            "Contactless payments use NFC technology",
            {"topic": "payments"}
        )
        self.vs.add_document(
            "Cross border transfers require compliance checks",
            {"topic": "compliance"}
        )

        results = self.vs.search("How does AI prevent fraud?", top_k=2)
        assert len(results) == 2
        assert results[0]["relevance_score"] >= results[1]["relevance_score"]

    def test_search_empty_collection(self):
        results = self.vs.search("anything")
        assert results == []

    def test_reset_collection(self):
        self.vs.add_document("Test document", {"source": "test"})
        assert self.vs.get_document_count() == 1
        self.vs.reset_collection()
        assert self.vs.get_document_count() == 0

    def test_health_check(self):
        assert self.vs.is_available() is True


# ── API Integration Tests ──────────────────────────────────

class TestAPIEndpoints:
    """Tests for FastAPI endpoints — manually injects services."""

    def setup_method(self):
        # Import app and services
        import app.main as main_module
        from app.vector_service import VectorService

        # Manually initialize vector_service (bypasses lifespan)
        main_module.vector_service = VectorService()  # In-memory
        main_module.vector_service.reset_collection()

        # LLM service stays None — LLM endpoints will return 503 (expected)
        main_module.llm_service = None

        self.client = TestClient(main_module.app)

    def test_health_endpoint(self):
        response = self.client.get("/health")
        assert response.status_code == 200
        data = response.json()
        # LLM is unavailable (no real key), vector is connected
        assert data["vector_service"] == "connected"
        assert "version" in data

    def test_add_document_endpoint(self):
        response = self.client.post("/api/documents", json={
            "content": "Test payment processing document",
            "metadata": {"source": "test"}
        })
        assert response.status_code == 200
        assert "document_id" in response.json()

    def test_add_document_validation_error(self):
        response = self.client.post("/api/documents", json={
            "content": "",
            "metadata": {}
        })
        assert response.status_code == 422

    def test_search_endpoint(self):
        # Add a document first
        self.client.post("/api/documents", json={
            "content": "Fraud detection in real-time payment systems",
            "metadata": {"source": "test"}
        })
        response = self.client.post("/api/search", json={
            "query": "fraud prevention",
            "top_k": 3
        })
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert len(data["results"]) >= 1
        assert "total_documents" in data

    def test_batch_documents_endpoint(self):
        response = self.client.post("/api/documents/batch", json={
            "documents": [
                {"content": "Tokenization secures payments", "metadata": {"topic": "security"}},
                {"content": "AI detects fraud patterns", "metadata": {"topic": "fraud"}}
            ]
        })
        assert response.status_code == 200
        data = response.json()
        assert "document_ids" in data
        assert len(data["document_ids"]) == 2

    def test_reset_endpoint(self):
        # Add then reset
        self.client.post("/api/documents", json={
            "content": "Temp document",
            "metadata": {"source": "test"}
        })
        response = self.client.post("/api/reset")
        assert response.status_code == 200
        assert response.json()["total_documents"] == 0

    def test_analyze_returns_503_without_llm(self):
        """LLM endpoints should return 503 when no API key is configured."""
        response = self.client.post("/api/analyze", json={
            "text": "Test text",
            "instruction": "Summarize"
        })
        assert response.status_code == 503

    def test_rag_returns_503_without_llm(self):
        """RAG endpoint should return 503 when no API key is configured."""
        response = self.client.post("/api/rag", json={
            "question": "Test question"
        })
        assert response.status_code == 503


# ── LLM Tests (only run if real API key is set) ───────────

@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY", "").startswith("sk-test"),
    reason="Real OPENAI_API_KEY not set — skipping LLM tests"
)
class TestLLMEndpoints:
    """Tests that require a real OpenAI API key."""

    def setup_method(self):
        import app.main as main_module
        from app.vector_service import VectorService
        from app.llm_service import LLMService

        main_module.vector_service = VectorService()
        main_module.llm_service = LLMService(api_key=os.getenv("OPENAI_API_KEY"))

        self.client = TestClient(main_module.app)

    def test_analyze_endpoint(self):
        response = self.client.post("/api/analyze", json={
            "text": "Mastercard revenue grew 13 percent in Q3 2025.",
            "instruction": "Summarize the key point"
        })
        assert response.status_code == 200
        data = response.json()
        assert "summary" in data
        assert "key_points" in data
        assert "model_used" in data

    def test_rag_endpoint(self):
        self.client.post("/api/documents", json={
            "content": "AI chatbots handle payment disputes using retrieval-augmented generation.",
            "metadata": {"source": "test"}
        })
        response = self.client.post("/api/rag", json={
            "question": "How is AI used in dispute resolution?"
        })
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data