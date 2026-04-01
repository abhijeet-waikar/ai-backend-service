import logging
import uuid
from typing import Optional

import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────

COLLECTION_NAME = "documents"
# ChromaDB uses its own embedding model by default (all-MiniLM-L6-v2)
# In production, you'd use OpenAI embeddings or a custom model


class VectorService:

    def __init__(self, persist_directory: Optional[str] = None):
        if persist_directory:
            self.client = chromadb.PersistentClient(path=persist_directory)
            logger.info(f"Vector DB initialized with persistence at: {persist_directory}")
        else:
            self.client = chromadb.Client()
            logger.info("Vector DB initialized (in-memory mode)")

        # Get or create the collection
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"description": "AI Backend Service document store"},
        )
        logger.info(
            f"Collection '{COLLECTION_NAME}' ready. Documents: {self.collection.count()}"
        )

    def is_available(self) -> bool:
        """Health check for the vector service."""
        try:
            self.collection.count()
            return True
        except Exception as e:
            logger.warning(f"Vector service health check failed: {e}")
            return False

    def add_document(self, content: str, metadata: Optional[dict] = None) -> str:
        """
        Add a single document to the vector store.

        What happens internally:
        1. ChromaDB generates an embedding vector from the text
        2. The vector + original text + metadata are stored together
        3. A unique ID is assigned for future reference

        Args:
            content: The text content to store and make searchable
            metadata: Optional dict of structured data (source, author, date, etc.)

        Returns:
            The generated document ID
        """
        doc_id = str(uuid.uuid4())[:8]  # Short UUID for readability
        clean_metadata = metadata or {}

        # ChromaDB metadata values must be str, int, float, or bool
        clean_metadata = {
            k: str(v) if not isinstance(v, (str, int, float, bool)) else v
            for k, v in clean_metadata.items()
        }

        self.collection.add(
            ids=[doc_id],
            documents=[content],
            metadatas=[clean_metadata],
        )

        logger.info(f"Document added: id={doc_id}, length={len(content)} chars")
        return doc_id

    def add_documents_batch(
        self, documents: list[dict]
    ) -> list[str]:
        """
        Add multiple documents in a single batch operation.

        Batch operations are more efficient than individual adds because:
        - Fewer round-trips to the embedding model
        - Single transaction to the storage layer

        Args:
            documents: List of dicts with 'content' and optional 'metadata' keys

        Returns:
            List of generated document IDs
        """
        ids = [str(uuid.uuid4())[:8] for _ in documents]
        contents = [doc["content"] for doc in documents]
        metadatas = []
        for doc in documents:
            meta = doc.get("metadata", {})
            clean = {
                k: str(v) if not isinstance(v, (str, int, float, bool)) else v
                for k, v in meta.items()
            }
            metadatas.append(clean)

        self.collection.add(
            ids=ids,
            documents=contents,
            metadatas=metadatas,
        )

        logger.info(f"Batch added: {len(ids)} documents")
        return ids

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """
        Search for documents similar to the query using vector similarity.

        How similarity search works:
        1. Your query text is converted to a vector (embedding)
        2. ChromaDB computes the distance between query vector and all stored vectors
        3. The closest vectors (= most semantically similar documents) are returned
        4. Distance is converted to a relevance score (0.0 to 1.0)

        Args:
            query: Natural language search query
            top_k: Number of results to return (default 3)

        Returns:
            List of dicts with content, metadata, and relevance_score
        """
        if self.collection.count() == 0:
            logger.warning("Search attempted on empty collection")
            return []

        results = self.collection.query(
            query_texts=[query],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        search_results = []
        if results and results["documents"] and results["documents"][0]:
            for i in range(len(results["documents"][0])):
                # Convert distance to similarity score (0-1, higher = more similar)
                # ChromaDB uses L2 distance by default; lower distance = more similar
                distance = results["distances"][0][i]
                relevance_score = round(max(0.0, 1.0 - (distance / 2.0)), 4)

                search_results.append({
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "relevance_score": relevance_score,
                })

        logger.info(f"Search for '{query[:50]}...' returned {len(search_results)} results")
        return search_results

    def get_document_count(self) -> int:
        """Return total number of documents in the collection."""
        return self.collection.count()

    def reset_collection(self) -> None:
        """Delete all documents. Useful for testing."""
        self.client.delete_collection(COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"description": "AI Backend Service document store"},
        )
        logger.info("Collection reset - all documents deleted")
