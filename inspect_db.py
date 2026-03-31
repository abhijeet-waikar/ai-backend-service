"""
inspect_db.py - View and query all documents in ChromaDB.

Usage:
    python inspect_db.py              # Show all documents
    python inspect_db.py "your query" # Search for similar documents
"""

import sys
import chromadb

PERSIST_DIR = "./chroma_data"
COLLECTION_NAME = "documents"


def inspect():
    """Show all documents stored in ChromaDB."""
    client = chromadb.PersistentClient(path=PERSIST_DIR)

    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception:
        print("Collection not found. Start the server and add documents first.")
        return

    count = collection.count()
    print(f"\n{'='*60}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Total documents: {count}")
    print(f"{'='*60}")

    if count == 0:
        print("No documents stored yet.")
        return

    # Fetch all documents
    all_docs = collection.get(
        include=["documents", "metadatas"]
    )

    for i in range(len(all_docs["ids"])):
        doc_id = all_docs["ids"][i]
        content = all_docs["documents"][i]
        metadata = all_docs["metadatas"][i]

        print(f"\n--- Document {i+1} ---")
        print(f"  ID:       {doc_id}")
        print(f"  Metadata: {metadata}")
        print(f"  Content:  {content[:150]}{'...' if len(content) > 150 else ''}")

    print(f"\n{'='*60}")


def search(query):
    """Search documents by semantic similarity."""
    client = chromadb.PersistentClient(path=PERSIST_DIR)

    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception:
        print("Collection not found.")
        return

    print(f"\nSearching for: \"{query}\"")
    print(f"{'='*60}")

    results = collection.query(
        query_texts=[query],
        n_results=min(5, collection.count()),
        include=["documents", "metadatas", "distances"]
    )

    if not results["documents"][0]:
        print("No results found.")
        return

    for i in range(len(results["documents"][0])):
        distance = results["distances"][0][i]
        score = round(max(0.0, 1.0 - (distance / 2.0)), 4)
        content = results["documents"][0][i]
        metadata = results["metadatas"][0][i]

        print(f"\n--- Result {i+1} (relevance: {score}) ---")
        print(f"  Metadata: {metadata}")
        print(f"  Content:  {content[:200]}{'...' if len(content) > 200 else ''}")

    print(f"\n{'='*60}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        search(" ".join(sys.argv[1:]))
    else:
        inspect()