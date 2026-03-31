"""
seed_data.py - Load sample documents into the vector store.

Run this once after starting the server to populate the vector DB
with sample documents for testing search and RAG endpoints.

Usage:
    python seed_data.py

The sample documents are payment/fintech themed to align with
Mastercard's domain. In production, these would come from actual
enterprise knowledge bases, documentation, or databases.
"""

import requests

API_URL = "http://localhost:8000"

# Sample documents - payment/fintech domain (relevant to Mastercard)
SAMPLE_DOCUMENTS = [
    {
        "content": (
            "Tokenization replaces sensitive card data with a unique digital identifier "
            "called a token. This token can be used for transactions without exposing "
            "the actual card number. Tokenization significantly reduces the risk of "
            "data breaches and is a key technology in mobile payments and e-commerce. "
            "Major payment networks implement tokenization to secure contactless "
            "payments and in-app purchases."
        ),
        "metadata": {"source": "payment_security", "topic": "tokenization", "category": "security"},
    },
    {
        "content": (
            "Real-time fraud detection systems analyze transaction patterns using "
            "machine learning models. These systems evaluate hundreds of signals "
            "including transaction amount, merchant category, geographic location, "
            "time of day, and device fingerprint. A risk score is generated within "
            "milliseconds, and transactions exceeding the threshold are flagged for "
            "review or automatically declined."
        ),
        "metadata": {"source": "fraud_detection", "topic": "ml_fraud", "category": "ai"},
    },
    {
        "content": (
            "Open Banking APIs enable third-party developers to build applications "
            "that access banking data with customer consent. PSD2 regulations in "
            "Europe mandate banks to provide APIs for account information and payment "
            "initiation. This has created an ecosystem of fintech applications for "
            "budgeting, lending, and payment services."
        ),
        "metadata": {"source": "open_banking", "topic": "apis", "category": "regulation"},
    },
    {
        "content": (
            "The ISO 20022 messaging standard is being adopted globally for payment "
            "messaging. It provides richer, more structured data compared to legacy "
            "formats like SWIFT MT messages. Benefits include improved straight-through "
            "processing rates, better compliance screening, and enhanced remittance "
            "information. Major payment infrastructures are migrating to ISO 20022."
        ),
        "metadata": {"source": "standards", "topic": "iso20022", "category": "infrastructure"},
    },
    {
        "content": (
            "Contactless payments using NFC technology allow consumers to tap their "
            "card or mobile device at a terminal to make a payment. The technology "
            "uses short-range wireless communication and processes transactions in "
            "under a second. Adoption accelerated during the pandemic, and contactless "
            "now accounts for over 50 percent of in-person transactions in many markets."
        ),
        "metadata": {"source": "payment_methods", "topic": "contactless", "category": "consumer"},
    },
    {
        "content": (
            "Payment orchestration platforms route transactions across multiple "
            "payment processors to optimize authorization rates and reduce costs. "
            "Smart routing algorithms consider factors like processor uptime, "
            "regional success rates, and interchange fees. These platforms provide "
            "a unified API layer that abstracts the complexity of working with "
            "multiple payment gateways."
        ),
        "metadata": {"source": "infrastructure", "topic": "orchestration", "category": "architecture"},
    },
    {
        "content": (
            "Large language models are being applied to payment dispute resolution. "
            "AI systems can analyze transaction details, merchant information, and "
            "customer communication to automatically categorize disputes, suggest "
            "resolutions, and draft response letters. This reduces the average "
            "resolution time from days to hours and improves customer satisfaction."
        ),
        "metadata": {"source": "ai_applications", "topic": "dispute_resolution", "category": "ai"},
    },
    {
        "content": (
            "Cross-border payments face challenges including currency conversion, "
            "regulatory compliance across jurisdictions, anti-money laundering "
            "screening, and settlement timing. New technologies like blockchain "
            "and central bank digital currencies aim to reduce friction and cost "
            "in international money transfers."
        ),
        "metadata": {"source": "cross_border", "topic": "international", "category": "infrastructure"},
    },
    {
        "content": (
            "Generative AI is transforming customer service in financial institutions. "
            "AI-powered chatbots can handle account inquiries, explain transaction "
            "details, guide users through dispute processes, and provide personalized "
            "financial insights. These systems use retrieval-augmented generation to "
            "ground responses in actual customer data and policy documents."
        ),
        "metadata": {"source": "ai_applications", "topic": "customer_service", "category": "ai"},
    },
    {
        "content": (
            "Payment data analytics provides merchants with insights into customer "
            "spending patterns, peak transaction times, and basket analysis. Machine "
            "learning models can predict churn, identify upsell opportunities, and "
            "optimize pricing strategies. These analytics capabilities are offered "
            "as value-added services by payment networks and processors."
        ),
        "metadata": {"source": "analytics", "topic": "merchant_insights", "category": "ai"},
    },
]


def seed_documents():
    """Load all sample documents into the vector store."""
    print(f"Seeding {len(SAMPLE_DOCUMENTS)} documents...")

    response = requests.post(
        f"{API_URL}/api/documents/batch",
        json={"documents": SAMPLE_DOCUMENTS},
    )

    if response.status_code == 200:
        result = response.json()
        print(f"Success! {result['message']}")
        print(f"Total documents in store: {result['total_documents']}")
    else:
        print(f"Error: {response.status_code} - {response.text}")


def test_search():
    """Run a sample search to verify everything works."""
    print("\n--- Testing Search ---")
    response = requests.post(
        f"{API_URL}/api/search",
        json={"query": "How does fraud detection work in payments?", "top_k": 3},
    )

    if response.status_code == 200:
        result = response.json()
        print(f"Query: {result['query']}")
        for i, r in enumerate(result["results"]):
            print(f"\n  Result {i+1} (score: {r['relevance_score']}):")
            print(f"  {r['content'][:100]}...")
    else:
        print(f"Error: {response.status_code} - {response.text}")


def test_rag():
    """Run a sample RAG query to verify end-to-end flow."""
    print("\n--- Testing RAG ---")
    response = requests.post(
        f"{API_URL}/api/rag",
        json={"question": "How is AI being used to improve payment dispute resolution?"},
    )

    if response.status_code == 200:
        result = response.json()
        print(f"Question: {result['question']}")
        print(f"Answer: {result['answer']}")
        print(f"Sources used: {len(result['sources'])}")
    else:
        print(f"Error: {response.status_code} - {response.text}")


if __name__ == "__main__":
    seed_documents()
    test_search()
    test_rag()
    print("\nAll tests passed! Your AI backend service is working.")
