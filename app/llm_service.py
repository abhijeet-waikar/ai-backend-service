"""
llm_service.py - OpenAI LLM integration with production-grade patterns.

Key patterns demonstrated:
1. Retry with exponential backoff (handles rate limits and transient errors)
2. Streaming responses (for real-time UI updates and lower latency perception)
3. Structured JSON output (reliable parsing of LLM responses)
4. Token usage tracking (cost monitoring)
5. Error classification (distinguishes retryable vs permanent failures)
"""

import json
import logging
from typing import AsyncGenerator

from openai import OpenAI, APIError, RateLimitError, APIConnectionError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────

MODEL = "gpt-4o-mini"  # Cost-effective model, good for most tasks
MAX_TOKENS = 1000
TEMPERATURE = 0.3  # Lower = more deterministic, better for structured output


class LLMService:
    """
    Handles all interactions with OpenAI's API.

    Design decisions:
    - Uses sync client (simpler, sufficient for most backend services)
    - Retry logic handles transient failures automatically
    - Structured output ensures reliable JSON parsing
    - Token tracking enables cost monitoring per request
    """

    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.model = MODEL
        logger.info(f"LLM Service initialized with model: {self.model}")

    def is_available(self) -> bool:
        """Quick health check - verifies API key is valid."""
        try:
            self.client.models.retrieve(self.model)
            return True
        except Exception as e:
            logger.warning(f"LLM service health check failed: {e}")
            return False

    # ── Retry decorator ────────────────────────────────────────
    # Retries up to 3 times with exponential backoff (1s, 2s, 4s)
    # Only retries on transient errors, NOT on bad requests (4xx)

    @retry(
        retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def analyze_text(self, text: str, instruction: str) -> dict:
        """
        Analyze text using the LLM and return structured JSON output.

        This demonstrates:
        - Structured output via system prompt engineering
        - Retry logic for transient API failures
        - Token usage tracking for cost monitoring
        - Error handling with meaningful error messages

        Args:
            text: The text content to analyze
            instruction: What the LLM should do with the text

        Returns:
            dict with keys: summary, key_points, sentiment, model_used, tokens_used
        """
        system_prompt = """You are a precise text analysis assistant. 
Respond ONLY with valid JSON in this exact format, no other text:
{
    "summary": "A concise summary of the analysis",
    "key_points": ["point 1", "point 2", "point 3"],
    "sentiment": "positive OR negative OR neutral"
}"""

        user_prompt = f"""Instruction: {instruction}

Text to analyze:
{text}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                response_format={"type": "json_object"},
            )

            # Parse the structured response
            content = response.choices[0].message.content
            result = json.loads(content)

            # Add metadata
            result["model_used"] = self.model
            result["tokens_used"] = response.usage.total_tokens if response.usage else 0

            # Ensure required fields exist with defaults
            result.setdefault("summary", "No summary generated")
            result.setdefault("key_points", [])
            result.setdefault("sentiment", "neutral")

            logger.info(
                f"Analysis complete. Tokens used: {result['tokens_used']}"
            )
            return result

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM JSON response: {e}")
            return {
                "summary": content if 'content' in dir() else "Parse error",
                "key_points": [],
                "sentiment": "neutral",
                "model_used": self.model,
                "tokens_used": 0,
            }
        except APIError as e:
            logger.error(f"OpenAI API error: {e.status_code} - {e.message}")
            raise

    # ── Streaming ──────────────────────────────────────────────

    def stream_response(self, prompt: str) -> str:
        """
        Stream a response from the LLM, collecting chunks into full text.

        Streaming is used when:
        - You want to show real-time output to users
        - You want faster time-to-first-token
        - You're building chat interfaces

        This method collects the stream into a complete string.
        In a real application, you'd yield chunks via Server-Sent Events.

        Args:
            prompt: The user's prompt

        Returns:
            Complete response text assembled from stream chunks
        """
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                stream=True,
            )

            full_response = []
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    full_response.append(chunk.choices[0].delta.content)

            return "".join(full_response)

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            raise

    # ── RAG Answer Generation ──────────────────────────────────

    @retry(
        retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    def generate_rag_answer(self, question: str, context_docs: list[str]) -> dict:
        """
        Generate an answer grounded in retrieved documents (RAG pattern).

        This is the 'Generation' step of Retrieval-Augmented Generation:
        1. Vector search retrieves relevant documents (done in vector_service)
        2. This method takes those documents as context
        3. The LLM generates an answer based ONLY on the provided context
        4. This reduces hallucination because the model has real data to reference

        Args:
            question: The user's question
            context_docs: List of document texts retrieved from vector search

        Returns:
            dict with answer, model_used
        """
        context = "\n\n---\n\n".join(
            [f"Document {i+1}:\n{doc}" for i, doc in enumerate(context_docs)]
        )

        system_prompt = """You are a helpful assistant that answers questions based ONLY on the provided context documents. 
If the answer cannot be found in the context, say "I don't have enough information in the provided documents to answer this question."
Always cite which document(s) you used."""

        user_prompt = f"""Context Documents:
{context}

Question: {question}

Answer based only on the above documents:"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )

        return {
            "answer": response.choices[0].message.content,
            "model_used": self.model,
        }
