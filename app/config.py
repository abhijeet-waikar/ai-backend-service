"""
config.py - Centralized configuration management.

All settings in one place. Override any value via environment variables.
This pattern lets the same codebase run in dev, staging, and production
without code changes - just different .env files.

Usage:
    from app.config import settings
    print(settings.OPENAI_MODEL)
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Application settings loaded from environment with sensible defaults."""

    # ── OpenAI ─────────────────────────────────────────────
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    OPENAI_MAX_TOKENS: int = int(os.getenv("OPENAI_MAX_TOKENS", "1000"))
    OPENAI_TEMPERATURE: float = float(os.getenv("OPENAI_TEMPERATURE", "0.3"))

    # ── Vector DB ──────────────────────────────────────────
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_data")
    CHROMA_COLLECTION: str = os.getenv("CHROMA_COLLECTION", "documents")

    # ── API ────────────────────────────────────────────────
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    API_VERSION: str = "1.0.1"

    # ── Retry ──────────────────────────────────────────────
    RETRY_MAX_ATTEMPTS: int = int(os.getenv("RETRY_MAX_ATTEMPTS", "3"))
    RETRY_MIN_WAIT: int = int(os.getenv("RETRY_MIN_WAIT", "1"))
    RETRY_MAX_WAIT: int = int(os.getenv("RETRY_MAX_WAIT", "8"))


settings = Settings()
