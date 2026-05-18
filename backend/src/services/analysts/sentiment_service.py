"""Sentiment analyst service backed by the shared knowledge layer."""

from typing import Any

from .base_service import KnowledgeBackedAnalystService


class SentimentAnalystService(KnowledgeBackedAnalystService):
    """Retrieve sentiment-oriented context from dynamic and foundational knowledge."""

    analyst_name = "sentiment_analyst"
    default_datasets = ("dynamic", "foundation")

    def build_query(self, subject: str, extra_context: str | None = None) -> str:
        base_query = super().build_query(subject, extra_context=extra_context)
        return f"{base_query} sentiment market reaction".strip()

    def default_metadata_filter(self) -> dict[str, Any]:
        return {"time_sensitivity": "high"}
