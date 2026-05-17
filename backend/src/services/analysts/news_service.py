"""News analyst service backed by the shared knowledge layer."""

from typing import Any

from .graph_service import KnowledgeBackedAnalystService


class NewsAnalystService(KnowledgeBackedAnalystService):
    """Retrieve time-sensitive news context from the dynamic knowledge base."""

    analyst_name = "news_analyst"
    default_datasets = ("dynamic",)

    def default_metadata_filter(self) -> dict[str, Any]:
        return {"category": "news"}
