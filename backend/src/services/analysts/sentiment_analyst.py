"""Sentiment analyst service backed by the shared knowledge layer."""

from typing import Any

from ...knowledge.repository import KnowledgeRepository
from ...llm.client import LLMClient, LLMRunnable
from ...tools.analyst.tooling import AnalystToolRegistry
from .base_agent import BaseLangGraphAnalystAgent, PromptProvider
from .graph_analyst import KnowledgeBackedAnalystService


class SentimentAnalystService(KnowledgeBackedAnalystService):
    """Retrieve sentiment-oriented context from dynamic and foundational knowledge."""

    analyst_name = "sentiment_analyst"
    default_datasets = ("dynamic", "foundation")

    def build_query(self, subject: str, extra_context: str | None = None) -> str:
        base_query = super().build_query(subject, extra_context=extra_context)
        return f"{base_query} sentiment market reaction".strip()

    def default_metadata_filter(self) -> dict[str, Any]:
        return {"time_sensitivity": "high"}


class SentimentAnalystAgent(BaseLangGraphAnalystAgent):
    """Agent wrapper around the sentiment analyst knowledge service."""

    def __init__(
        self,
        *,
        repository: KnowledgeRepository | None = None,
        service: SentimentAnalystService | None = None,
        tool_registry: AnalystToolRegistry | None = None,
        prompt_provider: PromptProvider | None = None,
        llm_client: LLMClient | None = None,
        llm: LLMRunnable | None = None,
    ) -> None:
        sentiment_service = service or SentimentAnalystService(repository=repository)
        super().__init__(
            analyst_name=sentiment_service.analyst_name,
            knowledge_service=sentiment_service,
            tool_registry=tool_registry,
            prompt_provider=prompt_provider,
            llm_client=llm_client,
            llm=llm,
        )
