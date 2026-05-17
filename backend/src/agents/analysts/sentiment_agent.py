"""Sentiment analyst agent wrapper."""

from __future__ import annotations

from ...knowledge.repository import KnowledgeRepository
from ...llm.client import LLMClient, LLMRunnable
from ...tools.analyst.tooling import AnalystToolRegistry
from ...services.analysts.sentiment_service import SentimentAnalystService
from .base_agent import BaseLangGraphAnalystAgent, PromptProvider


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
