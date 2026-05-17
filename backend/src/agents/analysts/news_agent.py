"""News analyst agent wrapper."""

from __future__ import annotations

from ...knowledge.repository import KnowledgeRepository
from ...llm.client import LLMClient, LLMRunnable
from ...tools.analyst.tooling import AnalystToolRegistry
from ...services.analysts.news_service import NewsAnalystService
from .base_agent import BaseLangGraphAnalystAgent, PromptProvider


class NewsAnalystAgent(BaseLangGraphAnalystAgent):
    """Agent wrapper around the news analyst knowledge service."""

    def __init__(
        self,
        *,
        repository: KnowledgeRepository | None = None,
        service: NewsAnalystService | None = None,
        tool_registry: AnalystToolRegistry | None = None,
        prompt_provider: PromptProvider | None = None,
        llm_client: LLMClient | None = None,
        llm: LLMRunnable | None = None,
    ) -> None:
        news_service = service or NewsAnalystService(repository=repository)
        super().__init__(
            analyst_name=news_service.analyst_name,
            knowledge_service=news_service,
            tool_registry=tool_registry,
            prompt_provider=prompt_provider,
            llm_client=llm_client,
            llm=llm,
        )
