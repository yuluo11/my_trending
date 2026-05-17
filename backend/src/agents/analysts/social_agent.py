"""Social analyst agent wrapper."""

from __future__ import annotations

from ...knowledge.repository import KnowledgeRepository
from ...llm.client import LLMClient, LLMRunnable
from ...tools.analyst.tooling import AnalystToolRegistry
from ...services.analysts.social_service import SocialAnalystService
from .base_agent import BaseLangGraphAnalystAgent, PromptProvider


class SocialAnalystAgent(BaseLangGraphAnalystAgent):
    """Agent wrapper around the social analyst knowledge service."""

    def __init__(
        self,
        *,
        repository: KnowledgeRepository | None = None,
        service: SocialAnalystService | None = None,
        tool_registry: AnalystToolRegistry | None = None,
        prompt_provider: PromptProvider | None = None,
        llm_client: LLMClient | None = None,
        llm: LLMRunnable | None = None,
    ) -> None:
        social_service = service or SocialAnalystService(repository=repository)
        super().__init__(
            analyst_name=social_service.analyst_name,
            knowledge_service=social_service,
            tool_registry=tool_registry,
            prompt_provider=prompt_provider,
            llm_client=llm_client,
            llm=llm,
        )
