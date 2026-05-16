"""Social analyst service backed by the shared knowledge layer."""

from typing import Any

from ...knowledge.repository import KnowledgeRepository
from ...llm.client import LLMClient, LLMRunnable
from ...tools.analyst.tooling import AnalystToolRegistry
from .base_agent import BaseLangGraphAnalystAgent, PromptProvider
from .graph_analyst import KnowledgeBackedAnalystService


class SocialAnalystService(KnowledgeBackedAnalystService):
    """Retrieve social and crowd-context knowledge for narrative monitoring."""

    analyst_name = "social_analyst"
    default_datasets = ("dynamic",)

    def build_query(self, subject: str, extra_context: str | None = None) -> str:
        base_query = super().build_query(subject, extra_context=extra_context)
        return f"{base_query} social narrative discussion".strip()

    def default_metadata_filter(self) -> dict[str, Any]:
        return {"dataset": "dynamic"}


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
