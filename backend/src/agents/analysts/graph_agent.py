"""Graph analyst agent wrapper."""

from __future__ import annotations

from ...knowledge.repository import KnowledgeRepository
from ...llm.client import LLMClient, LLMRunnable
from ...tools.analyst.tooling import AnalystToolRegistry
from ...services.analysts.graph_service import GraphAnalystService
from .base_agent import BaseLangGraphAnalystAgent, PromptProvider


class GraphAnalystAgent(BaseLangGraphAnalystAgent):
    """Agent wrapper around the graph analyst knowledge service."""

    def __init__(
        self,
        *,
        repository: KnowledgeRepository | None = None,
        service: GraphAnalystService | None = None,
        tool_registry: AnalystToolRegistry | None = None,
        prompt_provider: PromptProvider | None = None,
        llm_client: LLMClient | None = None,
        llm: LLMRunnable | None = None,
    ) -> None:
        graph_service = service or GraphAnalystService(repository=repository)
        super().__init__(
            analyst_name=graph_service.analyst_name,
            knowledge_service=graph_service,
            tool_registry=tool_registry,
            prompt_provider=prompt_provider,
            llm_client=llm_client,
            llm=llm,
        )
