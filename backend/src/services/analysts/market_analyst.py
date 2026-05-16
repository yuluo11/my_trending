"""Market analyst service backed by the shared knowledge layer."""

from ...knowledge.repository import KnowledgeRepository
from ...llm.client import LLMClient, LLMRunnable
from ...tools.analyst.tooling import AnalystToolRegistry
from .base_agent import BaseLangGraphAnalystAgent, PromptProvider
from .graph_analyst import KnowledgeBackedAnalystService


class MarketAnalystService(KnowledgeBackedAnalystService):
    """Retrieve strategy and market context for broader market analysis."""

    analyst_name = "market_analyst"
    default_datasets = ("foundation", "dynamic")


class MarketAnalystAgent(BaseLangGraphAnalystAgent):
    """Agent wrapper around the market analyst knowledge service."""

    def __init__(
        self,
        *,
        repository: KnowledgeRepository | None = None,
        service: MarketAnalystService | None = None,
        tool_registry: AnalystToolRegistry | None = None,
        prompt_provider: PromptProvider | None = None,
        llm_client: LLMClient | None = None,
        llm: LLMRunnable | None = None,
    ) -> None:
        market_service = service or MarketAnalystService(repository=repository)
        super().__init__(
            analyst_name=market_service.analyst_name,
            knowledge_service=market_service,
            tool_registry=tool_registry,
            prompt_provider=prompt_provider,
            llm_client=llm_client,
            llm=llm,
        )
