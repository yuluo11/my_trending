"""Concrete advisory decision agent."""

from __future__ import annotations

from ...llm.client import LLMClient, LLMRunnable
from ...services.decision.memory import DecisionKnowledgeService
from .base_agent import BaseDecisionAgent, PromptProvider


class DecisionAdvisoryAgent(BaseDecisionAgent):
    """Decision advisory agent backed by decision-memory retrieval."""

    def __init__(
        self,
        *,
        service: DecisionKnowledgeService,
        prompt_provider: PromptProvider | None = None,
        llm_client: LLMClient | None = None,
        llm: LLMRunnable | None = None,
    ) -> None:
        super().__init__(
            agent_name=service.agent_name,
            knowledge_service=service,
            prompt_provider=prompt_provider,
            llm_client=llm_client,
            llm=llm,
        )
