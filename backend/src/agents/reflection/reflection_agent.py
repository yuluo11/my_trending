"""Concrete post-decision reflection agent."""

from __future__ import annotations

from ...llm.client import LLMClient, LLMRunnable
from ...services.reflection import ReflectionContextService
from .base_agent import BaseReflectionAgent, PromptProvider


class ReflectionAgent(BaseReflectionAgent):
    """Reflection agent backed by post-decision context retrieval."""

    def __init__(
        self,
        *,
        service: ReflectionContextService,
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
