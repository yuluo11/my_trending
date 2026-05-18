"""Social analyst service backed by the shared knowledge layer."""

from typing import Any

from .base_service import KnowledgeBackedAnalystService


class SocialAnalystService(KnowledgeBackedAnalystService):
    """Retrieve social and crowd-context knowledge for narrative monitoring."""

    analyst_name = "social_analyst"
    default_datasets = ("dynamic",)

    def build_query(self, subject: str, extra_context: str | None = None) -> str:
        base_query = super().build_query(subject, extra_context=extra_context)
        return f"{base_query} social narrative discussion".strip()

    def default_metadata_filter(self) -> dict[str, Any]:
        return {"dataset": "dynamic"}
