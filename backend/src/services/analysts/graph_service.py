"""Graph analyst service backed by the shared knowledge layer."""

from __future__ import annotations

from .base_service import KnowledgeBackedAnalystService


class GraphAnalystService(KnowledgeBackedAnalystService):
    """Retrieve cross-cutting context used for relationship and workflow analysis."""

    analyst_name = "graph_analyst"
    default_datasets = ("foundation", "dynamic")

    def build_query(self, subject: str, extra_context: str | None = None) -> str:
        base_query = super().build_query(subject, extra_context=extra_context)
        return f"{base_query} relationships dependencies workflow context".strip()
