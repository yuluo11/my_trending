"""Tooling primitives for analyst agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from ...services.analysts.graph_service import KnowledgeBackedAnalystService


@dataclass(slots=True)
class ToolCallRequest:
    """Standard request envelope for analyst tool invocations."""

    tool_name: str
    subject: str
    symbol: str | None = None
    trade_date: str | None = None
    extra_context: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ToolCallResult:
    """Standardized result returned by analyst tools."""

    tool_name: str
    success: bool
    content: str
    structured_data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class AnalystTool(Protocol):
    """Minimal protocol that all analyst tools should satisfy."""

    name: str

    def invoke(self, request: ToolCallRequest) -> ToolCallResult:
        """Execute the tool for the provided analyst request."""


class AnalystToolRegistry:
    """Register and invoke tools that analyst agents may call."""

    def __init__(self) -> None:
        self._tools: dict[str, AnalystTool] = {}

    def register(self, tool: AnalystTool) -> None:
        """Register a tool by its public name."""
        self._tools[tool.name] = tool

    def has(self, tool_name: str) -> bool:
        """Return whether a tool is present in the registry."""
        return tool_name in self._tools

    def get(self, tool_name: str) -> AnalystTool:
        """Return a registered tool or raise a KeyError."""
        try:
            return self._tools[tool_name]
        except KeyError as error:
            raise KeyError(f"Unknown analyst tool: {tool_name}") from error

    def list_names(self) -> list[str]:
        """Return the registered tool names in stable order."""
        return sorted(self._tools)

    def invoke(self, request: ToolCallRequest) -> ToolCallResult:
        """Invoke a tool through the registry."""
        return self.get(request.tool_name).invoke(request)


class KnowledgeBaseSearchTool:
    """Tool adapter exposing the knowledge base to LangGraph analyst agents."""

    name = "search_knowledge_base"

    def __init__(self, service: KnowledgeBackedAnalystService) -> None:
        self.service = service

    def invoke(self, request: ToolCallRequest) -> ToolCallResult:
        """Search the knowledge base and return serialized analyst context."""
        try:
            datasets = request.arguments.get("datasets")
            metadata_filter = request.arguments.get("metadata_filter")
            max_documents = request.arguments.get("max_documents")
            payload = self.service.analyze(
                request.subject,
                extra_context=request.extra_context,
                datasets=tuple(datasets) if datasets else None,
                symbol=request.symbol,
                metadata_filter=metadata_filter,
                k=max_documents,
            )
        except Exception as error:  # pragma: no cover - defensive tool boundary
            return ToolCallResult(
                tool_name=self.name,
                success=False,
                content="Knowledge-base search failed.",
                error=str(error),
            )

        titles = [
            document.get("title", "")
            for document in payload.get("documents", [])
            if document.get("title")
        ]
        content = (
            f"Retrieved {payload.get('document_count', 0)} knowledge documents"
            + (f": {', '.join(titles[:3])}" if titles else ".")
        )
        return ToolCallResult(
            tool_name=self.name,
            success=True,
            content=content,
            structured_data=payload,
        )
