"""Base analyst service primitives backed by the knowledge layer."""

from __future__ import annotations

from typing import Any

from ...knowledge.indexing import KnowledgeIndexer
from ...knowledge.repository import DatasetName, KnowledgeRepository
from ...knowledge.retriever import KnowledgeRetriever, VectorRetrieverBackend


class KnowledgeBackedAnalystService:
    """Shared service base for analysts that depend on the knowledge layer.

    The class intentionally stays retrieval-first. LangGraph agents can build on
    top of it without having to duplicate query construction, metadata filters,
    or knowledge-base serialization.
    """

    analyst_name = "knowledge_backed_analyst"
    default_datasets: tuple[DatasetName, ...] = ("foundation", "dynamic")
    default_k = 4

    def __init__(
        self,
        repository: KnowledgeRepository | None = None,
        retriever: KnowledgeRetriever | None = None,
        backend: VectorRetrieverBackend | None = None,
    ) -> None:
        self.repository = repository or KnowledgeRepository()
        self.indexer = KnowledgeIndexer(self.repository)
        resolved_backend = backend or self.indexer.build_local_index(self.default_datasets)
        self.retriever = retriever or KnowledgeRetriever(self.repository, backend=resolved_backend)

    def default_metadata_filter(self) -> dict[str, Any]:
        """Return the analyst-specific metadata filter."""
        return {}

    def build_query(self, subject: str, extra_context: str | None = None) -> str:
        """Build a retrieval query from the current task input."""
        parts = [subject.strip()]
        if extra_context:
            parts.append(extra_context.strip())
        return " ".join(part for part in parts if part)

    def build_metadata_filter(
        self,
        *,
        symbol: str | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Merge default and call-specific metadata filters."""
        merged_filter = dict(self.default_metadata_filter())
        if symbol:
            merged_filter["symbol"] = symbol
        if metadata_filter:
            merged_filter.update(metadata_filter)
        return merged_filter

    def retrieve_context(
        self,
        query: str,
        *,
        datasets: tuple[DatasetName, ...] | None = None,
        symbol: str | None = None,
        metadata_filter: dict[str, Any] | None = None,
        k: int | None = None,
    ) -> list[Any]:
        """Fetch knowledge documents relevant to the current analysis task."""
        selected_datasets = datasets or self.default_datasets
        merged_filter = self.build_metadata_filter(
            symbol=symbol,
            metadata_filter=metadata_filter,
        )
        return self.retriever.search(
            query,
            datasets=selected_datasets,
            k=k or self.default_k,
            metadata_filter=merged_filter or None,
        )

    def analyze(
        self,
        subject: str,
        *,
        extra_context: str | None = None,
        datasets: tuple[DatasetName, ...] | None = None,
        symbol: str | None = None,
        metadata_filter: dict[str, Any] | None = None,
        k: int | None = None,
    ) -> dict[str, Any]:
        """Return a structured analysis payload with retrieved context."""
        selected_datasets = datasets or self.default_datasets
        query = self.build_query(subject, extra_context=extra_context)
        documents = self.retrieve_context(
            query,
            datasets=selected_datasets,
            symbol=symbol,
            metadata_filter=metadata_filter,
            k=k,
        )
        return self.build_analysis_context(
            subject,
            query=query,
            datasets=selected_datasets,
            documents=documents,
            symbol=symbol,
            extra_context=extra_context,
        )

    def build_analysis_context(
        self,
        subject: str,
        *,
        query: str,
        datasets: tuple[DatasetName, ...],
        documents: list[Any],
        symbol: str | None = None,
        extra_context: str | None = None,
    ) -> dict[str, Any]:
        """Build an agent-friendly context payload from retrieved documents."""
        serialized_documents = [self.serialize_document(document) for document in documents]
        evidence = self.collect_evidence(serialized_documents)
        return {
            "analyst": self.analyst_name,
            "subject": subject,
            "query": query,
            "symbol": symbol,
            "extra_context": extra_context,
            "datasets": list(datasets),
            "document_count": len(serialized_documents),
            "documents": serialized_documents,
            "evidence": evidence,
        }

    def serialize_document(self, document: Any) -> dict[str, Any]:
        """Convert a retrieved document into a service-friendly payload."""
        metadata = dict(getattr(document, "metadata", {}))
        return {
            "title": metadata.get("title", ""),
            "text": getattr(document, "page_content", ""),
            "metadata": metadata,
        }

    def collect_evidence(self, documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Normalize serialized documents into evidence entries for agents."""
        evidence: list[dict[str, Any]] = []
        for document in documents:
            metadata = dict(document.get("metadata", {}))
            evidence.append(
                {
                    "source_type": "knowledge_base",
                    "title": document.get("title", ""),
                    "content": self.build_excerpt(document.get("text", "")),
                    "metadata": metadata,
                }
            )
        return evidence

    def build_prompt_context(
        self,
        subject: str,
        *,
        extra_context: str | None = None,
        datasets: tuple[DatasetName, ...] | None = None,
        symbol: str | None = None,
        metadata_filter: dict[str, Any] | None = None,
        k: int | None = None,
    ) -> dict[str, Any]:
        """Return the knowledge-backed context used to render an analyst prompt."""
        return self.analyze(
            subject,
            extra_context=extra_context,
            datasets=datasets,
            symbol=symbol,
            metadata_filter=metadata_filter,
            k=k,
        )

    def build_excerpt(self, text: str, *, limit: int = 280) -> str:
        """Return a compact evidence excerpt suitable for prompts."""
        compact_text = " ".join(text.split())
        if len(compact_text) <= limit:
            return compact_text
        return compact_text[: limit - 3].rstrip() + "..."
