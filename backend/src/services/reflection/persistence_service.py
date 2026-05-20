"""Persistence helpers for reflection-generated postmortem memories."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from ...knowledge.ingest import KnowledgeIngestor
from ...knowledge.repository import DatasetName, KnowledgeRepository
from ..decision.memory import validate_decision_memory_record


class ReflectionPersistenceService:
    """Persist reflection-generated candidate memories into the knowledge base."""

    def __init__(
        self,
        repository: KnowledgeRepository | None = None,
        ingestor: KnowledgeIngestor | None = None,
    ) -> None:
        self.repository = repository or KnowledgeRepository()
        self.ingestor = ingestor or KnowledgeIngestor(self.repository)

    def persist_reflection_result(
        self,
        reflection_result: dict[str, Any] | None,
        *,
        dataset: DatasetName = "dynamic",
        force: bool = False,
        record_name: str | None = None,
    ) -> dict[str, Any]:
        """Persist a reflection candidate memory when the result is eligible."""
        if not isinstance(reflection_result, dict):
            return {
                "status": "skipped",
                "persisted": False,
                "reason": "reflection result must be a JSON object",
            }

        persistence = dict(reflection_result.get("memory_persistence", {}))
        should_persist = bool(persistence.get("should_persist"))
        if not force and not should_persist:
            blocking_issues = list(persistence.get("blocking_issues", []))
            reason = blocking_issues[0] if blocking_issues else "memory_persistence did not allow persistence"
            return {
                "status": "skipped",
                "persisted": False,
                "reason": reason,
                "memory_persistence": persistence,
            }

        candidate_memory = reflection_result.get("candidate_memory")
        if not isinstance(candidate_memory, dict):
            return {
                "status": "skipped",
                "persisted": False,
                "reason": "reflection result does not contain a candidate_memory payload",
                "memory_persistence": persistence,
            }

        prepared_record = self._prepare_record(candidate_memory, dataset=dataset)
        validation = validate_decision_memory_record(prepared_record, allowed_datasets=(dataset,))
        if not validation["is_valid"]:
            return {
                "status": "skipped",
                "persisted": False,
                "reason": "candidate_memory failed decision-memory validation",
                "errors": validation["errors"],
                "memory_persistence": persistence,
            }

        target_name = record_name or self._build_record_name(reflection_result, prepared_record)
        self._ensure_repository_ready()
        record_path = self.ingestor.ingest_text(
            dataset,
            target_name,
            prepared_record["text"],
            metadata=prepared_record["metadata"],
        )
        return {
            "status": "persisted",
            "persisted": True,
            "path": str(record_path),
            "record_name": record_path.stem,
            "title": prepared_record["metadata"].get("title", record_path.stem),
            "memory_persistence": persistence,
        }

    def _prepare_record(
        self,
        candidate_memory: dict[str, Any],
        *,
        dataset: DatasetName,
    ) -> dict[str, Any]:
        """Normalize the candidate memory into a writable processed-record shape."""
        metadata = dict(candidate_memory.get("metadata", {}))
        metadata["dataset"] = dataset
        return {
            "text": str(candidate_memory.get("text", "")).strip(),
            "metadata": metadata,
        }

    def _ensure_repository_ready(self) -> None:
        """Create the repository layout and initialize the manifest if needed."""
        self.repository.ensure_structure()
        if self.repository.manifest_exists():
            return
        self.repository.save_manifest(
            {
                "version": "0.1.0",
                "description": "Placeholder manifest for the project knowledge base.",
                "datasets": {
                    "foundation": {"raw": [], "processed": []},
                    "dynamic": {"raw": [], "processed": []},
                },
                "indexes": [],
            }
        )

    def _build_record_name(
        self,
        reflection_result: dict[str, Any],
        prepared_record: dict[str, Any],
    ) -> str:
        """Build a deterministic record name for persisted reflection memories."""
        metadata = dict(prepared_record.get("metadata", {}))
        symbol = str(reflection_result.get("symbol") or metadata.get("symbol") or "").strip().lower()
        trade_date = str(reflection_result.get("trade_date") or metadata.get("created_at") or "").strip()
        date_part = trade_date.split("T", 1)[0].replace("-", "_")
        title = str(metadata.get("title", "")).strip()

        name_parts = ["reflection_postmortem"]
        if symbol:
            name_parts.append(symbol)
        if date_part:
            name_parts.append(date_part)
        if title:
            name_parts.append(title)

        raw_name = "_".join(name_parts)
        normalized = re.sub(r"[^a-z0-9]+", "_", raw_name.lower()).strip("_")
        return normalized or "reflection_postmortem"
