"""Reflection-context service for post-decision review and postmortem prep."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ...knowledge.indexing import KnowledgeIndexer
from ...knowledge.repository import DatasetName, KnowledgeRepository
from ...knowledge.retriever import KnowledgeRetriever, VectorRetrieverBackend
from ..decision.memory import DecisionKnowledgeService
from .schema import (
    assess_post_trade_review_completeness,
    build_candidate_postmortem_record,
    infer_outcome_label,
    validate_post_trade_review,
)

if TYPE_CHECKING:
    from ...agents.reflection.base_agent import ReflectionTask


class ReflectionContextService:
    """Prepare post-decision reflection context and retrieve historical cases."""

    agent_name = "reflection_agent"
    default_datasets: tuple[DatasetName, ...] = ("dynamic",)
    default_k = 3

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
        self.decision_service = DecisionKnowledgeService(
            repository=self.repository,
            retriever=self.retriever,
            backend=resolved_backend,
        )

    def default_metadata_filter(self) -> dict[str, Any]:
        """Restrict reflection retrieval to decision-memory records by default."""
        return {"category": "decision_memory"}

    def build_query(self, task: "ReflectionTask") -> str:
        """Build a retrieval query from decision outputs and post-trade review data."""
        decision_output = task.decision_output or {}
        query_parts: list[str] = [task.subject.strip()]
        if task.symbol:
            query_parts.append(task.symbol.strip())

        decision_summary = str(decision_output.get("decision_summary", "")).strip()
        if decision_summary:
            query_parts.append(decision_summary)

        recommendation = str(decision_output.get("recommendation", "")).strip()
        if recommendation:
            query_parts.append(f"recommendation {recommendation}")

        outcome_summary = self.summarize_realized_outcome(
            task.realized_outcome,
            outcome_metrics=task.outcome_metrics,
            exit_context=task.exit_context,
        )
        if outcome_summary:
            query_parts.append(outcome_summary)

        execution_summary = self.summarize_execution_summary(task.execution_summary)
        if execution_summary:
            query_parts.append(execution_summary)

        post_trade_notes = self._join_text_fields(task.post_trade_notes, task.feedback_notes)
        if post_trade_notes:
            query_parts.append(post_trade_notes)

        overall_summary = str((task.analyst_payload or {}).get("overall_summary", "")).strip()
        if overall_summary:
            query_parts.append(overall_summary)

        for field_name, prefix in (
            ("key_signals", "signals"),
            ("portfolio_risks", "risks"),
            ("cross_analyst_observations", "observations"),
        ):
            values = [
                str(item).strip()
                for item in (task.analyst_payload or {}).get(field_name, [])
                if str(item).strip()
            ]
            if values:
                query_parts.append(prefix + " " + " ".join(values[:3]))

        return " ".join(part for part in query_parts if part)

    def build_reflection_profile(self, task: "ReflectionTask") -> dict[str, Any]:
        """Derive a compact postmortem profile from the current task."""
        decision_output = task.decision_output or {}
        decision_context = decision_output.get("decision_context", {})
        scenario_profile = decision_context.get("scenario_profile", {})
        return {
            "symbol": task.symbol,
            "recommendation": str(decision_output.get("recommendation", "")).strip().lower(),
            "decision_confidence": str(decision_output.get("confidence", "low")).strip().lower(),
            "outcome_label": infer_outcome_label(
                task.realized_outcome,
                outcome_metrics=task.outcome_metrics,
                exit_context=task.exit_context,
                post_trade_notes=task.post_trade_notes,
                feedback_notes=task.feedback_notes,
            ),
            "market_regime": str(
                scenario_profile.get("market_regime")
                or decision_output.get("market_regime")
                or "mixed"
            ).strip().lower(),
            "portfolio_state_tags": self._infer_portfolio_state_tags(task.portfolio_context, task.symbol),
            "decision_quality_hint": self._infer_decision_quality_hint(task),
            "exit_reason": str((task.exit_context or {}).get("exit_reason", "")).strip().lower(),
        }

    def retrieve_context(
        self,
        task: "ReflectionTask",
        *,
        query: str,
        reflection_profile: dict[str, Any],
        datasets: tuple[DatasetName, ...] | None = None,
        metadata_filter: dict[str, Any] | None = None,
        k: int | None = None,
    ) -> dict[str, Any]:
        """Retrieve and rank historical decision-memory records for reflection."""
        scenario_profile = {
            "market_regime": reflection_profile.get("market_regime", "mixed"),
            "analyst_alignment": "mixed",
            "signal_tags": [],
            "risk_tags": [],
            "timing_tags": [],
            "portfolio_state_tags": list(reflection_profile.get("portfolio_state_tags", [])),
        }
        return self.decision_service.retrieve_context(
            task,
            query=query,
            scenario_profile=scenario_profile,
            datasets=datasets,
            metadata_filter=metadata_filter,
            k=k,
        )

    def analyze(
        self,
        task: "ReflectionTask",
        *,
        datasets: tuple[DatasetName, ...] | None = None,
        metadata_filter: dict[str, Any] | None = None,
        k: int | None = None,
    ) -> dict[str, Any]:
        """Return a structured reflection payload for the post-decision agent."""
        selected_datasets = datasets or self.default_datasets
        merged_filter = dict(self.default_metadata_filter())
        if metadata_filter:
            merged_filter.update(metadata_filter)

        query = self.build_query(task)
        reflection_profile = self.build_reflection_profile(task)
        retrieval_context = self.retrieve_context(
            task,
            query=query,
            reflection_profile=reflection_profile,
            datasets=selected_datasets,
            metadata_filter=merged_filter,
            k=k,
        )
        return self.build_context(
            task,
            query=query,
            reflection_profile=reflection_profile,
            datasets=selected_datasets,
            ranked_documents=retrieval_context["ranked_documents"],
            validation_summary=retrieval_context["validation_summary"],
        )

    def build_context(
        self,
        task: "ReflectionTask",
        *,
        query: str,
        reflection_profile: dict[str, Any],
        datasets: tuple[DatasetName, ...],
        ranked_documents: list[dict[str, Any]],
        validation_summary: dict[str, Any],
    ) -> dict[str, Any]:
        """Build an agent-friendly postmortem context payload."""
        serialized_documents = [
            self.decision_service.serialize_document(item)
            for item in ranked_documents
        ]
        realized_outcome = self._normalize_realized_outcome(
            task.realized_outcome,
            outcome_metrics=task.outcome_metrics,
            exit_context=task.exit_context,
            post_trade_notes=task.post_trade_notes,
            feedback_notes=task.feedback_notes,
        )
        post_trade_validation = validate_post_trade_review(
            execution_summary=task.execution_summary,
            outcome_metrics=task.outcome_metrics,
            exit_context=task.exit_context,
        )
        post_trade_completeness = assess_post_trade_review_completeness(
            execution_summary=task.execution_summary,
            outcome_metrics=task.outcome_metrics,
            exit_context=task.exit_context,
            realized_outcome=task.realized_outcome,
            post_trade_notes=task.post_trade_notes,
            feedback_notes=task.feedback_notes,
        )
        candidate_memory_seed = {
            "title": self._build_candidate_title(task),
            "subject": task.subject,
            "symbol": task.symbol,
            "recommendation": str((task.decision_output or {}).get("recommendation", "")).strip()
            or "keep_watch",
            "confidence": str((task.decision_output or {}).get("confidence", "")).strip()
            or "medium",
            "outcome_label": reflection_profile.get("outcome_label", "unknown"),
            "exit_reason": str((task.exit_context or {}).get("exit_reason", "")).strip(),
        }
        return {
            "agent": self.agent_name,
            "subject": task.subject,
            "symbol": task.symbol,
            "trade_date": task.trade_date,
            "query": query,
            "reflection_profile": reflection_profile,
            "datasets": list(datasets),
            "document_count": len(serialized_documents),
            "validation_summary": validation_summary,
            "documents": serialized_documents,
            "historical_cases": serialized_documents,
            "evidence": self.decision_service.collect_evidence(serialized_documents),
            "original_decision": dict(task.decision_output or {}),
            "analyst_summary": self._build_analyst_summary(task.analyst_payload),
            "realized_outcome": realized_outcome,
            "execution_summary": dict(task.execution_summary or {}),
            "outcome_metrics": dict(task.outcome_metrics or {}),
            "exit_context": dict(task.exit_context or {}),
            "post_trade_validation": post_trade_validation,
            "post_trade_completeness": post_trade_completeness,
            "post_trade_notes": str(task.post_trade_notes or "").strip(),
            "feedback_notes": str(task.feedback_notes or "").strip(),
            "candidate_memory_seed": candidate_memory_seed,
        }

    def summarize_realized_outcome(
        self,
        realized_outcome: dict[str, Any] | None,
        *,
        outcome_metrics: dict[str, Any] | None = None,
        exit_context: dict[str, Any] | None = None,
    ) -> str:
        """Summarize outcome and performance fields into a compact retrieval string."""
        parts: list[str] = []
        if isinstance(realized_outcome, dict) and realized_outcome:
            parts.extend(
                str(realized_outcome.get(field_name, "")).strip()
                for field_name in ("outcome_label", "status", "summary", "result", "notes")
                if str(realized_outcome.get(field_name, "")).strip()
            )
        if isinstance(outcome_metrics, dict) and outcome_metrics:
            for field_name in (
                "realized_pnl_pct",
                "benchmark_relative_return_pct",
                "max_drawdown_pct",
                "holding_return_pct",
                "performance_assessment",
            ):
                value = str(outcome_metrics.get(field_name, "")).strip()
                if value:
                    parts.append(f"{field_name} {value}")
        if isinstance(exit_context, dict) and exit_context:
            for field_name in ("exit_reason", "exit_trigger", "summary"):
                value = str(exit_context.get(field_name, "")).strip()
                if value:
                    parts.append(value)
        return " ".join(parts)

    def summarize_execution_summary(self, execution_summary: dict[str, Any] | None) -> str:
        """Summarize execution fields for retrieval and prompt context."""
        if not isinstance(execution_summary, dict) or not execution_summary:
            return ""
        values = [
            str(execution_summary.get(field_name, "")).strip()
            for field_name in (
                "entry_date",
                "exit_date",
                "entry_price",
                "exit_price",
                "holding_period_days",
                "position_size_pct",
                "summary",
            )
        ]
        return " ".join(value for value in values if value)

    def build_candidate_memory(
        self,
        task: "ReflectionTask",
        *,
        reflection_summary: str,
        what_worked: list[str],
        what_failed_or_underweighted: list[str],
        lessons: list[str],
        future_adjustments: list[str],
    ) -> dict[str, Any]:
        """Build a candidate postmortem memory record from the current reflection task."""
        return build_candidate_postmortem_record(
            subject=task.subject,
            symbol=task.symbol,
            trade_date=task.trade_date,
            recommendation=str((task.decision_output or {}).get("recommendation", "")).strip()
            or "keep_watch",
            confidence=str((task.decision_output or {}).get("confidence", "")).strip()
            or "medium",
            outcome_label=infer_outcome_label(
                task.realized_outcome,
                outcome_metrics=task.outcome_metrics,
                exit_context=task.exit_context,
                post_trade_notes=task.post_trade_notes,
                feedback_notes=task.feedback_notes,
            ),
            reflection_summary=reflection_summary,
            what_worked=what_worked,
            what_failed_or_underweighted=what_failed_or_underweighted,
            lessons=lessons,
            future_adjustments=future_adjustments,
            execution_summary=task.execution_summary,
            outcome_metrics=task.outcome_metrics,
            exit_context=task.exit_context,
            post_trade_notes=self._join_text_fields(task.post_trade_notes, task.feedback_notes),
            dataset=str((task.datasets or self.default_datasets)[0]),
        )

    def _build_analyst_summary(self, analyst_payload: dict[str, Any] | None) -> dict[str, Any]:
        """Reduce the full analyst payload into a compact reflection block."""
        payload = analyst_payload or {}
        return {
            "overall_summary": str(payload.get("overall_summary", "")).strip(),
            "overall_confidence": str(payload.get("overall_confidence", "low")).strip(),
            "key_signals": [
                str(item).strip()
                for item in payload.get("key_signals", [])
                if str(item).strip()
            ],
            "portfolio_risks": [
                str(item).strip()
                for item in payload.get("portfolio_risks", [])
                if str(item).strip()
            ],
            "cross_analyst_observations": [
                str(item).strip()
                for item in payload.get("cross_analyst_observations", [])
                if str(item).strip()
            ],
        }

    def _normalize_realized_outcome(
        self,
        realized_outcome: dict[str, Any] | None,
        *,
        outcome_metrics: dict[str, Any] | None = None,
        exit_context: dict[str, Any] | None = None,
        post_trade_notes: str | None = None,
        feedback_notes: str | None = None,
    ) -> dict[str, Any]:
        """Normalize outcome data into a stable reflection payload."""
        normalized = dict(realized_outcome or {})
        normalized["outcome_label"] = infer_outcome_label(
            realized_outcome,
            outcome_metrics=outcome_metrics,
            exit_context=exit_context,
            post_trade_notes=post_trade_notes,
            feedback_notes=feedback_notes,
        )
        normalized["summary"] = self.summarize_realized_outcome(
            realized_outcome,
            outcome_metrics=outcome_metrics,
            exit_context=exit_context,
        )
        return normalized

    def _build_candidate_title(self, task: "ReflectionTask") -> str:
        """Build a stable candidate postmortem title."""
        if task.symbol:
            return f"{task.symbol.upper()} {task.subject} Postmortem"
        return f"{task.subject} Postmortem"

    def _infer_decision_quality_hint(self, task: "ReflectionTask") -> str:
        """Infer a coarse label describing how the original decision should be reviewed."""
        outcome_label = infer_outcome_label(
            task.realized_outcome,
            outcome_metrics=task.outcome_metrics,
            exit_context=task.exit_context,
            post_trade_notes=task.post_trade_notes,
            feedback_notes=task.feedback_notes,
        )
        if outcome_label == "failed":
            return "needs_deeper_revision"
        if outcome_label == "mixed":
            return "partial_reuse_only"
        if outcome_label == "worked":
            return "bounded_reinforcement"
        return "insufficient_outcome_signal"

    def _join_text_fields(self, *values: str | None) -> str:
        """Join optional text fields into one normalized string."""
        return " ".join(str(value).strip() for value in values if str(value or "").strip())

    def _infer_portfolio_state_tags(
        self,
        portfolio_context: dict[str, Any] | None,
        symbol: str | None,
    ) -> list[str]:
        """Infer a small set of portfolio-state tags for retrieval."""
        if not isinstance(portfolio_context, dict):
            return []
        tags: list[str] = []
        positions = portfolio_context.get("positions", [])
        if isinstance(positions, list) and positions:
            tags.append("has_positions")
        current_position = self._find_symbol_position(portfolio_context, symbol)
        if current_position is not None:
            tags.append("existing_position")
        else:
            tags.append("no_position")
        cash_pct = self._extract_percent(portfolio_context.get("cash_pct"))
        if cash_pct is not None:
            if cash_pct >= 10:
                tags.append("ample_cash")
            elif cash_pct < 5:
                tags.append("limited_cash")
        return tags

    def _find_symbol_position(
        self,
        portfolio_context: dict[str, Any],
        symbol: str | None,
    ) -> dict[str, Any] | None:
        """Return the current position object for the requested symbol when present."""
        if not symbol:
            return None
        positions = portfolio_context.get("positions", [])
        if not isinstance(positions, list):
            return None
        target_symbol = symbol.strip().upper()
        for position in positions:
            if not isinstance(position, dict):
                continue
            if str(position.get("symbol", "")).strip().upper() == target_symbol:
                return position
        return None

    def _extract_percent(self, value: Any) -> float | None:
        """Parse numeric percent-like values into floats."""
        if isinstance(value, bool) or value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            normalized = value.strip().rstrip("%")
            if not normalized:
                return None
            try:
                return float(normalized)
            except ValueError:
                return None
        return None
