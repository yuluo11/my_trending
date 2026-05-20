"""Persistence helpers for postmortem-guidance usage observations."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import re
from typing import Any

from ...knowledge.ingest import KnowledgeIngestor
from ...knowledge.repository import DatasetName, KnowledgeRepository


class DecisionGuidanceObservationService:
    """Persist structured records describing how decision runs used postmortem guidance."""

    def __init__(
        self,
        repository: KnowledgeRepository | None = None,
        ingestor: KnowledgeIngestor | None = None,
    ) -> None:
        self.repository = repository or KnowledgeRepository()
        self.ingestor = ingestor or KnowledgeIngestor(self.repository)

    def persist_guidance_observation(
        self,
        decision_result: dict[str, Any] | None,
        *,
        dataset: DatasetName = "dynamic",
        force: bool = False,
        record_name: str | None = None,
    ) -> dict[str, Any]:
        """Persist a decision-guidance observation when applied guidance is present."""
        if not isinstance(decision_result, dict):
            return {
                "status": "skipped",
                "persisted": False,
                "reason": "decision result must be a JSON object",
            }

        applied_guidance = self._normalize_string_list(
            decision_result.get("applied_postmortem_guidance")
        )
        if not applied_guidance and not force:
            return {
                "status": "skipped",
                "persisted": False,
                "reason": "decision result does not include applied_postmortem_guidance",
            }

        record = self.build_guidance_observation_record(
            decision_result,
            dataset=dataset,
            applied_guidance=applied_guidance,
        )
        target_name = record_name or self._build_record_name(decision_result, applied_guidance)

        self._ensure_repository_ready()
        record_path = self.ingestor.ingest_text(
            dataset,
            target_name,
            record["text"],
            metadata=record["metadata"],
        )
        return {
            "status": "persisted",
            "persisted": True,
            "path": str(record_path),
            "record_name": record_path.stem,
            "title": record["metadata"].get("title", record_path.stem),
            "applied_guidance_count": len(applied_guidance),
        }

    def build_guidance_observation_record(
        self,
        decision_result: dict[str, Any],
        *,
        dataset: DatasetName,
        applied_guidance: list[str] | None = None,
    ) -> dict[str, Any]:
        """Build a processed-record payload describing one decision's guidance usage."""
        normalized_guidance = applied_guidance or self._normalize_string_list(
            decision_result.get("applied_postmortem_guidance")
        )
        reference_cases = decision_result.get("reference_cases", [])
        reference_titles = [
            str(item.get("title", "")).strip()
            for item in reference_cases
            if isinstance(item, dict) and str(item.get("title", "")).strip()
        ]

        subject = str(decision_result.get("subject", "")).strip() or "Unspecified decision subject"
        symbol = str(decision_result.get("symbol", "")).strip().upper()
        recommendation = str(decision_result.get("recommendation", "")).strip().lower() or "keep_watch"
        confidence = str(decision_result.get("confidence", "")).strip().lower() or "medium"
        title = f"{symbol + ' ' if symbol else ''}{subject} Guidance Observation".strip()

        sections = [
            f"Decision summary: {str(decision_result.get('decision_summary', '')).strip()}",
            f"Recommendation: {recommendation}",
            f"Confidence: {confidence}",
            self._render_list_section("Applied postmortem guidance", normalized_guidance),
            self._render_text_section("Rationale", decision_result.get("rationale")),
            self._render_list_section("Reference cases", reference_titles),
            self._render_text_section(
                "Case fit assessment", decision_result.get("case_fit_assessment")
            ),
        ]
        text = "\n\n".join(section for section in sections if section).strip()

        metadata = {
            "source": "decision_guidance_observation",
            "title": title,
            "category": "decision_guidance_observation",
            "tags": self._build_tags(symbol, recommendation, normalized_guidance),
            "symbol": symbol,
            "topic": "decision-guidance-usage",
            "recommendation": recommendation,
            "confidence": confidence,
            "applied_guidance": normalized_guidance,
            "applied_guidance_count": len(normalized_guidance),
            "reference_case_titles": reference_titles,
            "dataset": dataset,
        }
        return {"text": text, "metadata": metadata}

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
        decision_result: dict[str, Any],
        applied_guidance: list[str],
    ) -> str:
        """Build a deterministic record name for persisted guidance observations."""
        symbol = str(decision_result.get("symbol", "")).strip().lower()
        trade_date = str(decision_result.get("trade_date", "")).strip()
        date_part = trade_date.split("T", 1)[0].replace("-", "_")
        subject = str(decision_result.get("subject", "")).strip().lower()

        name_parts = ["decision_guidance_observation"]
        if symbol:
            name_parts.append(symbol)
        if date_part:
            name_parts.append(date_part)
        if subject:
            name_parts.append(subject)
        if applied_guidance:
            name_parts.append(applied_guidance[0])

        raw_name = "_".join(name_parts)
        normalized = re.sub(r"[^a-z0-9]+", "_", raw_name.lower()).strip("_")
        return normalized or "decision_guidance_observation"

    def _normalize_string_list(self, value: Any) -> list[str]:
        """Normalize optional model output into a non-empty string list."""
        if isinstance(value, list):
            normalized = [str(item).strip() for item in value if str(item).strip()]
            return normalized
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        return []

    def _render_list_section(self, title: str, values: list[str]) -> str:
        """Render a simple bullet-list section."""
        if not values:
            return ""
        return f"{title}:\n" + "\n".join(f"- {value}" for value in values)

    def _render_text_section(self, title: str, value: Any) -> str:
        """Render a single text section."""
        normalized = str(value or "").strip()
        if not normalized:
            return ""
        return f"{title}: {normalized}"

    def _build_tags(
        self,
        symbol: str,
        recommendation: str,
        applied_guidance: list[str],
    ) -> list[str]:
        """Build stable metadata tags for later observation analysis."""
        tags = ["decision_guidance_observation", recommendation]
        if symbol:
            tags.append(symbol.lower())
        if applied_guidance:
            tags.append("postmortem_guidance_applied")
        seen: set[str] = set()
        normalized_tags: list[str] = []
        for tag in tags:
            normalized = str(tag).strip().lower().replace(" ", "_")
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            normalized_tags.append(normalized)
        return normalized_tags


class DecisionGuidanceObservationAnalyticsService:
    """Summarize persisted guidance-observation records for lightweight analysis."""

    def __init__(self, repository: KnowledgeRepository | None = None) -> None:
        self.repository = repository or KnowledgeRepository()

    def summarize_observations(
        self,
        *,
        dataset: DatasetName = "dynamic",
        symbol: str | None = None,
        recommendation: str | None = None,
        top_n: int = 5,
    ) -> dict[str, Any]:
        """Build a compact summary over persisted decision-guidance observations."""
        records = self.repository.load_all_processed_records(dataset)
        filtered_records = [
            record
            for record in records
            if self._is_guidance_observation(record)
            and self._matches_symbol(record, symbol)
            and self._matches_recommendation(record, recommendation)
        ]

        guidance_counter: Counter[str] = Counter()
        recommendation_counter: Counter[str] = Counter()
        symbol_counter: Counter[str] = Counter()
        reference_case_counter: Counter[str] = Counter()

        for record in filtered_records:
            metadata = dict(record.get("metadata", {}))
            for guidance in self._extract_applied_guidance(record):
                guidance_counter[guidance] += 1

            recommendation_label = str(metadata.get("recommendation", "")).strip().lower()
            if recommendation_label:
                recommendation_counter[recommendation_label] += 1

            symbol_label = str(metadata.get("symbol", "")).strip().upper()
            if symbol_label:
                symbol_counter[symbol_label] += 1

            for title in self._extract_reference_case_titles(record):
                reference_case_counter[title] += 1

        return {
            "dataset": dataset,
            "total_observations": len(filtered_records),
            "top_guidance": self._format_counter(guidance_counter, top_n=top_n),
            "recommendation_breakdown": self._format_counter(
                recommendation_counter,
                top_n=top_n,
            ),
            "symbol_breakdown": self._format_counter(symbol_counter, top_n=top_n),
            "top_reference_cases": self._format_counter(reference_case_counter, top_n=top_n),
        }

    def summarize_guidance_priors(
        self,
        *,
        datasets: tuple[DatasetName, ...] | list[DatasetName] | None = None,
        symbol: str | None = None,
        recommendation: str | None = None,
        top_n: int = 3,
    ) -> dict[str, Any]:
        """Build a compact guidance-prior summary for reuse in future decisions."""
        selected_datasets = tuple(datasets or ("dynamic",))
        guidance_counter: Counter[str] = Counter()
        recommendation_counter: Counter[str] = Counter()
        reference_case_counter: Counter[str] = Counter()
        total_observations = 0

        for dataset in selected_datasets:
            summary = self.summarize_observations(
                dataset=dataset,
                symbol=symbol,
                recommendation=recommendation,
                top_n=max(top_n * 2, 5),
            )
            total_observations += int(summary.get("total_observations", 0) or 0)
            for item in summary.get("top_guidance", []):
                label = str(item.get("label", "")).strip()
                count = int(item.get("count", 0) or 0)
                if label and count > 0:
                    guidance_counter[label] += count
            for item in summary.get("recommendation_breakdown", []):
                label = str(item.get("label", "")).strip()
                count = int(item.get("count", 0) or 0)
                if label and count > 0:
                    recommendation_counter[label] += count
            for item in summary.get("top_reference_cases", []):
                label = str(item.get("label", "")).strip()
                count = int(item.get("count", 0) or 0)
                if label and count > 0:
                    reference_case_counter[label] += count

        top_guidance = self._format_counter(guidance_counter, top_n=top_n)
        recommendation_breakdown = self._format_counter(recommendation_counter, top_n=top_n)
        top_reference_cases = self._format_counter(reference_case_counter, top_n=top_n)
        normalized_symbol = symbol.strip().upper() if isinstance(symbol, str) and symbol.strip() else None
        return {
            "datasets": list(selected_datasets),
            "symbol": normalized_symbol,
            "recommendation_filter": recommendation.strip().lower()
            if isinstance(recommendation, str) and recommendation.strip()
            else None,
            "total_observations": total_observations,
            "top_guidance": top_guidance,
            "recommendation_breakdown": recommendation_breakdown,
            "top_reference_cases": top_reference_cases,
            "summary": self._build_guidance_prior_summary(
                symbol=normalized_symbol,
                total_observations=total_observations,
                top_guidance=top_guidance,
                recommendation_breakdown=recommendation_breakdown,
            ),
        }

    def _is_guidance_observation(self, record: dict[str, Any]) -> bool:
        """Return whether a processed record is a decision-guidance observation."""
        metadata = dict(record.get("metadata", {}))
        return str(metadata.get("category", "")).strip() == "decision_guidance_observation"

    def _matches_symbol(self, record: dict[str, Any], symbol: str | None) -> bool:
        """Apply an optional symbol filter."""
        if not symbol:
            return True
        metadata = dict(record.get("metadata", {}))
        return str(metadata.get("symbol", "")).strip().upper() == symbol.strip().upper()

    def _matches_recommendation(
        self,
        record: dict[str, Any],
        recommendation: str | None,
    ) -> bool:
        """Apply an optional recommendation filter."""
        if not recommendation:
            return True
        metadata = dict(record.get("metadata", {}))
        return (
            str(metadata.get("recommendation", "")).strip().lower()
            == recommendation.strip().lower()
        )

    def _extract_applied_guidance(self, record: dict[str, Any]) -> list[str]:
        """Extract applied guidance strings from metadata or fallback text parsing."""
        metadata = dict(record.get("metadata", {}))
        raw_value = metadata.get("applied_guidance", [])
        if isinstance(raw_value, list):
            normalized = [str(item).strip() for item in raw_value if str(item).strip()]
            if normalized:
                return normalized

        text = str(record.get("text", "")).strip()
        match = re.search(r"Applied postmortem guidance:\n((?:- .+\n?){1,8})", text)
        if not match:
            return []
        return [
            line.removeprefix("- ").strip()
            for line in match.group(1).splitlines()
            if line.removeprefix("- ").strip()
        ]

    def _extract_reference_case_titles(self, record: dict[str, Any]) -> list[str]:
        """Extract reference-case titles from metadata or fallback text parsing."""
        metadata = dict(record.get("metadata", {}))
        raw_value = metadata.get("reference_case_titles", [])
        if isinstance(raw_value, list):
            normalized = [str(item).strip() for item in raw_value if str(item).strip()]
            if normalized:
                return normalized

        text = str(record.get("text", "")).strip()
        match = re.search(r"Reference cases:\n((?:- .+\n?){1,8})", text)
        if not match:
            return []
        return [
            line.removeprefix("- ").strip()
            for line in match.group(1).splitlines()
            if line.removeprefix("- ").strip()
        ]

    def _format_counter(self, counter: Counter[str], *, top_n: int) -> list[dict[str, Any]]:
        """Format a counter into a stable list of count entries."""
        return [
            {"label": label, "count": count}
            for label, count in counter.most_common(top_n)
        ]

    def _build_guidance_prior_summary(
        self,
        *,
        symbol: str | None,
        total_observations: int,
        top_guidance: list[dict[str, Any]],
        recommendation_breakdown: list[dict[str, Any]],
    ) -> str:
        """Render a short natural-language summary for prompt usage."""
        if total_observations <= 0 or not top_guidance:
            return ""

        symbol_prefix = f"For {symbol}, " if symbol else ""
        lead_guidance = top_guidance[0]
        guidance_label = str(lead_guidance.get("label", "")).strip()
        guidance_count = int(lead_guidance.get("count", 0) or 0)
        summary = (
            f"{symbol_prefix}recurring applied guidance has most often emphasized "
            f"'{guidance_label}' ({guidance_count} observation"
            f"{'' if guidance_count == 1 else 's'}"
            ")."
        )
        if recommendation_breakdown:
            labels = ", ".join(
                str(item.get("label", "")).strip()
                for item in recommendation_breakdown[:2]
                if str(item.get("label", "")).strip()
            )
            if labels:
                summary += f" It has most often appeared with {labels} decisions."
        return summary
