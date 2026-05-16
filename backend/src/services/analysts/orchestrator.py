"""Internal orchestration layer for coordinating multiple analyst agents."""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Any

from ...llm.client import LLMClient
from .base_agent import AnalystTask, BaseLangGraphAnalystAgent


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    """Return a de-duplicated list while preserving the first-seen order."""
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        normalized = value.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


@dataclass(slots=True)
class AnalystOrchestrator:
    """Internal coordinator that powers analyst realization runs."""

    analysts: dict[str, BaseLangGraphAnalystAgent]
    sequence: tuple[str, ...]
    llm_client: LLMClient | None = None
    prompts_dir: Path | None = None

    def run(self, task: AnalystTask) -> dict[str, Any]:
        """Execute each analyst in sequence and return an aggregated result."""
        analyst_results: list[dict[str, Any]] = []
        messages: list[Any] = list(task.messages)

        for analyst_name in self.sequence:
            analyst = self.analysts[analyst_name]
            staged_task = AnalystTask(
                subject=task.subject,
                symbol=task.symbol,
                trade_date=task.trade_date,
                extra_context=task.extra_context,
                datasets=task.datasets,
                metadata_filter=task.metadata_filter,
                max_documents=task.max_documents,
                messages=list(messages),
            )
            result = analyst.invoke(staged_task)
            analyst_results.append(result)
            messages.append(analyst.build_agent_message(result))

        return self.aggregate(task, analyst_results=analyst_results, messages=messages)

    def aggregate(
        self,
        task: AnalystTask,
        *,
        analyst_results: list[dict[str, Any]],
        messages: list[Any],
    ) -> dict[str, Any]:
        """Aggregate analyst outputs into a single orchestration payload."""
        fallback_key_signals = _dedupe_preserve_order(
            [
                signal
                for result in analyst_results
                for signal in result.get("signals", [])
            ]
        )
        fallback_key_risks = _dedupe_preserve_order(
            [
                risk
                for result in analyst_results
                for risk in result.get("risks", [])
            ]
        )
        fallback_cross_analyst_observations = self.build_cross_analyst_observations(analyst_results)
        fallback_overall_summary = self.build_overall_summary(
            task,
            analyst_results=analyst_results,
            key_signals=fallback_key_signals,
            key_risks=fallback_key_risks,
        )
        fallback_overall_confidence = self.calculate_overall_confidence(analyst_results)

        synthesis = self.synthesize_realization(
            task,
            analyst_results=analyst_results,
            fallback_summary=fallback_overall_summary,
            fallback_confidence=fallback_overall_confidence,
            fallback_signals=fallback_key_signals,
            fallback_risks=fallback_key_risks,
            fallback_observations=fallback_cross_analyst_observations,
        )

        return {
            "subject": task.subject,
            "symbol": task.symbol,
            "trade_date": task.trade_date,
            "extra_context": task.extra_context,
            "analyst_sequence": list(self.sequence),
            "overall_summary": synthesis["overall_summary"],
            "overall_confidence": synthesis["overall_confidence"],
            "key_signals": synthesis["key_signals"],
            "portfolio_risks": synthesis["portfolio_risks"],
            "cross_analyst_observations": synthesis["cross_analyst_observations"],
            "analyst_results": analyst_results,
            "message_count": len(messages),
            "messages": messages,
        }

    def synthesize_realization(
        self,
        task: AnalystTask,
        *,
        analyst_results: list[dict[str, Any]],
        fallback_summary: str,
        fallback_confidence: str,
        fallback_signals: list[str],
        fallback_risks: list[str],
        fallback_observations: list[str],
    ) -> dict[str, Any]:
        """Synthesize a top-level realization via LLM or fallback aggregation."""
        if self.llm_client is None:
            return {
                "overall_summary": fallback_summary,
                "overall_confidence": fallback_confidence,
                "key_signals": fallback_signals,
                "portfolio_risks": fallback_risks,
                "cross_analyst_observations": fallback_observations,
            }

        prompt = self.render_realization_prompt(task)
        llm_payload = self.build_realization_payload(task, analyst_results=analyst_results)
        llm_result = self.llm_client.invoke_json(
            prompt,
            payload=llm_payload,
            schema=self.realization_output_schema(),
        )
        return self.normalize_realization_result(
            llm_result,
            fallback_summary=fallback_summary,
            fallback_confidence=fallback_confidence,
            fallback_signals=fallback_signals,
            fallback_risks=fallback_risks,
            fallback_observations=fallback_observations,
        )

    def render_realization_prompt(self, task: AnalystTask) -> str:
        """Render the prompt used for top-level realization synthesis."""
        shared_prompt = self._read_prompt("shared/base_prompt.txt")
        orchestration_prompt = self._read_prompt("orchestration/analyst_realization.txt")
        context_lines = [
            f"subject: {task.subject}",
            f"symbol: {task.symbol or 'N/A'}",
            f"trade_date: {task.trade_date or 'N/A'}",
            f"extra_context: {task.extra_context or 'N/A'}",
        ]
        return "\n\n".join(
            block
            for block in (
                shared_prompt,
                orchestration_prompt,
                "\n".join(context_lines),
            )
            if block
        )

    def build_realization_payload(
        self,
        task: AnalystTask,
        *,
        analyst_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Build the structured payload sent into the orchestrator synthesis LLM."""
        return {
            "task": {
                "subject": task.subject,
                "symbol": task.symbol,
                "trade_date": task.trade_date,
                "extra_context": task.extra_context,
            },
            "analyst_sequence": list(self.sequence),
            "analyst_results": analyst_results,
            "instructions": (
                "Return a JSON object with keys: overall_summary, overall_confidence, "
                "key_signals, portfolio_risks, and cross_analyst_observations."
            ),
        }

    def realization_output_schema(self) -> dict[str, Any]:
        """Return the target output schema for orchestrator synthesis."""
        return {
            "overall_summary": "string",
            "overall_confidence": "low|medium|high",
            "key_signals": ["string"],
            "portfolio_risks": ["string"],
            "cross_analyst_observations": ["string"],
        }

    def normalize_realization_result(
        self,
        llm_result: dict[str, Any],
        *,
        fallback_summary: str,
        fallback_confidence: str,
        fallback_signals: list[str],
        fallback_risks: list[str],
        fallback_observations: list[str],
    ) -> dict[str, Any]:
        """Normalize LLM synthesis output into the realization result shape."""
        confidence = str(llm_result.get("overall_confidence", fallback_confidence)).strip().lower()
        if confidence not in {"low", "medium", "high"}:
            confidence = fallback_confidence

        return {
            "overall_summary": str(llm_result.get("overall_summary", "")).strip()
            or fallback_summary,
            "overall_confidence": confidence,
            "key_signals": self._normalize_string_list(
                llm_result.get("key_signals"),
                fallback=fallback_signals,
            ),
            "portfolio_risks": self._normalize_string_list(
                llm_result.get("portfolio_risks"),
                fallback=fallback_risks,
            ),
            "cross_analyst_observations": self._normalize_string_list(
                llm_result.get("cross_analyst_observations"),
                fallback=fallback_observations,
            ),
        }

    def build_overall_summary(
        self,
        task: AnalystTask,
        *,
        analyst_results: list[dict[str, Any]],
        key_signals: list[str],
        key_risks: list[str],
    ) -> str:
        """Build a concise top-level summary from the analyst outputs."""
        confidence = self.calculate_overall_confidence(analyst_results)
        first_signal = key_signals[0] if key_signals else "No shared bullish signal was identified."
        first_risk = key_risks[0] if key_risks else "No major cross-analyst risk was identified."
        return (
            f"Analyst orchestrator reviewed {len(analyst_results)} analyst perspectives for "
            f"{task.subject}. Overall confidence is {confidence}. "
            f"Leading signal: {first_signal} Leading risk: {first_risk}"
        )

    def calculate_overall_confidence(self, analyst_results: list[dict[str, Any]]) -> str:
        """Collapse per-analyst confidence levels into a single label."""
        confidence_scores = {"low": 1, "medium": 2, "high": 3}
        if not analyst_results:
            return "low"

        scores = [
            confidence_scores.get(str(result.get("confidence", "medium")).lower(), 2)
            for result in analyst_results
        ]
        average_score = sum(scores) / len(scores)
        if average_score >= 2.5:
            return "high"
        if average_score >= 1.5:
            return "medium"
        return "low"

    def build_cross_analyst_observations(
        self,
        analyst_results: list[dict[str, Any]],
    ) -> list[str]:
        """Extract lightweight cross-analyst observations from the results."""
        observations: list[str] = []
        analyst_names = [result.get("analyst", "unknown_analyst") for result in analyst_results]

        high_confidence_analysts = [
            result.get("analyst", "unknown_analyst")
            for result in analyst_results
            if str(result.get("confidence", "")).lower() == "high"
        ]
        if high_confidence_analysts:
            observations.append(
                "High-confidence support came from: " + ", ".join(high_confidence_analysts) + "."
            )

        empty_document_analysts = [
            result.get("analyst", "unknown_analyst")
            for result in analyst_results
            if not result.get("documents")
        ]
        if empty_document_analysts:
            observations.append(
                "These analysts had thin knowledge coverage: "
                + ", ".join(empty_document_analysts)
                + "."
            )

        if len(analyst_names) >= 2:
            observations.append(
                "The orchestration preserved analyst sequencing across: "
                + " -> ".join(analyst_names)
                + "."
            )

        if not observations:
            observations.append("Cross-analyst observations are limited in the current run.")
        return observations

    def _normalize_string_list(self, value: Any, *, fallback: list[str]) -> list[str]:
        """Normalize an LLM output field into a non-empty string list."""
        if isinstance(value, list):
            normalized = [str(item).strip() for item in value if str(item).strip()]
            return normalized or fallback
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        return fallback

    def _read_prompt(self, relative_path: str) -> str:
        """Read a prompt asset relative to the orchestrator prompt root."""
        if self.prompts_dir is None:
            return ""
        prompt_path = self.prompts_dir / relative_path
        if not prompt_path.exists():
            return ""
        return prompt_path.read_text(encoding="utf-8").strip()
