"""Agent primitives for post-decision reflection and lesson extraction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypedDict

from ...knowledge.repository import DatasetName
from ...llm.client import LLMClient, LLMRunnable, ensure_llm_client
from ...services.decision.memory import validate_decision_memory_record
from ...services.reflection import (
    ReflectionContextService,
    assess_memory_persistence_candidate,
    infer_outcome_label,
    normalize_confidence_change,
    normalize_execution_summary,
    normalize_exit_context,
    normalize_outcome_metrics,
)
from ..decision.base_agent import FilePromptProvider, PromptProvider

ALLOWED_CONFIDENCE_CHANGES = {"increase", "keep", "decrease"}


@dataclass(slots=True)
class ReflectionTask:
    """Structured input consumed by the post-decision reflection agent."""

    subject: str
    symbol: str | None = None
    trade_date: str | None = None
    extra_context: str | None = None
    decision_output: dict[str, Any] = field(default_factory=dict)
    analyst_payload: dict[str, Any] = field(default_factory=dict)
    portfolio_context: dict[str, Any] | None = None
    execution_summary: dict[str, Any] | None = None
    outcome_metrics: dict[str, Any] | None = None
    exit_context: dict[str, Any] | None = None
    post_trade_notes: str | None = None
    realized_outcome: dict[str, Any] | None = None
    feedback_notes: str | None = None
    datasets: tuple[DatasetName, ...] | None = None
    metadata_filter: dict[str, Any] | None = None
    max_documents: int | None = None
    messages: list[Any] = field(default_factory=list)

    @classmethod
    def from_decision_payload(
        cls,
        decision_output: dict[str, Any],
        *,
        analyst_payload: dict[str, Any] | None = None,
        portfolio_context: dict[str, Any] | None = None,
        execution_summary: dict[str, Any] | None = None,
        outcome_metrics: dict[str, Any] | None = None,
        exit_context: dict[str, Any] | None = None,
        post_trade_notes: str | None = None,
        realized_outcome: dict[str, Any] | None = None,
        feedback_notes: str | None = None,
        datasets: tuple[DatasetName, ...] | None = None,
        metadata_filter: dict[str, Any] | None = None,
        max_documents: int | None = None,
        messages: list[Any] | None = None,
    ) -> "ReflectionTask":
        """Build a reflection task from a finalized decision payload."""
        return cls(
            subject=str(
                decision_output.get("subject")
                or (analyst_payload or {}).get("subject")
                or ""
            ).strip(),
            symbol=decision_output.get("symbol") or (analyst_payload or {}).get("symbol"),
            trade_date=decision_output.get("trade_date") or (analyst_payload or {}).get("trade_date"),
            extra_context=(analyst_payload or {}).get("extra_context"),
            decision_output=dict(decision_output or {}),
            analyst_payload=dict(analyst_payload or {}),
            portfolio_context=portfolio_context,
            execution_summary=normalize_execution_summary(execution_summary) or None,
            outcome_metrics=normalize_outcome_metrics(outcome_metrics) or None,
            exit_context=normalize_exit_context(exit_context) or None,
            post_trade_notes=str(post_trade_notes or "").strip() or None,
            realized_outcome=dict(realized_outcome or {}) or None,
            feedback_notes=str(feedback_notes or "").strip() or None,
            datasets=datasets,
            metadata_filter=metadata_filter,
            max_documents=max_documents,
            messages=list(messages or decision_output.get("messages", [])),
        )


class ReflectionRuntimeState(TypedDict, total=False):
    """State shape for future reflection-oriented graph composition."""

    subject: str
    symbol: str | None
    trade_date: str | None
    extra_context: str | None
    decision_output: dict[str, Any]
    analyst_payload: dict[str, Any]
    portfolio_context: dict[str, Any] | None
    execution_summary: dict[str, Any] | None
    outcome_metrics: dict[str, Any] | None
    exit_context: dict[str, Any] | None
    post_trade_notes: str | None
    realized_outcome: dict[str, Any] | None
    feedback_notes: str | None
    datasets: tuple[DatasetName, ...] | list[DatasetName] | None
    metadata_filter: dict[str, Any] | None
    max_documents: int | None
    messages: list[Any]
    reflection_output: dict[str, Any]


class BaseReflectionAgent:
    """Shared implementation for post-decision reflection agents."""

    agent_name = "reflection_agent"

    def __init__(
        self,
        *,
        agent_name: str | None = None,
        knowledge_service: ReflectionContextService,
        prompt_provider: PromptProvider | None = None,
        llm_client: LLMClient | None = None,
        llm: LLMRunnable | None = None,
    ) -> None:
        if agent_name:
            self.agent_name = agent_name
        self.knowledge_service = knowledge_service
        self.prompt_provider = prompt_provider
        self.llm_client = ensure_llm_client(llm_client=llm_client, llm=llm)

    def retrieve_reflection_context(self, task: ReflectionTask) -> dict[str, Any]:
        """Retrieve historical context relevant to the current reflection task."""
        return self.knowledge_service.analyze(
            task,
            datasets=task.datasets,
            metadata_filter=task.metadata_filter,
            k=task.max_documents,
        )

    def build_prompt_context(
        self,
        task: ReflectionTask,
        reflection_context: dict[str, Any],
    ) -> dict[str, Any]:
        """Build the runtime context used while rendering the reflection prompt."""
        return {
            "subject": task.subject,
            "symbol": task.symbol,
            "trade_date": task.trade_date,
            "outcome_label": infer_outcome_label(
                task.realized_outcome,
                outcome_metrics=task.outcome_metrics,
                exit_context=task.exit_context,
                post_trade_notes=task.post_trade_notes,
                feedback_notes=task.feedback_notes,
            ),
            "post_trade_notes": self._join_text_fields(task.post_trade_notes, task.feedback_notes),
            "decision_summary": str((task.decision_output or {}).get("decision_summary", "")).strip(),
            "recommendation": str((task.decision_output or {}).get("recommendation", "")).strip(),
            "reflection_context": reflection_context,
        }

    def render_prompt(self, prompt_context: dict[str, Any]) -> str:
        """Render the reflection prompt from shared and role-specific prompt assets."""
        if self.prompt_provider is None:
            return ""

        shared_prompt = self.prompt_provider.get_shared_prompt()
        role_prompt = self.prompt_provider.get_analyst_prompt(self.agent_name)
        context_lines = [
            f"subject: {prompt_context.get('subject', '')}",
            f"symbol: {prompt_context.get('symbol') or 'N/A'}",
            f"trade_date: {prompt_context.get('trade_date') or 'N/A'}",
            f"recommendation: {prompt_context.get('recommendation') or 'N/A'}",
            f"outcome_label: {prompt_context.get('outcome_label') or 'unknown'}",
            f"post_trade_notes: {prompt_context.get('post_trade_notes') or 'N/A'}",
            (
                "historical_reflection_documents: "
                f"{prompt_context['reflection_context'].get('document_count', 0)}"
            ),
        ]
        return "\n\n".join(
            block
            for block in (
                shared_prompt,
                role_prompt,
                "\n".join(context_lines),
            )
            if block
        )

    def build_llm_payload(
        self,
        task: ReflectionTask,
        *,
        reflection_context: dict[str, Any],
    ) -> dict[str, Any]:
        """Build the structured payload sent into the configured llm_client."""
        return {
            "task": {
                "subject": task.subject,
                "symbol": task.symbol,
                "trade_date": task.trade_date,
                "portfolio_context": task.portfolio_context or {},
            },
            "original_decision": dict(task.decision_output or {}),
            "analyst_context": reflection_context.get("analyst_summary", {}),
            "post_trade_review": {
                "execution_summary": reflection_context.get("execution_summary", {}),
                "realized_outcome": reflection_context.get("realized_outcome", {}),
                "outcome_metrics": reflection_context.get("outcome_metrics", {}),
                "exit_context": reflection_context.get("exit_context", {}),
                "post_trade_notes": self._join_text_fields(
                    task.post_trade_notes,
                    task.feedback_notes,
                ),
            },
            "historical_context": {
                "query": reflection_context.get("query", ""),
                "reflection_profile": reflection_context.get("reflection_profile", {}),
                "document_count": reflection_context.get("document_count", 0),
                "validation_summary": reflection_context.get("validation_summary", {}),
                "documents": reflection_context.get("documents", []),
                "evidence": reflection_context.get("evidence", []),
            },
            "instructions": (
                "Return a JSON object with keys: reflection_summary, what_worked, "
                "what_failed_or_underweighted, lessons, future_adjustments, "
                "confidence_change, reference_cases, and candidate_memory. Keep the "
                "adjustments bounded and avoid overstating certainty from a single case."
            ),
        }

    def invoke(self, task: ReflectionTask) -> dict[str, Any]:
        """Run the reflection flow end to end."""
        reflection_context = self.retrieve_reflection_context(task)
        prompt_context = self.build_prompt_context(task, reflection_context)
        prompt = self.render_prompt(prompt_context)
        return self.synthesize(task, reflection_context, prompt)

    def synthesize(
        self,
        task: ReflectionTask,
        reflection_context: dict[str, Any],
        prompt: str,
    ) -> dict[str, Any]:
        """Create the final reflection payload using the model or fallback logic."""
        if self.llm_client is not None:
            return self._synthesize_with_llm(task, reflection_context, prompt)
        return self._synthesize_fallback(task, reflection_context, prompt)

    def _synthesize_with_llm(
        self,
        task: ReflectionTask,
        reflection_context: dict[str, Any],
        prompt: str,
    ) -> dict[str, Any]:
        """Invoke the llm_client and normalize the reflection output."""
        llm_payload = self.build_llm_payload(task, reflection_context=reflection_context)
        parsed_response = self.llm_client.invoke_json(
            prompt,
            payload=llm_payload,
            schema=self.reflection_output_schema(),
        )
        return self.normalize_llm_result(
            parsed_response,
            task=task,
            reflection_context=reflection_context,
            prompt=prompt,
        )

    def _synthesize_fallback(
        self,
        task: ReflectionTask,
        reflection_context: dict[str, Any],
        prompt: str,
    ) -> dict[str, Any]:
        """Produce a cautious deterministic reflection result."""
        outcome_label = infer_outcome_label(
            task.realized_outcome,
            outcome_metrics=task.outcome_metrics,
            exit_context=task.exit_context,
            post_trade_notes=task.post_trade_notes,
            feedback_notes=task.feedback_notes,
        )
        reference_cases = self._fallback_reference_cases(reflection_context)
        what_worked = self._fallback_what_worked(task, reflection_context, outcome_label)
        what_failed = self._fallback_what_failed_or_underweighted(task, reflection_context, outcome_label)
        lessons = self._fallback_lessons(task, outcome_label, what_worked, what_failed)
        future_adjustments = self._fallback_future_adjustments(task, outcome_label)
        confidence_change = self._fallback_confidence_change(outcome_label)
        reflection_summary = self._fallback_reflection_summary(task, outcome_label)
        candidate_memory = self.knowledge_service.build_candidate_memory(
            task,
            reflection_summary=reflection_summary,
            what_worked=what_worked,
            what_failed_or_underweighted=what_failed,
            lessons=lessons,
            future_adjustments=future_adjustments,
        )
        result = {
            "subject": task.subject,
            "symbol": task.symbol,
            "trade_date": task.trade_date,
            "reflection_summary": reflection_summary,
            "what_worked": what_worked,
            "what_failed_or_underweighted": what_failed,
            "lessons": lessons,
            "future_adjustments": future_adjustments,
            "confidence_change": confidence_change,
            "reference_cases": reference_cases,
            "candidate_memory": candidate_memory,
            "prompt": prompt,
            "reflection_context": reflection_context,
        }
        result["memory_persistence"] = assess_memory_persistence_candidate(
            reflection_result=result,
            outcome_label=outcome_label,
            post_trade_validation=reflection_context.get("post_trade_validation"),
        )
        return result

    def normalize_llm_result(
        self,
        llm_result: dict[str, Any],
        *,
        task: ReflectionTask,
        reflection_context: dict[str, Any],
        prompt: str,
    ) -> dict[str, Any]:
        """Normalize LLM output into the shared reflection payload shape."""
        fallback_result = self._synthesize_fallback(task, reflection_context, prompt)
        candidate_memory = llm_result.get("candidate_memory")
        if not self._is_valid_candidate_memory(candidate_memory):
            candidate_memory = fallback_result["candidate_memory"]

        result = {
            "subject": task.subject,
            "symbol": task.symbol,
            "trade_date": task.trade_date,
            "reflection_summary": str(
                llm_result.get("reflection_summary", fallback_result["reflection_summary"])
            ).strip()
            or fallback_result["reflection_summary"],
            "what_worked": self._normalize_string_list(
                llm_result.get("what_worked"),
                fallback=fallback_result["what_worked"],
            ),
            "what_failed_or_underweighted": self._normalize_string_list(
                llm_result.get("what_failed_or_underweighted"),
                fallback=fallback_result["what_failed_or_underweighted"],
            ),
            "lessons": self._normalize_string_list(
                llm_result.get("lessons"),
                fallback=fallback_result["lessons"],
            ),
            "future_adjustments": self._normalize_string_list(
                llm_result.get("future_adjustments"),
                fallback=fallback_result["future_adjustments"],
            ),
            "confidence_change": normalize_confidence_change(
                llm_result.get("confidence_change"),
                default=fallback_result["confidence_change"],
            ),
            "reference_cases": self._normalize_reference_cases(
                llm_result.get("reference_cases"),
                fallback=fallback_result["reference_cases"],
            ),
            "candidate_memory": candidate_memory,
            "prompt": prompt,
            "reflection_context": reflection_context,
            "raw_model_output": llm_result,
        }
        result["memory_persistence"] = assess_memory_persistence_candidate(
            reflection_result=result,
            outcome_label=infer_outcome_label(
                task.realized_outcome,
                outcome_metrics=task.outcome_metrics,
                exit_context=task.exit_context,
                post_trade_notes=task.post_trade_notes,
                feedback_notes=task.feedback_notes,
            ),
            post_trade_validation=reflection_context.get("post_trade_validation"),
        )
        return result

    def reflection_output_schema(self) -> dict[str, Any]:
        """Return the target structured schema for reflection outputs."""
        return {
            "reflection_summary": "string",
            "what_worked": ["string"],
            "what_failed_or_underweighted": ["string"],
            "lessons": ["string"],
            "future_adjustments": ["string"],
            "confidence_change": "increase|keep|decrease",
            "reference_cases": [
                {
                    "title": "string",
                    "memory_type": "decision_case|decision_postmortem|external_reference_decision",
                    "fit": "high|medium|low",
                    "why_relevant": "string",
                }
            ],
            "candidate_memory": {"text": "string", "metadata": {"title": "string"}},
        }

    def build_agent_message(self, result: dict[str, Any]) -> Any:
        """Create a graph-friendly message from the reflection payload."""
        content = result.get("reflection_summary", "")
        try:
            from langchain_core.messages import AIMessage
        except ModuleNotFoundError:
            return {
                "role": "assistant",
                "name": self.agent_name,
                "content": content,
            }
        return AIMessage(content=content, name=self.agent_name)

    def _fallback_reflection_summary(self, task: ReflectionTask, outcome_label: str) -> str:
        """Build a bounded summary of the post-decision review."""
        recommendation = str((task.decision_output or {}).get("recommendation", "keep_watch")).strip()
        if outcome_label == "worked":
            return (
                f"The original {recommendation} stance for {task.subject} appears directionally "
                "reasonable, but the lesson should be reused cautiously and with scenario-fit checks."
            )
        if outcome_label == "failed":
            return (
                f"The original {recommendation} stance for {task.subject} needs revision because "
                "later evidence suggests important risks or disconfirming signals were underweighted."
            )
        if outcome_label == "mixed":
            return (
                f"The original {recommendation} stance for {task.subject} captured part of the "
                "setup, but the review points to uneven signal quality and only partial reuse."
            )
        return (
            f"The original {recommendation} stance for {task.subject} can be reviewed for reusable "
            "reasoning patterns, but the realized outcome is not yet strong enough to justify a firm conclusion."
        )

    def _fallback_what_worked(
        self,
        task: ReflectionTask,
        reflection_context: dict[str, Any],
        outcome_label: str,
    ) -> list[str]:
        """List what the original decision process handled well."""
        worked: list[str] = []
        if (task.decision_output or {}).get("portfolio_context_used"):
            worked.append("The original decision incorporated portfolio context rather than treating the setup in isolation.")
        if reflection_context.get("document_count", 0):
            worked.append("Historical decision-memory cases were available to frame the current setup against prior analogs.")
        if outcome_label == "worked":
            worked.append("The final recommendation stayed bounded enough to align with the later realized outcome.")
        elif outcome_label == "mixed":
            worked.append("Parts of the decision logic remained useful even though the final outcome was not cleanly one-sided.")
        else:
            worked.append("The reflection still has reusable context about how the setup was framed at decision time.")
        return worked

    def _fallback_what_failed_or_underweighted(
        self,
        task: ReflectionTask,
        reflection_context: dict[str, Any],
        outcome_label: str,
    ) -> list[str]:
        """List what the original decision process missed or underweighted."""
        failed: list[str] = []
        post_trade_notes = self._join_text_fields(task.post_trade_notes, task.feedback_notes)
        if outcome_label == "failed":
            failed.append("The original decision did not sufficiently stress-test the thesis against adverse or disconfirming outcomes.")
        if outcome_label == "mixed":
            failed.append("The original decision treated the setup too cleanly and underweighted scenario branching.")
        if not reflection_context.get("document_count", 0):
            failed.append("No strong historical analog was retrieved, which limited external checks on the original reasoning.")
        realized_pnl = self._extract_float(
            (task.outcome_metrics or {}).get(
                "realized_pnl_pct",
                (task.outcome_metrics or {}).get("pnl_pct"),
            )
        )
        if realized_pnl is not None and realized_pnl < 0:
            failed.append(
                f"The trade lost about {realized_pnl:.2f}% on a realized basis, which suggests the original thesis or timing assumptions need tightening."
            )
        if post_trade_notes:
            failed.append(f"Later review notes raised additional points: {post_trade_notes}")
        if not failed:
            failed.append("No major reasoning failure was confirmed, but the review should avoid over-generalizing from one case.")
        return failed

    def _fallback_lessons(
        self,
        task: ReflectionTask,
        outcome_label: str,
        what_worked: list[str],
        what_failed_or_underweighted: list[str],
    ) -> list[str]:
        """Extract reusable lessons from the reflection review."""
        lessons = [
            "Reuse historical cases only when the current setup matches on scenario fit, not just superficial similarity.",
        ]
        holding_period_days = self._extract_float((task.execution_summary or {}).get("holding_period_days"))
        if holding_period_days is not None:
            lessons.append(
                f"The realized holding period was about {holding_period_days:.0f} day(s), so future reviews should check whether the original thesis horizon matched the actual trade duration."
            )
        if outcome_label == "failed":
            lessons.append("Future reviews should explicitly compare the base case against at least one failed analog before reusing the pattern.")
        if outcome_label == "worked":
            lessons.append("A correct outcome still warrants bounded reuse because a single successful case does not prove the broader rule.")
        if (task.decision_output or {}).get("portfolio_context_used"):
            lessons.append("Portfolio constraints should remain part of postmortem review because they materially shape the quality of the recommendation.")
        if len(what_failed_or_underweighted) > len(what_worked):
            lessons.append("When underweighted risks dominate the review, future confidence should stay conservative until the missing checks are fixed.")
        return lessons

    def _fallback_future_adjustments(
        self,
        task: ReflectionTask,
        outcome_label: str,
    ) -> list[str]:
        """Propose bounded future process adjustments."""
        adjustments = [
            "Capture the realized outcome and lessons as a reusable postmortem memory before future similar setups are reviewed.",
            "Require future reflections to compare the final recommendation against both supporting and disconfirming historical cases.",
        ]
        if outcome_label == "failed":
            adjustments.append("Lower future confidence on this pattern until a revised checklist addresses the missed risks.")
        elif outcome_label == "worked":
            adjustments.append("Keep future reuse measured instead of upgrading the pattern into a blanket rule after one success.")
        else:
            adjustments.append("Treat future uses of this pattern as conditional until more outcome evidence accumulates.")
        return adjustments

    def _fallback_confidence_change(self, outcome_label: str) -> str:
        """Convert the realized outcome into a bounded confidence-change label."""
        if outcome_label == "worked":
            return "increase"
        if outcome_label == "failed":
            return "decrease"
        return "keep"

    def _fallback_reference_cases(
        self,
        reflection_context: dict[str, Any],
    ) -> list[dict[str, str]]:
        """Convert retrieved documents into compact reflection reference cases."""
        reference_cases: list[dict[str, str]] = []
        for document in reflection_context.get("documents", [])[:3]:
            metadata = dict(document.get("metadata", {}))
            reference_cases.append(
                {
                    "title": str(document.get("title", "")).strip() or "Untitled decision case",
                    "memory_type": str(metadata.get("memory_type", "decision_case")).strip()
                    or "decision_case",
                    "fit": str(document.get("fit") or metadata.get("fit") or "medium").strip().lower()
                    or "medium",
                    "why_relevant": "; ".join(document.get("match_reasons", [])[:2])
                    or str(metadata.get("subject", "")).strip()
                    or "Retrieved as a potentially relevant reflection case.",
                }
            )
        return reference_cases

    def _normalize_string_list(self, value: Any, *, fallback: list[str]) -> list[str]:
        """Normalize a model value into a non-empty list of strings."""
        if isinstance(value, list):
            normalized = [str(item).strip() for item in value if str(item).strip()]
            return normalized or fallback
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        return fallback

    def _normalize_reference_cases(
        self,
        value: Any,
        *,
        fallback: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        """Normalize model output into the shared reference-case schema."""
        if not isinstance(value, list):
            return fallback

        normalized_cases: list[dict[str, str]] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", "")).strip()
            if not title:
                continue
            memory_type = str(item.get("memory_type", "decision_case")).strip() or "decision_case"
            fit = str(item.get("fit", "medium")).strip().lower() or "medium"
            if fit not in {"high", "medium", "low"}:
                fit = "medium"
            why_relevant = (
                str(item.get("why_relevant", "")).strip()
                or "Returned by the reflection retrieval layer."
            )
            normalized_cases.append(
                {
                    "title": title,
                    "memory_type": memory_type,
                    "fit": fit,
                    "why_relevant": why_relevant,
                }
            )
        return normalized_cases or fallback

    def _is_valid_candidate_memory(self, value: Any) -> bool:
        """Return whether a model-supplied candidate memory matches the memory schema."""
        if not isinstance(value, dict):
            return False
        validation = validate_decision_memory_record(value, allowed_datasets=("dynamic",))
        return bool(validation["is_valid"])

    def _join_text_fields(self, *values: str | None) -> str:
        """Join optional free-form notes into one normalized string."""
        return " ".join(str(value).strip() for value in values if str(value or "").strip())

    def _extract_float(self, value: Any) -> float | None:
        """Parse float-like values from post-trade metrics."""
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
