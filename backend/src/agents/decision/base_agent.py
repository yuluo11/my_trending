"""Agent primitives for advisory-style decision synthesis."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, TypedDict

from ...knowledge.repository import DatasetName
from ...llm.client import LLMClient, LLMRunnable, ensure_llm_client
from ...services.decision.memory import DecisionKnowledgeService

ALLOWED_RECOMMENDATIONS = {
    "consider_buy",
    "consider_reduce",
    "hold",
    "keep_watch",
    "no_trade",
}
ALLOWED_CONFIDENCE = {"low", "medium", "high"}


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    """Return de-duplicated strings while keeping the first-seen ordering."""
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        normalized = str(value).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


@dataclass(slots=True)
class DecisionTask:
    """Structured input consumed by the decision advisory agent."""

    subject: str
    symbol: str | None = None
    trade_date: str | None = None
    extra_context: str | None = None
    overall_summary: str = ""
    overall_confidence: str = "low"
    key_signals: list[str] = field(default_factory=list)
    portfolio_risks: list[str] = field(default_factory=list)
    cross_analyst_observations: list[str] = field(default_factory=list)
    analyst_results: list[dict[str, Any]] = field(default_factory=list)
    analyst_sequence: list[str] = field(default_factory=list)
    datasets: tuple[DatasetName, ...] | None = None
    metadata_filter: dict[str, Any] | None = None
    max_documents: int | None = None
    messages: list[Any] = field(default_factory=list)

    @classmethod
    def from_analyst_payload(
        cls,
        analyst_payload: dict[str, Any],
        *,
        datasets: tuple[DatasetName, ...] | None = None,
        metadata_filter: dict[str, Any] | None = None,
        max_documents: int | None = None,
        messages: list[Any] | None = None,
    ) -> "DecisionTask":
        """Build a decision task directly from analyst orchestrator output."""
        return cls(
            subject=str(analyst_payload.get("subject", "")).strip(),
            symbol=analyst_payload.get("symbol"),
            trade_date=analyst_payload.get("trade_date"),
            extra_context=analyst_payload.get("extra_context"),
            overall_summary=str(analyst_payload.get("overall_summary", "")).strip(),
            overall_confidence=str(analyst_payload.get("overall_confidence", "low")).strip(),
            key_signals=[
                str(item).strip()
                for item in analyst_payload.get("key_signals", [])
                if str(item).strip()
            ],
            portfolio_risks=[
                str(item).strip()
                for item in analyst_payload.get("portfolio_risks", [])
                if str(item).strip()
            ],
            cross_analyst_observations=[
                str(item).strip()
                for item in analyst_payload.get("cross_analyst_observations", [])
                if str(item).strip()
            ],
            analyst_results=list(analyst_payload.get("analyst_results", [])),
            analyst_sequence=[
                str(item).strip()
                for item in analyst_payload.get("analyst_sequence", [])
                if str(item).strip()
            ],
            datasets=datasets,
            metadata_filter=metadata_filter,
            max_documents=max_documents,
            messages=list(messages or analyst_payload.get("messages", [])),
        )


class DecisionRuntimeState(TypedDict, total=False):
    """State shape for future decision-oriented graph composition."""

    subject: str
    symbol: str | None
    trade_date: str | None
    extra_context: str | None
    overall_summary: str
    overall_confidence: str
    key_signals: list[str]
    portfolio_risks: list[str]
    cross_analyst_observations: list[str]
    analyst_results: list[dict[str, Any]]
    analyst_sequence: list[str]
    datasets: tuple[DatasetName, ...] | list[DatasetName] | None
    metadata_filter: dict[str, Any] | None
    max_documents: int | None
    messages: list[Any]
    decision_output: dict[str, Any]


class PromptProvider(Protocol):
    """Prompt source used by decision agents."""

    def get_shared_prompt(self) -> str:
        """Return the shared prompt frame used by decision agents."""

    def get_analyst_prompt(self, analyst_name: str) -> str:
        """Return the role-specific prompt body."""


class FilePromptProvider:
    """Load the shared frame and role-specific prompts from disk."""

    def __init__(
        self,
        prompts_dir: str | Path,
        *,
        shared_prompt_name: str = "base_prompt.txt",
        shared_dir_name: str = "shared",
        roles_dir_name: str = "roles",
        suffix: str = ".txt",
    ) -> None:
        self.prompts_dir = Path(prompts_dir)
        self.shared_prompt_name = shared_prompt_name
        self.shared_dir_name = shared_dir_name
        self.roles_dir_name = roles_dir_name
        self.suffix = suffix

    def get_shared_prompt(self) -> str:
        """Return the shared prompt frame from disk."""
        return self._read_prompt(self.prompts_dir / self.shared_dir_name / self.shared_prompt_name)

    def get_analyst_prompt(self, analyst_name: str) -> str:
        """Return the agent-specific prompt body from disk."""
        return self._read_prompt(
            self.prompts_dir / self.roles_dir_name / f"{analyst_name}{self.suffix}"
        )

    def _read_prompt(self, prompt_path: Path) -> str:
        """Read a prompt file if it exists, otherwise return an empty prompt."""
        if not prompt_path.exists():
            return ""
        return prompt_path.read_text(encoding="utf-8").strip()


class BaseDecisionAgent:
    """Shared implementation for advisory decision agents."""

    agent_name = "decision_advisory"

    def __init__(
        self,
        *,
        agent_name: str | None = None,
        knowledge_service: DecisionKnowledgeService,
        prompt_provider: PromptProvider | None = None,
        llm_client: LLMClient | None = None,
        llm: LLMRunnable | None = None,
    ) -> None:
        if agent_name:
            self.agent_name = agent_name
        self.knowledge_service = knowledge_service
        self.prompt_provider = prompt_provider
        self.llm_client = ensure_llm_client(llm_client=llm_client, llm=llm)

    def retrieve_decision_context(self, task: DecisionTask) -> dict[str, Any]:
        """Retrieve decision-memory context relevant to the current advisory task."""
        return self.knowledge_service.analyze(
            task,
            datasets=task.datasets,
            metadata_filter=task.metadata_filter,
            k=task.max_documents,
        )

    def build_prompt_context(
        self,
        task: DecisionTask,
        decision_context: dict[str, Any],
    ) -> dict[str, Any]:
        """Build the runtime context used while rendering the advisory prompt."""
        return {
            "subject": task.subject,
            "symbol": task.symbol,
            "trade_date": task.trade_date,
            "extra_context": task.extra_context,
            "overall_confidence": task.overall_confidence,
            "analyst_count": len(task.analyst_results),
            "decision_context": decision_context,
        }

    def render_prompt(self, prompt_context: dict[str, Any]) -> str:
        """Render the decision prompt from shared and role-specific prompt assets."""
        if self.prompt_provider is None:
            return ""

        shared_prompt = self.prompt_provider.get_shared_prompt()
        role_prompt = self.prompt_provider.get_analyst_prompt(self.agent_name)
        context_lines = [
            f"subject: {prompt_context.get('subject', '')}",
            f"symbol: {prompt_context.get('symbol') or 'N/A'}",
            f"trade_date: {prompt_context.get('trade_date') or 'N/A'}",
            f"extra_context: {prompt_context.get('extra_context') or 'N/A'}",
            f"overall_confidence: {prompt_context.get('overall_confidence') or 'low'}",
            f"analyst_count: {prompt_context.get('analyst_count', 0)}",
            (
                "decision_memory_documents: "
                f"{prompt_context['decision_context'].get('document_count', 0)}"
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
        task: DecisionTask,
        *,
        decision_context: dict[str, Any],
    ) -> dict[str, Any]:
        """Build the structured payload sent into the configured llm_client."""
        return {
            "task": {
                "subject": task.subject,
                "symbol": task.symbol,
                "trade_date": task.trade_date,
                "extra_context": task.extra_context,
            },
            "analysis": {
                "overall_summary": task.overall_summary,
                "overall_confidence": task.overall_confidence,
                "key_signals": task.key_signals,
                "portfolio_risks": task.portfolio_risks,
                "cross_analyst_observations": task.cross_analyst_observations,
                "analyst_sequence": task.analyst_sequence,
                "analyst_results": task.analyst_results,
            },
            "decision_memory": {
                "query": decision_context.get("query", ""),
                "document_count": decision_context.get("document_count", 0),
                "documents": decision_context.get("documents", []),
                "evidence": decision_context.get("evidence", []),
            },
            "instructions": (
                "Return a JSON object with keys: decision_summary, recommendation, "
                "aggregated_risks, rationale, confidence, reference_cases, and "
                "case_fit_assessment. This is advisory only and must not imply trade execution."
            ),
        }

    def invoke(self, task: DecisionTask) -> dict[str, Any]:
        """Run the decision advisory flow end to end."""
        decision_context = self.retrieve_decision_context(task)
        prompt_context = self.build_prompt_context(task, decision_context)
        prompt = self.render_prompt(prompt_context)
        return self.synthesize(task, decision_context, prompt)

    def synthesize(
        self,
        task: DecisionTask,
        decision_context: dict[str, Any],
        prompt: str,
    ) -> dict[str, Any]:
        """Create the final decision payload using the model or fallback logic."""
        if self.llm_client is not None:
            return self._synthesize_with_llm(task, decision_context, prompt)
        return self._synthesize_fallback(task, decision_context, prompt)

    def _synthesize_with_llm(
        self,
        task: DecisionTask,
        decision_context: dict[str, Any],
        prompt: str,
    ) -> dict[str, Any]:
        """Invoke the llm_client and normalize the advisory output."""
        llm_payload = self.build_llm_payload(task, decision_context=decision_context)
        parsed_response = self.llm_client.invoke_json(
            prompt,
            payload=llm_payload,
            schema=self.decision_output_schema(),
        )
        return self.normalize_llm_result(
            parsed_response,
            task=task,
            decision_context=decision_context,
            prompt=prompt,
        )

    def _synthesize_fallback(
        self,
        task: DecisionTask,
        decision_context: dict[str, Any],
        prompt: str,
    ) -> dict[str, Any]:
        """Produce a cautious deterministic advisory result."""
        recommendation = self._fallback_recommendation(task)
        reference_cases = self._fallback_reference_cases(decision_context)
        aggregated_risks = _dedupe_preserve_order(task.portfolio_risks)
        if not aggregated_risks:
            aggregated_risks = ["Current analyst evidence does not yet support a stronger action."]

        decision_summary = (
            f"Current analyst evidence for {task.subject} supports a {recommendation} stance."
        )
        rationale_parts = [
            f"The decision layer reviewed {len(task.analyst_results)} analyst perspective(s).",
            f"Overall analyst confidence is {self._normalize_confidence(task.overall_confidence)}.",
        ]
        if task.key_signals:
            rationale_parts.append(f"Leading signals: {', '.join(task.key_signals[:3])}.")
        if reference_cases:
            rationale_parts.append(
                f"Referenced {len(reference_cases)} decision-memory case(s) as supporting context."
            )
        rationale = " ".join(rationale_parts)

        if reference_cases:
            case_fit_assessment = (
                "Reference cases were used as advisory context only and should not be copied directly."
            )
        else:
            case_fit_assessment = (
                "No matching decision-memory cases were retrieved, so the recommendation relies on "
                "current analyst synthesis only."
            )

        return {
            "subject": task.subject,
            "symbol": task.symbol,
            "trade_date": task.trade_date,
            "decision_summary": decision_summary,
            "recommendation": recommendation,
            "aggregated_risks": aggregated_risks,
            "rationale": rationale,
            "confidence": self._normalize_confidence(task.overall_confidence),
            "reference_cases": reference_cases,
            "case_fit_assessment": case_fit_assessment,
            "prompt": prompt,
            "decision_context": decision_context,
        }

    def normalize_llm_result(
        self,
        llm_result: dict[str, Any],
        *,
        task: DecisionTask,
        decision_context: dict[str, Any],
        prompt: str,
    ) -> dict[str, Any]:
        """Normalize LLM output into the shared decision payload shape."""
        fallback_result = self._synthesize_fallback(task, decision_context, prompt)
        recommendation = str(llm_result.get("recommendation", fallback_result["recommendation"]))
        recommendation = recommendation.strip().lower()
        if recommendation not in ALLOWED_RECOMMENDATIONS:
            recommendation = fallback_result["recommendation"]

        return {
            "subject": task.subject,
            "symbol": task.symbol,
            "trade_date": task.trade_date,
            "decision_summary": str(
                llm_result.get("decision_summary", fallback_result["decision_summary"])
            ).strip()
            or fallback_result["decision_summary"],
            "recommendation": recommendation,
            "aggregated_risks": self._normalize_string_list(
                llm_result.get("aggregated_risks"),
                fallback=fallback_result["aggregated_risks"],
            ),
            "rationale": str(llm_result.get("rationale", fallback_result["rationale"])).strip()
            or fallback_result["rationale"],
            "confidence": self._normalize_confidence(
                llm_result.get("confidence", fallback_result["confidence"])
            ),
            "reference_cases": self._normalize_reference_cases(
                llm_result.get("reference_cases"),
                fallback=fallback_result["reference_cases"],
            ),
            "case_fit_assessment": str(
                llm_result.get("case_fit_assessment", fallback_result["case_fit_assessment"])
            ).strip()
            or fallback_result["case_fit_assessment"],
            "prompt": prompt,
            "decision_context": decision_context,
            "raw_model_output": llm_result,
        }

    def decision_output_schema(self) -> dict[str, Any]:
        """Return the target structured schema for decision advisory outputs."""
        return {
            "decision_summary": "string",
            "recommendation": "consider_buy|consider_reduce|hold|keep_watch|no_trade",
            "aggregated_risks": ["string"],
            "rationale": "string",
            "confidence": "low|medium|high",
            "reference_cases": [
                {
                    "title": "string",
                    "memory_type": "decision_case|decision_postmortem|external_reference_decision",
                    "fit": "high|medium|low",
                    "why_relevant": "string",
                }
            ],
            "case_fit_assessment": "string",
        }

    def build_agent_message(self, result: dict[str, Any]) -> Any:
        """Create a graph-friendly message from the advisory decision payload."""
        content = result.get("decision_summary", "")
        try:
            from langchain_core.messages import AIMessage
        except ModuleNotFoundError:
            return {
                "role": "assistant",
                "name": self.agent_name,
                "content": content,
            }
        return AIMessage(content=content, name=self.agent_name)

    def _fallback_recommendation(self, task: DecisionTask) -> str:
        """Generate a conservative recommendation when no model output is available."""
        if not task.key_signals:
            return "no_trade"
        if self._has_material_conflict(task.cross_analyst_observations):
            return "keep_watch"
        if self._normalize_confidence(task.overall_confidence) == "high":
            return "hold"
        return "keep_watch"

    def _fallback_reference_cases(
        self,
        decision_context: dict[str, Any],
    ) -> list[dict[str, str]]:
        """Convert retrieved decision-memory documents into compact reference case records."""
        reference_cases: list[dict[str, str]] = []
        for document in decision_context.get("documents", [])[:3]:
            metadata = dict(document.get("metadata", {}))
            reference_cases.append(
                {
                    "title": str(document.get("title", "")).strip() or "Untitled decision case",
                    "memory_type": str(metadata.get("memory_type", "decision_case")).strip()
                    or "decision_case",
                    "fit": "medium",
                    "why_relevant": str(metadata.get("subject", "")).strip()
                    or "Retrieved as a potentially similar decision-memory case.",
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
        """Normalize model output into the reference-case schema."""
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
                or "Returned by the decision-memory retrieval layer."
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

    def _normalize_confidence(self, value: Any) -> str:
        """Normalize confidence labels into the supported set."""
        confidence = str(value or "low").strip().lower()
        if confidence not in ALLOWED_CONFIDENCE:
            return "low"
        return confidence

    def _has_material_conflict(self, observations: list[str]) -> bool:
        """Detect obvious cross-analyst disagreement from observation strings."""
        conflict_markers = ("disagree", "conflict", "mixed", "diverge", "uncertain")
        for observation in observations:
            normalized = observation.lower()
            if any(marker in normalized for marker in conflict_markers):
                return True
        return False
