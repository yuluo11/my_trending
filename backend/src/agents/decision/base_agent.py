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
    portfolio_context: dict[str, Any] | None = None
    datasets: tuple[DatasetName, ...] | None = None
    metadata_filter: dict[str, Any] | None = None
    max_documents: int | None = None
    messages: list[Any] = field(default_factory=list)

    @classmethod
    def from_analyst_payload(
        cls,
        analyst_payload: dict[str, Any],
        *,
        portfolio_context: dict[str, Any] | None = None,
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
            portfolio_context=portfolio_context
            if portfolio_context is not None
            else analyst_payload.get("portfolio_context"),
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
    portfolio_context: dict[str, Any] | None
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
            "portfolio_context_used": bool(task.portfolio_context),
            "portfolio_context_summary": self.summarize_portfolio_context(task.portfolio_context),
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
            f"portfolio_context_used: {prompt_context.get('portfolio_context_used', False)}",
            (
                "portfolio_context: "
                f"{prompt_context.get('portfolio_context_summary') or 'N/A'}"
            ),
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
                "portfolio_context": task.portfolio_context or {},
            },
            "analysis": self.build_analysis_payload(task),
            "decision_memory": self.build_decision_memory_payload(decision_context),
            "instructions": (
                "Return a JSON object with keys: decision_summary, recommendation, "
                "portfolio_context_used, portfolio_context_summary, position_impact, "
                "timing_decision, action_conditions, no_action_reasons, aggregated_risks, "
                "rationale, confidence, reference_cases, and case_fit_assessment. This is "
                "advisory only and must not imply trade execution. Use portfolio context when it "
                "is provided, and use scenario fit, not just raw similarity, when discussing "
                "reference cases."
            ),
        }

    def build_analysis_payload(self, task: DecisionTask) -> dict[str, Any]:
        """Build the analyst-synthesis block consumed by the decision model."""
        return {
            "overall_summary": task.overall_summary,
            "overall_confidence": task.overall_confidence,
            "key_signals": task.key_signals,
            "portfolio_risks": task.portfolio_risks,
            "cross_analyst_observations": task.cross_analyst_observations,
            "analyst_sequence": task.analyst_sequence,
            "analyst_results": task.analyst_results,
        }

    def build_decision_memory_payload(
        self,
        decision_context: dict[str, Any],
    ) -> dict[str, Any]:
        """Build the retrieved decision-memory block consumed by the decision model."""
        return {
            "query": decision_context.get("query", ""),
            "scenario_profile": decision_context.get("scenario_profile", {}),
            "document_count": decision_context.get("document_count", 0),
            "validation_summary": decision_context.get("validation_summary", {}),
            "documents": decision_context.get("documents", []),
            "evidence": decision_context.get("evidence", []),
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
        position_impact = self._fallback_position_impact(task, recommendation)
        timing_decision = self._fallback_timing_decision(task, recommendation)
        action_conditions = self._fallback_action_conditions(task, recommendation)
        no_action_reasons = self._fallback_no_action_reasons(task, recommendation)

        decision_summary = (
            f"Current analyst evidence for {task.subject} supports a {recommendation} stance."
        )
        portfolio_context_summary = self.summarize_portfolio_context(task.portfolio_context)
        rationale_parts = [
            f"The decision layer reviewed {len(task.analyst_results)} analyst perspective(s).",
            f"Overall analyst confidence is {self._normalize_confidence(task.overall_confidence)}.",
        ]
        if portfolio_context_summary:
            rationale_parts.append(f"Portfolio context considered: {portfolio_context_summary}.")
        if task.key_signals:
            rationale_parts.append(f"Leading signals: {', '.join(task.key_signals[:3])}.")
        if reference_cases:
            rationale_parts.append(
                f"Referenced {len(reference_cases)} decision-memory case(s) as supporting context."
            )
        rationale = " ".join(rationale_parts)

        case_fit_assessment = self._fallback_case_fit_assessment(reference_cases)

        return {
            "subject": task.subject,
            "symbol": task.symbol,
            "trade_date": task.trade_date,
            "decision_summary": decision_summary,
            "recommendation": recommendation,
            "portfolio_context_used": bool(task.portfolio_context),
            "portfolio_context_summary": portfolio_context_summary,
            "position_impact": position_impact,
            "timing_decision": timing_decision,
            "action_conditions": action_conditions,
            "no_action_reasons": no_action_reasons,
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
            "portfolio_context_used": bool(task.portfolio_context),
            "portfolio_context_summary": str(
                llm_result.get(
                    "portfolio_context_summary",
                    fallback_result["portfolio_context_summary"],
                )
            ).strip()
            or fallback_result["portfolio_context_summary"],
            "position_impact": str(
                llm_result.get("position_impact", fallback_result["position_impact"])
            ).strip()
            or fallback_result["position_impact"],
            "timing_decision": str(
                llm_result.get("timing_decision", fallback_result["timing_decision"])
            ).strip()
            or fallback_result["timing_decision"],
            "action_conditions": self._normalize_string_list(
                llm_result.get("action_conditions"),
                fallback=fallback_result["action_conditions"],
            ),
            "no_action_reasons": self._normalize_string_list(
                llm_result.get("no_action_reasons"),
                fallback=fallback_result["no_action_reasons"],
            ),
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
            "portfolio_context_used": "boolean",
            "portfolio_context_summary": "string",
            "position_impact": "string",
            "timing_decision": "string",
            "action_conditions": ["string"],
            "no_action_reasons": ["string"],
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
        portfolio_context = task.portfolio_context or {}
        current_position = self._find_symbol_position(portfolio_context, task.symbol)
        current_weight = self._extract_position_weight(current_position)
        max_weight = self._extract_max_single_name_pct(portfolio_context)
        cash_pct = self._extract_percent(portfolio_context.get("cash_pct"))
        has_conflict = self._has_material_conflict(task.cross_analyst_observations)
        confidence = self._normalize_confidence(task.overall_confidence)

        if current_weight is not None and max_weight is not None and current_weight > max_weight:
            return "consider_reduce"
        if not task.key_signals:
            return "no_trade"
        if has_conflict:
            return "keep_watch"
        if (
            current_weight is not None
            and max_weight is not None
            and current_weight >= max_weight * 0.9
        ):
            return "hold"
        if current_weight is None and confidence == "high" and (cash_pct is None or cash_pct >= 5):
            return "consider_buy"
        if current_weight is None and cash_pct is not None and cash_pct < 5:
            return "keep_watch"
        if confidence == "high":
            return "hold"
        return "keep_watch"

    def _fallback_position_impact(
        self,
        task: DecisionTask,
        recommendation: str,
    ) -> str:
        """Describe portfolio impact using current holdings when available."""
        portfolio_context = task.portfolio_context or {}
        current_position = self._find_symbol_position(portfolio_context, task.symbol)
        current_weight = self._extract_position_weight(current_position)
        cash_pct = self._extract_percent(portfolio_context.get("cash_pct"))
        max_weight = self._extract_max_single_name_pct(portfolio_context)

        if task.symbol and current_weight is not None:
            if recommendation == "consider_reduce":
                return (
                    f"The current {task.symbol} position is about {current_weight:.1f}% and the "
                    "advisory stance favors reducing exposure rather than adding to it."
                )
            if recommendation == "hold":
                return (
                    f"The current {task.symbol} position is about {current_weight:.1f}% and the "
                    "advisory stance is to maintain that exposure for now."
                )
            return (
                f"The current {task.symbol} position is about {current_weight:.1f}% and the "
                "advisory stance does not support immediate resizing."
            )

        if task.symbol and recommendation == "consider_buy":
            if cash_pct is not None and max_weight is not None:
                return (
                    f"No existing {task.symbol} position was identified. With cash near "
                    f"{cash_pct:.1f}% and a single-name limit near {max_weight:.1f}%, the "
                    "current setup supports considering a measured new position rather than a full-size entry."
                )
            if cash_pct is not None:
                return (
                    f"No existing {task.symbol} position was identified. Cash is about "
                    f"{cash_pct:.1f}%, so the setup can be treated as a measured add candidate."
                )
            return (
                f"No existing {task.symbol} position was identified, and the current setup "
                "supports only a measured initial exposure rather than an aggressive entry."
            )

        if task.symbol and recommendation in {"keep_watch", "no_trade"}:
            base = f"No existing {task.symbol} position was identified."
            if cash_pct is not None:
                return (
                    f"{base} Cash is about {cash_pct:.1f}%, but the current setup does not yet "
                    "justify adding new exposure."
                )
            return f"{base} The current setup does not yet justify adding new exposure."

        if task.symbol and recommendation == "hold" and max_weight is not None:
            return (
                f"Any future exposure to {task.symbol} should stay mindful of the single-name "
                f"limit near {max_weight:.1f}%."
            )

        return (
            "The current advisory stance is not expected to materially change portfolio exposure "
            "until stronger confirmation appears."
        )

    def _fallback_timing_decision(
        self,
        task: DecisionTask,
        recommendation: str,
    ) -> str:
        """Provide a bounded view on timing rather than an execution instruction."""
        scenario_profile = self._scenario_profile_from_task(task)
        cash_pct = self._extract_percent((task.portfolio_context or {}).get("cash_pct"))
        if recommendation == "consider_reduce":
            return "Current conditions support reviewing exposure now rather than waiting for a stronger risk signal."
        if recommendation == "consider_buy":
            if "near_local_high" in scenario_profile.get("timing_tags", []):
                return "The setup is constructive, but because it looks extended, any new exposure should wait for confirmation or be sized in gradually."
            return "The setup is constructive enough to consider a measured entry now, provided exposure is staged rather than rushed."
        if self._has_material_conflict(task.cross_analyst_observations):
            return "Waiting for clearer analyst alignment is preferable before changing exposure."
        if "near_local_high" in scenario_profile.get("timing_tags", []):
            return "The setup looks extended, so waiting for better confirmation or a less stretched entry is preferable."
        if cash_pct is not None and cash_pct < 5:
            return "Limited cash makes immediate action less attractive, so timing should improve only if conviction strengthens meaningfully."
        if self._normalize_confidence(task.overall_confidence) == "high":
            return "The setup is actionable only in a measured way, with preference for staged decision-making over urgency."
        return "The current setup is better treated as a watchlist decision than an immediate portfolio action."

    def _fallback_action_conditions(
        self,
        task: DecisionTask,
        recommendation: str,
    ) -> list[str]:
        """List the conditions that would strengthen the advisory stance."""
        conditions = [
            "Keep the recommendation bounded to current analyst evidence and any matching decision-memory cases.",
        ]
        if task.key_signals:
            conditions.append(
                f"Act only if the leading signals remain intact: {', '.join(task.key_signals[:2])}."
            )
        cash_pct = self._extract_percent((task.portfolio_context or {}).get("cash_pct"))
        max_weight = self._extract_max_single_name_pct(task.portfolio_context or {})
        if max_weight is not None and task.symbol:
            conditions.append(
                f"Any exposure change should remain within the single-name limit near {max_weight:.1f}%."
            )
        if recommendation == "consider_buy" and cash_pct is not None:
            conditions.append(
                f"Keep enough liquidity after any entry; current cash is about {cash_pct:.1f}%."
            )
        if recommendation in {"keep_watch", "no_trade"}:
            conditions.append(
                "Look for stronger cross-analyst confirmation before upgrading the stance."
            )
        return conditions

    def _fallback_no_action_reasons(
        self,
        task: DecisionTask,
        recommendation: str,
    ) -> list[str]:
        """Explain why the current stance is still constrained."""
        reasons: list[str] = []
        if recommendation in {"keep_watch", "no_trade", "hold"}:
            reasons.append("Current evidence does not yet justify a more aggressive portfolio change.")
        if self._has_material_conflict(task.cross_analyst_observations):
            reasons.append(
                "Cross-analyst alignment is not strong enough to support a higher-conviction action."
            )
        cash_pct = self._extract_percent((task.portfolio_context or {}).get("cash_pct"))
        if recommendation != "consider_buy" and cash_pct is not None and cash_pct < 5:
            reasons.append(
                "Available cash is limited, which reduces flexibility for adding new exposure."
            )
        current_weight = self._extract_position_weight(
            self._find_symbol_position(task.portfolio_context or {}, task.symbol)
        )
        max_weight = self._extract_max_single_name_pct(task.portfolio_context or {})
        if (
            recommendation == "hold"
            and current_weight is not None
            and max_weight is not None
            and current_weight >= max_weight * 0.9
        ):
            reasons.append("Current exposure is already close to the single-name limit.")
        if task.portfolio_risks:
            reasons.append(f"Key risks remain active: {', '.join(task.portfolio_risks[:2])}.")
        if not reasons:
            reasons.append("The current stance remains advisory and conditional rather than execution-oriented.")
        return reasons

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
                    "fit": str(document.get("fit") or metadata.get("fit") or "medium").strip().lower()
                    or "medium",
                    "why_relevant": "; ".join(document.get("match_reasons", [])[:2])
                    or str(metadata.get("subject", "")).strip()
                    or "Retrieved as a potentially similar decision-memory case.",
                }
            )
        return reference_cases

    def _fallback_case_fit_assessment(
        self,
        reference_cases: list[dict[str, str]],
    ) -> str:
        """Summarize how well retrieved reference cases match the current setup."""
        if not reference_cases:
            return (
                "No matching decision-memory cases were retrieved, so the recommendation relies on "
                "current analyst synthesis only."
            )

        fits = [str(case.get("fit", "low")).strip().lower() for case in reference_cases]
        if any(fit == "high" for fit in fits):
            return (
                "At least one reference case shows high scenario fit, but it remains supporting "
                "context rather than an instruction to copy."
            )
        if any(fit == "medium" for fit in fits):
            return (
                "Reference cases show partial scenario fit and were used to frame similarities and "
                "differences, not to force a decision."
            )
        return (
            "The retrieved reference cases have weak scenario fit, so they mainly serve as a "
            "confidence check rather than a directional guide."
        )

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

    def summarize_portfolio_context(self, portfolio_context: dict[str, Any] | None) -> str:
        """Summarize portfolio inputs into a compact prompt/debug string."""
        if not isinstance(portfolio_context, dict) or not portfolio_context:
            return ""

        positions = portfolio_context.get("positions")
        position_count = len(positions) if isinstance(positions, list) else 0
        cash_pct = self._extract_percent(portfolio_context.get("cash_pct"))
        max_weight = self._extract_max_single_name_pct(portfolio_context)
        summary_parts = [f"positions={position_count}"]
        if cash_pct is not None:
            summary_parts.append(f"cash_pct={cash_pct:.1f}")
        if max_weight is not None:
            summary_parts.append(f"max_single_name_pct={max_weight:.1f}")
        return ", ".join(summary_parts)

    def _scenario_profile_from_task(self, task: DecisionTask) -> dict[str, Any]:
        """Infer lightweight timing cues directly from the task."""
        combined = " ".join(
            [task.subject, task.extra_context or "", task.overall_summary, *task.cross_analyst_observations]
        ).lower()
        timing_tags: list[str] = []
        if any(keyword in combined for keyword in ("high", "extended", "stretched", "overbought")):
            timing_tags.append("near_local_high")
        if any(keyword in combined for keyword in ("event", "earnings", "guidance", "catalyst")):
            timing_tags.append("event_window")
        return {"timing_tags": timing_tags}

    def _find_symbol_position(
        self,
        portfolio_context: dict[str, Any],
        symbol: str | None,
    ) -> dict[str, Any] | None:
        """Return the current position object for the requested symbol if present."""
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

    def _extract_position_weight(self, position: dict[str, Any] | None) -> float | None:
        """Extract a normalized percent weight from a position object."""
        if not isinstance(position, dict):
            return None
        for field_name in ("weight_pct", "weight"):
            value = self._extract_percent(position.get(field_name))
            if value is not None:
                return value
        return None

    def _extract_max_single_name_pct(self, portfolio_context: dict[str, Any]) -> float | None:
        """Extract the active single-name position limit if one is available."""
        direct_value = self._extract_percent(portfolio_context.get("max_single_name_pct"))
        if direct_value is not None:
            return direct_value
        position_limits = portfolio_context.get("position_limits")
        if isinstance(position_limits, dict):
            return self._extract_percent(position_limits.get("max_single_name_pct"))
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

    def _has_material_conflict(self, observations: list[str]) -> bool:
        """Detect obvious cross-analyst disagreement from observation strings."""
        conflict_markers = ("disagree", "conflict", "mixed", "diverge", "uncertain")
        for observation in observations:
            normalized = observation.lower()
            if any(marker in normalized for marker in conflict_markers):
                return True
        return False
