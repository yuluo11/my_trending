"""Schema helpers for reflection outputs and candidate postmortem records."""

from __future__ import annotations

from typing import Any

from ..decision.memory import validate_decision_memory_record

ALLOWED_CONFIDENCE_CHANGES = {"increase", "keep", "decrease"}
ALLOWED_OUTCOME_LABELS = {"worked", "failed", "mixed", "unknown"}
PERCENT_METRIC_FIELDS = (
    "realized_pnl_pct",
    "pnl_pct",
    "benchmark_relative_return_pct",
    "benchmark_relative_return",
    "max_drawdown_pct",
    "holding_return_pct",
    "position_size_pct",
)
NUMERIC_METRIC_FIELDS = ("holding_period_days",)


def normalize_confidence_change(value: Any, *, default: str = "keep") -> str:
    """Normalize confidence-change labels into the supported set."""
    normalized = str(value or default).strip().lower()
    if normalized not in ALLOWED_CONFIDENCE_CHANGES:
        return default
    return normalized


def normalize_execution_summary(execution_summary: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize execution summary fields into a stable post-trade shape."""
    if not isinstance(execution_summary, dict):
        return {}

    normalized = {
        "entry_date": _normalize_text(execution_summary.get("entry_date")),
        "exit_date": _normalize_text(execution_summary.get("exit_date")),
        "entry_price": _extract_float(execution_summary.get("entry_price")),
        "exit_price": _extract_float(execution_summary.get("exit_price")),
        "holding_period_days": _extract_float(execution_summary.get("holding_period_days")),
        "position_size_pct": _extract_float(execution_summary.get("position_size_pct")),
        "summary": _normalize_text(execution_summary.get("summary")),
    }
    return {key: value for key, value in normalized.items() if value not in (None, "")}


def normalize_outcome_metrics(outcome_metrics: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize outcome metrics into a consistent quant-review schema."""
    if not isinstance(outcome_metrics, dict):
        return {}

    normalized: dict[str, Any] = {
        "outcome_label": _normalize_outcome_label(outcome_metrics.get("outcome_label")),
        "performance_assessment": _normalize_text(outcome_metrics.get("performance_assessment")),
        "summary": _normalize_text(outcome_metrics.get("summary")),
        "result": _normalize_text(outcome_metrics.get("result")),
        "notes": _normalize_text(outcome_metrics.get("notes")),
    }
    for field_name in PERCENT_METRIC_FIELDS + NUMERIC_METRIC_FIELDS:
        value = _extract_float(outcome_metrics.get(field_name))
        if value is not None:
            normalized[field_name] = value

    if "realized_pnl_pct" not in normalized and "pnl_pct" in normalized:
        normalized["realized_pnl_pct"] = normalized["pnl_pct"]
    if (
        "benchmark_relative_return_pct" not in normalized
        and "benchmark_relative_return" in normalized
    ):
        normalized["benchmark_relative_return_pct"] = normalized["benchmark_relative_return"]

    return {key: value for key, value in normalized.items() if value not in (None, "")}


def normalize_exit_context(exit_context: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize exit-related fields into a stable review shape."""
    if not isinstance(exit_context, dict):
        return {}

    normalized = {
        "exit_date": _normalize_text(exit_context.get("exit_date")),
        "exit_reason": _normalize_text(exit_context.get("exit_reason")),
        "exit_trigger": _normalize_text(exit_context.get("exit_trigger")),
        "status": _normalize_text(exit_context.get("status")),
        "summary": _normalize_text(exit_context.get("summary")),
        "notes": _normalize_text(exit_context.get("notes")),
    }
    return {key: value for key, value in normalized.items() if value not in (None, "")}


def infer_outcome_label(
    realized_outcome: dict[str, Any] | None,
    *,
    outcome_metrics: dict[str, Any] | None = None,
    exit_context: dict[str, Any] | None = None,
    post_trade_notes: str | None = None,
    feedback_notes: str | None = None,
) -> str:
    """Infer a normalized outcome label from post-trade review inputs."""
    if isinstance(realized_outcome, dict):
        explicit = str(realized_outcome.get("outcome_label", "")).strip().lower()
        if explicit in ALLOWED_OUTCOME_LABELS:
            return explicit

        status_text = " ".join(
            str(realized_outcome.get(field_name, "")).strip()
            for field_name in ("status", "summary", "result", "notes")
        ).lower()
        inferred = _infer_outcome_label_from_text(status_text)
        if inferred != "unknown":
            return inferred

    if isinstance(outcome_metrics, dict):
        explicit = str(outcome_metrics.get("outcome_label", "")).strip().lower()
        if explicit in ALLOWED_OUTCOME_LABELS:
            return explicit

        pnl_like = _extract_float(
            outcome_metrics.get("realized_pnl_pct", outcome_metrics.get("pnl_pct"))
        )
        if pnl_like is not None:
            if pnl_like > 0:
                return "worked"
            if pnl_like < 0:
                return "failed"

        relative_like = _extract_float(
            outcome_metrics.get(
                "benchmark_relative_return_pct",
                outcome_metrics.get("benchmark_relative_return"),
            )
        )
        if relative_like is not None:
            if relative_like > 0:
                return "worked"
            if relative_like < 0:
                return "failed"

        metrics_text = " ".join(
            str(outcome_metrics.get(field_name, "")).strip()
            for field_name in (
                "summary",
                "result",
                "notes",
                "exit_reason",
                "performance_assessment",
            )
        ).lower()
        inferred = _infer_outcome_label_from_text(metrics_text)
        if inferred != "unknown":
            return inferred

    if isinstance(exit_context, dict):
        exit_text = " ".join(
            str(exit_context.get(field_name, "")).strip()
            for field_name in ("exit_reason", "summary", "notes", "status")
        ).lower()
        inferred = _infer_outcome_label_from_text(exit_text)
        if inferred != "unknown":
            return inferred

    post_trade_text = str(post_trade_notes or "").strip().lower()
    inferred = _infer_outcome_label_from_text(post_trade_text)
    if inferred != "unknown":
        return inferred

    feedback_text = str(feedback_notes or "").strip().lower()
    inferred = _infer_outcome_label_from_text(feedback_text)
    if inferred != "unknown":
        return inferred
    return "unknown"


def build_candidate_postmortem_record(
    *,
    subject: str,
    symbol: str | None,
    trade_date: str | None,
    recommendation: str,
    confidence: str,
    outcome_label: str,
    reflection_summary: str,
    what_worked: list[str],
    what_failed_or_underweighted: list[str],
    lessons: list[str],
    future_adjustments: list[str],
    execution_summary: dict[str, Any] | None = None,
    outcome_metrics: dict[str, Any] | None = None,
    exit_context: dict[str, Any] | None = None,
    post_trade_notes: str | None = None,
    dataset: str = "dynamic",
) -> dict[str, Any]:
    """Build a candidate decision-memory postmortem record from reflection output."""
    normalized_outcome = outcome_label if outcome_label in ALLOWED_OUTCOME_LABELS else "unknown"
    title_subject = subject.strip() or "Unspecified decision subject"
    title = title_subject
    if symbol:
        title = f"{symbol.strip().upper()} {title_subject}"
    title = f"{title} Postmortem"

    text_sections = [
        f"Reflection summary: {reflection_summary.strip()}",
        _render_mapping_section("Execution summary", execution_summary),
        _render_mapping_section("Outcome metrics", outcome_metrics),
        _render_mapping_section("Exit context", exit_context),
        _render_text_section("Post-trade notes", post_trade_notes),
        _render_section("What worked", what_worked),
        _render_section("What failed or was underweighted", what_failed_or_underweighted),
        _render_section("Reusable lessons", lessons),
        _render_section("Future adjustments", future_adjustments),
    ]
    text = "\n\n".join(section for section in text_sections if section).strip()

    record = {
        "text": text,
        "metadata": {
            "source": "reflection_postmortem",
            "source_type": "internal",
            "title": title,
            "created_at": trade_date or "",
            "updated_at": trade_date or "",
            "category": "decision_memory",
            "memory_type": "decision_postmortem",
            "tags": _normalize_tags([symbol, "postmortem", normalized_outcome]),
            "symbol": (symbol or "").strip().upper(),
            "subject": title_subject,
            "topic": "decision-memory",
            "recommendation": str(recommendation or "keep_watch").strip().lower() or "keep_watch",
            "confidence": str(confidence or "medium").strip().lower() or "medium",
            "market_regime": "mixed",
            "analyst_alignment": "mixed",
            "signal_tags": [],
            "risk_tags": [],
            "timing_tags": [],
            "portfolio_state_tags": [],
            "outcome_label": normalized_outcome,
            "quality_score": 0.7,
            "dataset": dataset,
        },
    }
    return record


def validate_reflection_result(result: dict[str, Any] | None) -> dict[str, Any]:
    """Validate the minimal shape of a reflection output."""
    errors: list[str] = []
    normalized_result = dict(result or {})

    if not isinstance(result, dict):
        return {
            "is_valid": False,
            "errors": ["reflection result must be a JSON object"],
            "normalized_result": {},
        }

    summary = str(result.get("reflection_summary", "")).strip()
    if not summary:
        errors.append("reflection_summary must be a non-empty string")
    normalized_result["reflection_summary"] = summary

    for field_name in (
        "what_worked",
        "what_failed_or_underweighted",
        "lessons",
        "future_adjustments",
        "reference_cases",
    ):
        if not isinstance(result.get(field_name), list):
            errors.append(f"{field_name} must be a list")

    normalized_result["confidence_change"] = normalize_confidence_change(
        result.get("confidence_change"),
        default="keep",
    )

    candidate_memory = result.get("candidate_memory")
    if candidate_memory is not None:
        validation = validate_decision_memory_record(candidate_memory, allowed_datasets=("dynamic",))
        if not validation["is_valid"]:
            errors.append("candidate_memory must satisfy the decision-memory schema")

    return {
        "is_valid": not errors,
        "errors": errors,
        "normalized_result": normalized_result,
    }


def validate_post_trade_review(
    *,
    execution_summary: dict[str, Any] | None = None,
    outcome_metrics: dict[str, Any] | None = None,
    exit_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate normalized post-trade review fields."""
    errors: list[str] = []
    warnings: list[str] = []

    normalized_execution = normalize_execution_summary(execution_summary)
    normalized_metrics = normalize_outcome_metrics(outcome_metrics)
    normalized_exit = normalize_exit_context(exit_context)

    if (
        "entry_date" in normalized_execution
        and "exit_date" in normalized_execution
        and normalized_execution["entry_date"] > normalized_execution["exit_date"]
    ):
        errors.append("execution_summary.entry_date must be on or before exit_date")

    holding_period_days = normalized_execution.get("holding_period_days")
    if isinstance(holding_period_days, (int, float)) and holding_period_days < 0:
        errors.append("execution_summary.holding_period_days must be non-negative")

    max_drawdown_pct = normalized_metrics.get("max_drawdown_pct")
    if isinstance(max_drawdown_pct, (int, float)) and max_drawdown_pct > 0:
        warnings.append("outcome_metrics.max_drawdown_pct is usually zero or negative")

    if "exit_reason" not in normalized_exit and normalized_execution.get("exit_date"):
        warnings.append("exit_context.exit_reason is recommended when an exit_date is provided")

    return {
        "is_valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "execution_summary": normalized_execution,
        "outcome_metrics": normalized_metrics,
        "exit_context": normalized_exit,
    }


def assess_post_trade_review_completeness(
    *,
    execution_summary: dict[str, Any] | None = None,
    outcome_metrics: dict[str, Any] | None = None,
    exit_context: dict[str, Any] | None = None,
    realized_outcome: dict[str, Any] | None = None,
    post_trade_notes: str | None = None,
    feedback_notes: str | None = None,
) -> dict[str, Any]:
    """Assess whether the post-trade review is complete enough for reuse."""
    normalized_execution = normalize_execution_summary(execution_summary)
    normalized_metrics = normalize_outcome_metrics(outcome_metrics)
    normalized_exit = normalize_exit_context(exit_context)
    outcome_label = infer_outcome_label(
        realized_outcome,
        outcome_metrics=normalized_metrics,
        exit_context=normalized_exit,
        post_trade_notes=post_trade_notes,
        feedback_notes=feedback_notes,
    )
    combined_notes = _normalize_text(post_trade_notes or feedback_notes)

    present_inputs: list[str] = []
    missing_inputs: list[str] = []

    for field_name, has_value in (
        ("execution_summary", bool(normalized_execution)),
        ("outcome_metrics", bool(normalized_metrics)),
        ("exit_context", bool(normalized_exit)),
        ("post_trade_notes", bool(combined_notes)),
    ):
        if has_value:
            present_inputs.append(field_name)
        else:
            missing_inputs.append(field_name)

    if outcome_label == "unknown":
        missing_inputs.append("known_outcome_label")

    completeness_score = len(present_inputs) / 4.0
    if outcome_label != "unknown" and completeness_score >= 0.75:
        status = "complete"
    elif completeness_score >= 0.5:
        status = "partial"
    else:
        status = "insufficient"

    return {
        "status": status,
        "outcome_label": outcome_label,
        "completeness_score": round(completeness_score, 3),
        "present_inputs": present_inputs,
        "missing_inputs": missing_inputs,
    }


def assess_memory_persistence_candidate(
    *,
    reflection_result: dict[str, Any] | None,
    outcome_label: str,
    post_trade_validation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assess whether a reflection result is strong enough to persist as memory."""
    validation = validate_reflection_result(reflection_result)
    blocking_issues: list[str] = []
    supporting_reasons: list[str] = []

    if outcome_label == "unknown":
        blocking_issues.append("outcome_label is still unknown")
    else:
        supporting_reasons.append(f"outcome_label is {outcome_label}")

    if not validation["is_valid"]:
        blocking_issues.extend(validation["errors"])
    else:
        supporting_reasons.append("reflection output passed schema validation")

    result = dict(reflection_result or {})
    if not result.get("lessons"):
        blocking_issues.append("reflection result does not include reusable lessons")
    else:
        supporting_reasons.append("reflection result includes reusable lessons")

    if not result.get("future_adjustments"):
        blocking_issues.append("reflection result does not include future adjustments")
    else:
        supporting_reasons.append("reflection result includes future adjustments")

    if isinstance(post_trade_validation, dict):
        if not post_trade_validation.get("is_valid", True):
            blocking_issues.extend(post_trade_validation.get("errors", []))
        elif post_trade_validation.get("warnings"):
            supporting_reasons.append("post-trade review is valid with non-blocking warnings")
        else:
            supporting_reasons.append("post-trade review inputs are valid")

    should_persist = len(blocking_issues) == 0
    if not should_persist:
        priority = "do_not_persist"
    elif outcome_label in {"failed", "mixed"}:
        priority = "high"
    else:
        priority = "medium"

    return {
        "should_persist": should_persist,
        "priority": priority,
        "blocking_issues": blocking_issues,
        "supporting_reasons": supporting_reasons,
    }


def _render_section(title: str, items: list[str]) -> str:
    """Render a compact bullet list section for postmortem memory text."""
    normalized_items = [str(item).strip() for item in items if str(item).strip()]
    if not normalized_items:
        return ""
    return f"{title}:\n" + "\n".join(f"- {item}" for item in normalized_items)


def _render_text_section(title: str, value: str | None) -> str:
    """Render a single text section when a value is present."""
    normalized = str(value or "").strip()
    if not normalized:
        return ""
    return f"{title}: {normalized}"


def _render_mapping_section(title: str, value: dict[str, Any] | None) -> str:
    """Render a mapping as compact bullet lines."""
    if not isinstance(value, dict) or not value:
        return ""
    lines = [
        f"- {key}: {str(item).strip()}"
        for key, item in value.items()
        if str(key).strip() and str(item).strip()
    ]
    if not lines:
        return ""
    return f"{title}:\n" + "\n".join(lines)


def _normalize_tags(values: list[Any]) -> list[str]:
    """Normalize optional values into a stable metadata tag list."""
    tags: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = str(value or "").strip().lower().replace(" ", "_")
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        tags.append(normalized)
    return tags


def _normalize_outcome_label(value: Any) -> str:
    """Normalize optional outcome labels into the supported set."""
    normalized = str(value or "").strip().lower()
    if normalized in ALLOWED_OUTCOME_LABELS:
        return normalized
    return ""


def _normalize_text(value: Any) -> str:
    """Normalize optional text-like values into stripped strings."""
    return str(value or "").strip()


def _infer_outcome_label_from_text(text: str) -> str:
    """Infer a coarse outcome label from text."""
    if not text:
        return "unknown"
    if any(
        keyword in text
        for keyword in ("worked", "successful", "correct", "played out", "profit", "profitable")
    ):
        return "worked"
    if any(
        keyword in text
        for keyword in (
            "failed",
            "wrong",
            "loss",
            "losing",
            "missed",
            "underperformed",
            "broke",
            "stopped out",
        )
    ):
        return "failed"
    if any(keyword in text for keyword in ("mixed", "partial", "partly", "unclear")):
        return "mixed"
    return "unknown"


def _extract_float(value: Any) -> float | None:
    """Parse a float-like value when present."""
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
