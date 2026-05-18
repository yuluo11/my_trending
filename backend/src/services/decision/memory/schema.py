"""Schema and validation helpers for standardized decision-memory records."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

ALLOWED_MEMORY_TYPES = {
    "decision_case",
    "decision_postmortem",
    "external_reference_decision",
}
ALLOWED_SOURCE_TYPES = {"internal", "external"}
ALLOWED_RECOMMENDATIONS = {
    "consider_buy",
    "consider_reduce",
    "hold",
    "keep_watch",
    "no_trade",
}
ALLOWED_CONFIDENCE = {"low", "medium", "high"}
ALLOWED_ANALYST_ALIGNMENT = {"aligned", "mixed", "conflicted"}
ALLOWED_OUTCOME_LABELS = {"worked", "failed", "mixed", "unknown"}

REQUIRED_METADATA_FIELDS = (
    "title",
    "category",
    "memory_type",
    "source_type",
    "subject",
    "recommendation",
    "confidence",
    "dataset",
)

OPTIONAL_METADATA_FIELDS = (
    "source",
    "symbol",
    "topic",
    "tags",
    "signal_tags",
    "risk_tags",
    "timing_tags",
    "portfolio_state_tags",
    "market_regime",
    "analyst_alignment",
    "outcome_label",
    "quality_score",
    "created_at",
    "updated_at",
)

LIST_METADATA_FIELDS = (
    "tags",
    "signal_tags",
    "risk_tags",
    "timing_tags",
    "portfolio_state_tags",
)


def normalize_decision_memory_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize decision-memory metadata into a stable retrieval shape."""
    normalized = dict(metadata or {})
    normalized["category"] = "decision_memory"
    normalized["memory_type"] = _normalize_choice(
        normalized.get("memory_type"),
        allowed=ALLOWED_MEMORY_TYPES,
        default="decision_case",
    )
    normalized["source_type"] = _normalize_choice(
        normalized.get("source_type"),
        allowed=ALLOWED_SOURCE_TYPES,
        default="internal",
    )
    normalized["recommendation"] = _normalize_choice(
        normalized.get("recommendation"),
        allowed=ALLOWED_RECOMMENDATIONS,
        default="keep_watch",
    )
    normalized["confidence"] = _normalize_choice(
        normalized.get("confidence"),
        allowed=ALLOWED_CONFIDENCE,
        default="medium",
    )
    normalized["analyst_alignment"] = _normalize_choice(
        normalized.get("analyst_alignment"),
        allowed=ALLOWED_ANALYST_ALIGNMENT,
        default="mixed",
    )
    normalized["outcome_label"] = _normalize_choice(
        normalized.get("outcome_label"),
        allowed=ALLOWED_OUTCOME_LABELS,
        default="unknown",
    )
    normalized["title"] = _normalize_string(
        normalized.get("title"),
        default="Untitled Decision Memory",
    )
    normalized["subject"] = _normalize_string(
        normalized.get("subject"),
        default="Unspecified decision subject",
    )
    normalized["symbol"] = _normalize_string(normalized.get("symbol"), default="")
    normalized["dataset"] = _normalize_string(normalized.get("dataset"), default="dynamic")
    normalized["source"] = _normalize_string(normalized.get("source"), default="")
    normalized["topic"] = _normalize_string(normalized.get("topic"), default="decision-memory")
    normalized["market_regime"] = _normalize_string(
        normalized.get("market_regime"),
        default="mixed",
    ).lower()
    normalized["tags"] = _normalize_string_list(normalized.get("tags"))
    normalized["signal_tags"] = _normalize_string_list(normalized.get("signal_tags"))
    normalized["risk_tags"] = _normalize_string_list(normalized.get("risk_tags"))
    normalized["timing_tags"] = _normalize_string_list(normalized.get("timing_tags"))
    normalized["portfolio_state_tags"] = _normalize_string_list(
        normalized.get("portfolio_state_tags")
    )
    normalized["created_at"] = _normalize_string(normalized.get("created_at"), default="")
    normalized["updated_at"] = _normalize_string(normalized.get("updated_at"), default="")
    normalized["quality_score"] = _normalize_quality_score(normalized.get("quality_score"))
    return normalized


def decision_memory_record_template() -> dict[str, Any]:
    """Return a canonical decision-memory record template."""
    return {
        "text": (
            "Describe the historical decision setup, the advisory recommendation that was made, "
            "the core reasoning, and the main risks or lessons that should be reusable later."
        ),
        "metadata": {
            "source": "internal_decision_log",
            "source_type": "internal",
            "title": "Descriptive Decision Memory Title",
            "created_at": "YYYY-MM-DDTHH:MM:SS+08:00",
            "updated_at": "YYYY-MM-DDTHH:MM:SS+08:00",
            "category": "decision_memory",
            "memory_type": "decision_case",
            "tags": ["theme", "setup", "broad-context"],
            "symbol": "TICKER",
            "subject": "Concise scenario label",
            "topic": "decision-memory",
            "recommendation": "keep_watch",
            "confidence": "medium",
            "market_regime": "event_driven",
            "analyst_alignment": "mixed",
            "signal_tags": ["news_catalyst", "momentum"],
            "risk_tags": ["crowded_trade", "valuation_risk"],
            "timing_tags": ["short_term", "event_window"],
            "portfolio_state_tags": ["existing_position", "near_single_name_limit"],
            "outcome_label": "worked",
            "quality_score": 0.75,
            "dataset": "dynamic",
        },
    }


def validate_decision_memory_metadata(
    metadata: dict[str, Any] | None,
    *,
    allowed_datasets: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Validate raw decision-memory metadata before it is normalized."""
    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(metadata, dict):
        return {
            "is_valid": False,
            "errors": ["metadata must be a JSON object"],
            "warnings": [],
            "normalized_metadata": normalize_decision_memory_metadata(None),
        }

    normalized_metadata = normalize_decision_memory_metadata(metadata)
    raw_metadata = dict(metadata)

    for field_name in REQUIRED_METADATA_FIELDS:
        if _is_blank(raw_metadata.get(field_name)):
            errors.append(f"missing required metadata field: {field_name}")

    if str(raw_metadata.get("category", "")).strip() != "decision_memory":
        errors.append("metadata.category must be 'decision_memory'")

    _validate_choice_field(
        raw_metadata,
        "memory_type",
        allowed=ALLOWED_MEMORY_TYPES,
        errors=errors,
    )
    _validate_choice_field(
        raw_metadata,
        "source_type",
        allowed=ALLOWED_SOURCE_TYPES,
        errors=errors,
    )
    _validate_choice_field(
        raw_metadata,
        "recommendation",
        allowed=ALLOWED_RECOMMENDATIONS,
        errors=errors,
    )
    _validate_choice_field(
        raw_metadata,
        "confidence",
        allowed=ALLOWED_CONFIDENCE,
        errors=errors,
    )
    _validate_choice_field(
        raw_metadata,
        "analyst_alignment",
        allowed=ALLOWED_ANALYST_ALIGNMENT,
        errors=errors,
        optional=True,
    )
    _validate_choice_field(
        raw_metadata,
        "outcome_label",
        allowed=ALLOWED_OUTCOME_LABELS,
        errors=errors,
        optional=True,
    )

    if allowed_datasets:
        allowed_dataset_set = {
            str(dataset).strip()
            for dataset in allowed_datasets
            if str(dataset).strip()
        }
        dataset = str(raw_metadata.get("dataset", "")).strip()
        if dataset and dataset not in allowed_dataset_set:
            allowed_labels = ", ".join(sorted(allowed_dataset_set))
            errors.append(
                f"metadata.dataset must be one of [{allowed_labels}], got '{dataset}'"
            )

    for field_name in LIST_METADATA_FIELDS:
        _validate_list_metadata_field(raw_metadata, field_name, errors=errors)

    quality_score = raw_metadata.get("quality_score")
    if quality_score not in (None, ""):
        parsed_quality = _parse_float(quality_score)
        if parsed_quality is None:
            errors.append("metadata.quality_score must be numeric when provided")
        elif not 0.0 <= parsed_quality <= 1.0:
            errors.append("metadata.quality_score must be between 0 and 1")

    memory_type = str(raw_metadata.get("memory_type", "")).strip().lower()
    source_type = str(raw_metadata.get("source_type", "")).strip().lower()

    if memory_type == "decision_case" and _is_blank(raw_metadata.get("signal_tags")):
        warnings.append("decision_case records work better with signal_tags")
    if memory_type == "decision_case" and _is_blank(raw_metadata.get("risk_tags")):
        warnings.append("decision_case records work better with risk_tags")
    if memory_type == "decision_case" and _is_blank(raw_metadata.get("portfolio_state_tags")):
        warnings.append("decision_case records work better with portfolio_state_tags")
    if memory_type == "decision_postmortem" and _is_blank(raw_metadata.get("outcome_label")):
        warnings.append("decision_postmortem records should usually provide outcome_label")
    if source_type == "external" and _is_blank(raw_metadata.get("source")):
        warnings.append("external decision memories should provide metadata.source")
    if _is_blank(raw_metadata.get("created_at")):
        warnings.append("metadata.created_at is recommended for auditability")
    if _is_blank(raw_metadata.get("updated_at")):
        warnings.append("metadata.updated_at is recommended for auditability")
    if _is_blank(raw_metadata.get("symbol")) and memory_type == "decision_case":
        warnings.append("symbol is blank; this case will only match on scenario similarity")

    return {
        "is_valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "normalized_metadata": normalized_metadata,
    }


def validate_decision_memory_record(
    record: dict[str, Any] | None,
    *,
    allowed_datasets: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Validate a full decision-memory record including text and metadata."""
    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(record, dict):
        return {
            "is_valid": False,
            "errors": ["record must be a JSON object"],
            "warnings": [],
            "normalized_metadata": normalize_decision_memory_metadata(None),
            "text": "",
            "title": "Untitled Decision Memory",
        }

    text = str(record.get("text", "") or "").strip()
    if not text:
        errors.append("record.text must be a non-empty string")
    elif len(text) < 80:
        warnings.append("record.text is very short; retrieval quality may be weak")

    metadata_validation = validate_decision_memory_metadata(
        record.get("metadata"),
        allowed_datasets=allowed_datasets,
    )
    errors.extend(metadata_validation["errors"])
    warnings.extend(metadata_validation["warnings"])
    normalized_metadata = metadata_validation["normalized_metadata"]

    return {
        "is_valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "normalized_metadata": normalized_metadata,
        "text": text,
        "title": normalized_metadata.get("title", "Untitled Decision Memory"),
    }


def summarize_decision_memory_validation(
    validations: list[dict[str, Any]],
    *,
    max_examples: int = 5,
) -> dict[str, Any]:
    """Build a compact validation summary for retrieval diagnostics."""
    invalid_examples: list[dict[str, Any]] = []
    warning_examples: list[dict[str, Any]] = []

    valid_count = 0
    invalid_count = 0
    warning_count = 0
    valid_warning_count = 0
    invalid_warning_count = 0

    for validation in validations:
        title = str(validation.get("title", "Untitled Decision Memory"))
        errors = list(validation.get("errors", []))
        warnings = list(validation.get("warnings", []))
        is_valid = bool(validation.get("is_valid"))
        if is_valid:
            valid_count += 1
        else:
            invalid_count += 1
            if len(invalid_examples) < max_examples:
                invalid_examples.append({"title": title, "errors": errors[:3]})
        if warnings:
            warning_count += 1
            if is_valid:
                valid_warning_count += 1
            else:
                invalid_warning_count += 1
            if len(warning_examples) < max_examples:
                warning_examples.append({"title": title, "warnings": warnings[:3]})

    return {
        "total_candidates": len(validations),
        "valid_candidates": valid_count,
        "invalid_candidates": invalid_count,
        "warning_candidates": warning_count,
        "valid_warning_candidates": valid_warning_count,
        "invalid_warning_candidates": invalid_warning_count,
        "invalid_examples": invalid_examples,
        "warning_examples": warning_examples,
    }


def _normalize_choice(value: Any, *, allowed: set[str], default: str) -> str:
    """Normalize a string-like field into an allowed choice."""
    normalized = _normalize_string(value, default=default).lower()
    if normalized not in allowed:
        return default
    return normalized


def _normalize_string(value: Any, *, default: str) -> str:
    """Normalize optional values into stripped strings."""
    normalized = str(value or "").strip()
    return normalized or default


def _normalize_string_list(value: Any) -> list[str]:
    """Normalize a field into a de-duplicated list of strings."""
    if not isinstance(value, list):
        return []
    seen: set[str] = set()
    normalized_values: list[str] = []
    for item in value:
        normalized = str(item).strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        normalized_values.append(normalized)
    return normalized_values


def _normalize_quality_score(value: Any) -> float:
    """Clamp quality scores into the [0, 1] range."""
    if isinstance(value, bool) or value is None:
        return 0.5
    if isinstance(value, (int, float)):
        return min(max(float(value), 0.0), 1.0)
    if isinstance(value, str):
        try:
            parsed = float(value.strip())
        except ValueError:
            return 0.5
        return min(max(parsed, 0.0), 1.0)
    return 0.5


def _validate_choice_field(
    metadata: dict[str, Any],
    field_name: str,
    *,
    allowed: set[str],
    errors: list[str],
    optional: bool = False,
) -> None:
    """Validate a raw metadata choice field against allowed values."""
    raw_value = metadata.get(field_name)
    if optional and _is_blank(raw_value):
        return
    normalized = str(raw_value or "").strip().lower()
    if normalized not in allowed:
        allowed_labels = ", ".join(sorted(allowed))
        errors.append(
            f"metadata.{field_name} must be one of [{allowed_labels}], got '{raw_value}'"
        )


def _validate_list_metadata_field(
    metadata: dict[str, Any],
    field_name: str,
    *,
    errors: list[str],
) -> None:
    """Validate that list-like metadata fields are stored as string lists."""
    raw_value = metadata.get(field_name)
    if raw_value in (None, ""):
        return
    if not isinstance(raw_value, list):
        errors.append(f"metadata.{field_name} must be a list of strings")
        return
    if any(not str(item).strip() for item in raw_value):
        errors.append(f"metadata.{field_name} must not contain blank items")


def _parse_float(value: Any) -> float | None:
    """Parse an optional numeric-like value into a float."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _is_blank(value: Any) -> bool:
    """Return True when a field should be treated as empty."""
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, list):
        return len(value) == 0
    return False
