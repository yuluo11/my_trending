"""Reflection-layer services.

Only postmortem context shaping, retrieval, and schema helpers live here.
Reasoning belongs under ``agents.reflection`` so the reflection layer stays easy
to debug and review.
"""

from .context_service import ReflectionContextService
from .persistence_service import ReflectionPersistenceService
from .schema import (
    ALLOWED_CONFIDENCE_CHANGES,
    assess_memory_persistence_candidate,
    assess_post_trade_review_completeness,
    build_candidate_postmortem_record,
    infer_outcome_label,
    normalize_confidence_change,
    normalize_execution_summary,
    normalize_exit_context,
    normalize_outcome_metrics,
    validate_post_trade_review,
    validate_reflection_result,
)

__all__ = [
    "ALLOWED_CONFIDENCE_CHANGES",
    "ReflectionContextService",
    "ReflectionPersistenceService",
    "assess_memory_persistence_candidate",
    "assess_post_trade_review_completeness",
    "build_candidate_postmortem_record",
    "infer_outcome_label",
    "normalize_confidence_change",
    "normalize_execution_summary",
    "normalize_exit_context",
    "normalize_outcome_metrics",
    "validate_post_trade_review",
    "validate_reflection_result",
]
