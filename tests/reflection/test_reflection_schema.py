from __future__ import annotations

import unittest

from backend.src.services.reflection import (
    assess_memory_persistence_candidate,
    assess_post_trade_review_completeness,
    normalize_execution_summary,
    normalize_exit_context,
    normalize_outcome_metrics,
    validate_post_trade_review,
)


class ReflectionSchemaTests(unittest.TestCase):
    def test_normalize_outcome_metrics_parses_percent_like_fields(self) -> None:
        normalized = normalize_outcome_metrics(
            {
                "realized_pnl_pct": "4.25%",
                "benchmark_relative_return": "-1.50%",
                "max_drawdown_pct": "-3.75%",
                "holding_period_days": "12",
                "performance_assessment": "Constructive follow-through",
            }
        )

        self.assertEqual(4.25, normalized["realized_pnl_pct"])
        self.assertEqual(-1.5, normalized["benchmark_relative_return"])
        self.assertEqual(-1.5, normalized["benchmark_relative_return_pct"])
        self.assertEqual(-3.75, normalized["max_drawdown_pct"])
        self.assertEqual(12.0, normalized["holding_period_days"])

    def test_normalize_execution_and_exit_context_drop_blank_values(self) -> None:
        execution = normalize_execution_summary(
            {
                "entry_date": "2026-05-21",
                "exit_date": "",
                "entry_price": "101.5",
                "summary": "  ",
            }
        )
        exit_context = normalize_exit_context(
            {
                "exit_reason": "trimmed into strength",
                "notes": "",
            }
        )

        self.assertEqual({"entry_date": "2026-05-21", "entry_price": 101.5}, execution)
        self.assertEqual({"exit_reason": "trimmed into strength"}, exit_context)

    def test_validate_post_trade_review_reports_date_and_drawdown_issues(self) -> None:
        validation = validate_post_trade_review(
            execution_summary={
                "entry_date": "2026-05-28",
                "exit_date": "2026-05-21",
                "holding_period_days": "-2",
            },
            outcome_metrics={"max_drawdown_pct": "2.5%"},
            exit_context={},
        )

        self.assertFalse(validation["is_valid"])
        self.assertTrue(
            any("entry_date" in error for error in validation["errors"])
        )
        self.assertTrue(
            any("holding_period_days" in error for error in validation["errors"])
        )
        self.assertTrue(
            any("max_drawdown_pct" in warning for warning in validation["warnings"])
        )

    def test_assess_post_trade_review_completeness_distinguishes_complete_and_partial(self) -> None:
        complete = assess_post_trade_review_completeness(
            execution_summary={"entry_date": "2026-05-21", "exit_date": "2026-05-28"},
            outcome_metrics={"realized_pnl_pct": "2.5%"},
            exit_context={"exit_reason": "trimmed into strength"},
            realized_outcome={"outcome_label": "worked"},
            post_trade_notes="The thesis played out with a measured exit.",
        )
        partial = assess_post_trade_review_completeness(
            outcome_metrics={"realized_pnl_pct": "0%"},
            realized_outcome={},
            post_trade_notes="Need a fuller review later.",
        )

        self.assertEqual("complete", complete["status"])
        self.assertEqual("partial", partial["status"])
        self.assertIn("execution_summary", partial["missing_inputs"])

    def test_assess_memory_persistence_candidate_requires_known_outcome_and_lessons(self) -> None:
        assessment = assess_memory_persistence_candidate(
            reflection_result={
                "reflection_summary": "The original setup partially worked.",
                "what_worked": ["Timing caution helped."],
                "what_failed_or_underweighted": ["Upside persistence was underweighted."],
                "lessons": ["Wait for stronger follow-through confirmation."],
                "future_adjustments": ["Require a tighter catalyst timing checklist."],
                "reference_cases": [],
                "candidate_memory": None,
            },
            outcome_label="mixed",
            post_trade_validation={"is_valid": True, "warnings": [], "errors": []},
        )
        blocked = assess_memory_persistence_candidate(
            reflection_result={
                "reflection_summary": "Need more evidence.",
                "what_worked": [],
                "what_failed_or_underweighted": [],
                "lessons": [],
                "future_adjustments": [],
                "reference_cases": [],
            },
            outcome_label="unknown",
            post_trade_validation={"is_valid": True, "warnings": [], "errors": []},
        )

        self.assertTrue(assessment["should_persist"])
        self.assertEqual("high", assessment["priority"])
        self.assertFalse(blocked["should_persist"])
        self.assertTrue(
            any("unknown" in issue for issue in blocked["blocking_issues"])
        )


if __name__ == "__main__":
    unittest.main()
