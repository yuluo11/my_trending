from __future__ import annotations

import unittest

from backend.src.agents.reflection.base_agent import BaseReflectionAgent, ReflectionTask
from backend.src.services.decision.memory import validate_decision_memory_record


class FakeReflectionContextService:
    agent_name = "reflection_agent"

    def analyze(self, task, *, datasets=None, metadata_filter=None, k=None):
        return {
            "query": task.subject,
            "reflection_profile": {"outcome_label": "failed"},
            "document_count": 1,
            "validation_summary": {
                "total_candidates": 1,
                "valid_candidates": 1,
                "invalid_candidates": 0,
                "warning_candidates": 0,
                "valid_warning_candidates": 0,
                "invalid_warning_candidates": 0,
                "invalid_examples": [],
                "warning_examples": [],
            },
            "documents": [
                {
                    "title": "Prior Failed Momentum Case",
                    "text": "A prior case showed that chasing extended momentum without enough downside review failed.",
                    "metadata": {
                        "title": "Prior Failed Momentum Case",
                        "memory_type": "decision_postmortem",
                        "fit": "medium",
                    },
                    "fit": "medium",
                    "match_reasons": ["similar subject framing"],
                }
            ],
            "historical_cases": [],
            "evidence": [],
            "original_decision": dict(task.decision_output or {}),
            "analyst_summary": {
                "overall_summary": "Signals were strong but the setup looked extended.",
            },
            "realized_outcome": {
                "outcome_label": "failed",
                "summary": "The setup reversed quickly after the decision.",
            },
            "execution_summary": dict(task.execution_summary or {}),
            "outcome_metrics": dict(task.outcome_metrics or {}),
            "exit_context": dict(task.exit_context or {}),
            "post_trade_notes": task.post_trade_notes or "",
            "feedback_notes": task.feedback_notes or "",
            "candidate_memory_seed": {
                "title": "NVDA NVIDIA event-driven rally watch Postmortem",
                "subject": task.subject,
                "symbol": task.symbol,
                "recommendation": "keep_watch",
                "confidence": "medium",
                "outcome_label": "failed",
            },
        }

    def build_candidate_memory(
        self,
        task,
        *,
        reflection_summary,
        what_worked,
        what_failed_or_underweighted,
        lessons,
        future_adjustments,
    ):
        from backend.src.services.reflection import build_candidate_postmortem_record

        return build_candidate_postmortem_record(
            subject=task.subject,
            symbol=task.symbol,
            trade_date=task.trade_date,
            recommendation="keep_watch",
            confidence="medium",
            outcome_label="failed",
            reflection_summary=reflection_summary,
            what_worked=what_worked,
            what_failed_or_underweighted=what_failed_or_underweighted,
            lessons=lessons,
            future_adjustments=future_adjustments,
            execution_summary=task.execution_summary,
            outcome_metrics=task.outcome_metrics,
            exit_context=task.exit_context,
            post_trade_notes=task.post_trade_notes,
        )


class ReflectionAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = BaseReflectionAgent(knowledge_service=FakeReflectionContextService())

    def test_from_decision_payload_preserves_post_trade_review_fields(self) -> None:
        task = ReflectionTask.from_decision_payload(
            {
                "subject": "NVIDIA event-driven rally watch",
                "symbol": "NVDA",
                "trade_date": "2026-05-20",
                "decision_summary": "The stance stayed cautious because the setup looked extended.",
                "recommendation": "keep_watch",
                "confidence": "medium",
            },
            execution_summary={
                "entry_date": "2026-05-21",
                "exit_date": "2026-05-28",
                "holding_period_days": "7",
            },
            outcome_metrics={"realized_pnl_pct": "-3.2%"},
            exit_context={"exit_reason": "stopped out after momentum failure"},
            post_trade_notes="The thesis horizon was too generous for this momentum setup.",
            realized_outcome={"outcome_label": "failed"},
            feedback_notes="The setup reversed faster than expected.",
        )

        self.assertEqual("failed", task.realized_outcome["outcome_label"])
        self.assertEqual(-3.2, task.outcome_metrics["realized_pnl_pct"])
        self.assertEqual(7.0, task.execution_summary["holding_period_days"])
        self.assertEqual("stopped out after momentum failure", task.exit_context["exit_reason"])
        self.assertIn("momentum setup", task.post_trade_notes)
        self.assertEqual("The setup reversed faster than expected.", task.feedback_notes)

    def test_fallback_emits_postmortem_fields_and_candidate_memory(self) -> None:
        task = ReflectionTask(
            subject="NVIDIA event-driven rally watch",
            symbol="NVDA",
            trade_date="2026-05-20",
            decision_output={
                "decision_summary": "Signals were constructive but the setup looked extended.",
                "recommendation": "keep_watch",
                "confidence": "medium",
                "portfolio_context_used": True,
            },
            portfolio_context={
                "cash_pct": 7,
                "positions": [{"symbol": "NVDA", "weight_pct": 7.5}],
            },
            execution_summary={
                "entry_date": "2026-05-21",
                "exit_date": "2026-05-28",
                "holding_period_days": 7,
                "position_size_pct": 7.5,
            },
            outcome_metrics={
                "realized_pnl_pct": -4.4,
                "max_drawdown_pct": -6.1,
                "benchmark_relative_return_pct": -3.5,
            },
            exit_context={
                "exit_reason": "stopped out after catalyst follow-through failed",
                "exit_trigger": "break below support",
            },
            post_trade_notes="The trade was too dependent on immediate momentum continuation.",
            realized_outcome={
                "outcome_label": "failed",
                "summary": "The setup reversed quickly after the decision.",
            },
            feedback_notes="The review should account for how quickly momentum failed.",
        )

        result = self.agent.invoke(task)

        self.assertEqual("decrease", result["confidence_change"])
        self.assertTrue(result["what_worked"])
        self.assertTrue(result["what_failed_or_underweighted"])
        self.assertTrue(result["lessons"])
        self.assertTrue(result["future_adjustments"])
        self.assertTrue(result["reference_cases"])
        self.assertTrue(result["memory_persistence"]["should_persist"])
        self.assertEqual("high", result["memory_persistence"]["priority"])
        self.assertIn("holding period", " ".join(result["lessons"]).lower())
        self.assertIn("4.40%", " ".join(result["what_failed_or_underweighted"]))
        validation = validate_decision_memory_record(
            result["candidate_memory"],
            allowed_datasets=("dynamic",),
        )
        self.assertTrue(validation["is_valid"])


if __name__ == "__main__":
    unittest.main()
