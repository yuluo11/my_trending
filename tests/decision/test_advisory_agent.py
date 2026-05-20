from __future__ import annotations

import unittest

from backend.src.agents.decision.base_agent import BaseDecisionAgent, DecisionTask


class FakeDecisionKnowledgeService:
    agent_name = "decision_advisory"

    def analyze(self, task, *, datasets=None, metadata_filter=None, k=None):
        return {
            "query": task.subject,
            "scenario_profile": {"market_regime": "mixed"},
            "document_count": 0,
            "validation_summary": {
                "total_candidates": 0,
                "valid_candidates": 0,
                "invalid_candidates": 0,
                "warning_candidates": 0,
                "valid_warning_candidates": 0,
                "invalid_warning_candidates": 0,
                "invalid_examples": [],
                "warning_examples": [],
            },
            "documents": [],
            "evidence": [],
            "postmortem_lessons": [],
            "guidance_priors": {
                "symbol": task.symbol,
                "total_observations": 0,
                "top_guidance": [],
                "recommendation_breakdown": [],
                "top_reference_cases": [],
                "summary": "",
            },
        }


class DecisionAdvisoryAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = BaseDecisionAgent(knowledge_service=FakeDecisionKnowledgeService())

    def test_from_analyst_payload_preserves_portfolio_context(self) -> None:
        portfolio_context = {
            "cash_pct": 18,
            "max_single_name_pct": 10,
            "positions": [{"symbol": "NVDA", "weight_pct": 8.5}],
        }

        task = DecisionTask.from_analyst_payload(
            {
                "subject": "NVIDIA event-driven rally watch",
                "symbol": "NVDA",
                "overall_summary": "Signals are strong but the setup looks extended.",
            },
            portfolio_context=portfolio_context,
        )

        self.assertEqual(portfolio_context, task.portfolio_context)
        payload = self.agent.build_llm_payload(
            task,
            decision_context=self.agent.retrieve_decision_context(task),
        )
        self.assertEqual(portfolio_context, payload["task"]["portfolio_context"])
        self.assertIn("validation_summary", payload["decision_memory"])
        self.assertIn("postmortem_lessons", payload["decision_memory"])
        self.assertIn("guidance_priors", payload["decision_memory"])
        self.assertIn("portfolio_context_summary", payload["instructions"])
        self.assertIn("postmortem lessons", payload["instructions"])
        self.assertIn("recurring guidance priors", payload["instructions"])
        self.assertIn("applied_postmortem_guidance", payload["instructions"])

    def test_fallback_uses_portfolio_limits_and_emits_enhanced_fields(self) -> None:
        task = DecisionTask(
            subject="NVIDIA event-driven rally watch",
            symbol="NVDA",
            overall_summary="Catalyst and sentiment are strong but the setup looks extended.",
            overall_confidence="medium",
            key_signals=["news catalyst remains active", "momentum is still strong"],
            portfolio_risks=["crowded trade risk is rising", "valuation risk is elevated"],
            cross_analyst_observations=["Signals are constructive but timing looks stretched"],
            portfolio_context={
                "cash_pct": 12,
                "max_single_name_pct": 10,
                "positions": [{"symbol": "NVDA", "weight_pct": 11.2}],
            },
        )

        result = self.agent.invoke(task)

        self.assertEqual("consider_reduce", result["recommendation"])
        self.assertTrue(result["portfolio_context_used"])
        self.assertIn("cash_pct=12.0", result["portfolio_context_summary"])
        self.assertIn("11.2%", result["position_impact"])
        self.assertTrue(result["timing_decision"])
        self.assertTrue(result["action_conditions"])
        self.assertTrue(result["no_action_reasons"])
        self.assertIn("single-name limit", " ".join(result["action_conditions"]).lower())

    def test_fallback_can_escalate_to_consider_buy_for_high_conviction_new_position(self) -> None:
        task = DecisionTask(
            subject="NVIDIA constructive reset",
            symbol="NVDA",
            overall_summary="Catalyst, trend, and analyst confidence are aligned after consolidation.",
            overall_confidence="high",
            key_signals=["news catalyst remains active", "trend has stabilized"],
            portfolio_risks=["valuation risk remains present"],
            cross_analyst_observations=["Signals are constructive and aligned"],
            portfolio_context={
                "cash_pct": 16,
                "max_single_name_pct": 10,
                "positions": [],
            },
        )

        result = self.agent.invoke(task)

        self.assertEqual("consider_buy", result["recommendation"])
        self.assertIn("No existing NVDA position", result["position_impact"])
        self.assertIn("measured", result["timing_decision"].lower())
        self.assertIn("current cash is about 16.0%", " ".join(result["action_conditions"]).lower())

    def test_low_cash_without_position_stays_cautious(self) -> None:
        task = DecisionTask(
            subject="NVIDIA constructive reset",
            symbol="NVDA",
            overall_summary="Catalyst and trend are constructive.",
            overall_confidence="high",
            key_signals=["news catalyst remains active"],
            portfolio_risks=["valuation risk remains present"],
            cross_analyst_observations=["Signals are constructive and aligned"],
            portfolio_context={
                "cash_pct": 3,
                "max_single_name_pct": 10,
                "positions": [],
            },
        )

        result = self.agent.invoke(task)

        self.assertEqual("keep_watch", result["recommendation"])
        self.assertTrue(
            any("available cash is limited" in reason.lower() for reason in result["no_action_reasons"])
        )

    def test_fallback_explicitly_mentions_postmortem_lessons_when_present(self) -> None:
        agent = BaseDecisionAgent(
            knowledge_service=type(
                "LessonDecisionKnowledgeService",
                (),
                {
                    "agent_name": "decision_advisory",
                    "analyze": lambda self, task, **kwargs: {
                        "query": task.subject,
                        "scenario_profile": {"market_regime": "mixed"},
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
                        "documents": [],
                        "evidence": [],
                        "postmortem_lessons": [
                            {
                                "title": "Momentum Rebound Postmortem",
                                "fit": "medium",
                                "lesson": "Require stronger confirmation before adding to an extended move.",
                            }
                        ],
                        "guidance_priors": {
                            "symbol": task.symbol,
                            "total_observations": 3,
                            "top_guidance": [
                                {
                                    "label": "Require stronger confirmation before adding to an extended move.",
                                    "count": 3,
                                }
                            ],
                            "recommendation_breakdown": [
                                {"label": "keep_watch", "count": 2},
                            ],
                            "top_reference_cases": [
                                {"label": "Momentum Rebound Postmortem", "count": 3},
                            ],
                            "summary": "For NVDA, recurring applied guidance has most often emphasized 'Require stronger confirmation before adding to an extended move.' (3 observations).",
                        },
                    },
                },
            )()
        )
        task = DecisionTask(
            subject="NVIDIA momentum rebound review",
            symbol="NVDA",
            overall_summary="Momentum is still active but the rebound looks fragile.",
            overall_confidence="medium",
            key_signals=["momentum is still active"],
            portfolio_risks=["event fade risk is rising"],
            cross_analyst_observations=["Signals are constructive but timing looks stretched"],
        )

        result = agent.invoke(task)

        self.assertIn("postmortem lesson", result["rationale"].lower())
        self.assertIn("recurring guidance prior", result["rationale"].lower())
        self.assertTrue(
            any("stronger confirmation" in item.lower() for item in result["no_action_reasons"])
            or any("stronger confirmation" in item.lower() for item in result["action_conditions"])
        )
        self.assertTrue(result["applied_postmortem_guidance"])
        self.assertIn("stronger confirmation", result["applied_postmortem_guidance"][0].lower())

    def test_fallback_downgrades_confidence_when_recurring_guidance_conflicts(self) -> None:
        agent = BaseDecisionAgent(
            knowledge_service=type(
                "GuidanceConflictDecisionKnowledgeService",
                (),
                {
                    "agent_name": "decision_advisory",
                    "analyze": lambda self, task, **kwargs: {
                        "query": task.subject,
                        "scenario_profile": {"market_regime": "mixed"},
                        "document_count": 0,
                        "validation_summary": {
                            "total_candidates": 0,
                            "valid_candidates": 0,
                            "invalid_candidates": 0,
                            "warning_candidates": 0,
                            "valid_warning_candidates": 0,
                            "invalid_warning_candidates": 0,
                            "invalid_examples": [],
                            "warning_examples": [],
                        },
                        "documents": [],
                        "evidence": [],
                        "postmortem_lessons": [],
                        "guidance_priors": {
                            "symbol": task.symbol,
                            "total_observations": 3,
                            "top_guidance": [
                                {
                                    "label": "Require stronger confirmation before adding to an extended move.",
                                    "count": 3,
                                }
                            ],
                            "recommendation_breakdown": [
                                {"label": "keep_watch", "count": 3},
                            ],
                            "top_reference_cases": [],
                            "summary": "For NVDA, recurring applied guidance has most often emphasized stronger confirmation before adding.",
                        },
                    },
                },
            )()
        )
        task = DecisionTask(
            subject="NVIDIA constructive reset",
            symbol="NVDA",
            overall_summary="Catalyst, trend, and analyst confidence are aligned after consolidation.",
            overall_confidence="high",
            key_signals=["news catalyst remains active", "trend has stabilized"],
            portfolio_risks=["valuation risk remains present"],
            cross_analyst_observations=["Signals are constructive and aligned"],
            portfolio_context={
                "cash_pct": 16,
                "max_single_name_pct": 10,
                "positions": [],
            },
        )

        result = agent.invoke(task)

        self.assertEqual("consider_buy", result["recommendation"])
        self.assertEqual("medium", result["confidence"])
        self.assertIn("confidence stays one step lower", result["rationale"].lower())


if __name__ == "__main__":
    unittest.main()
