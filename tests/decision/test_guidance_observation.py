from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from backend.src.knowledge.repository import KnowledgeRepository
from backend.src.services.decision import (
    DecisionGuidanceObservationAnalyticsService,
    DecisionGuidanceObservationService,
)


class DecisionGuidanceObservationServiceTests(unittest.TestCase):
    def test_persist_guidance_observation_skips_without_applied_guidance(self) -> None:
        with TemporaryDirectory() as tmpdir:
            repository = KnowledgeRepository(data_root=Path(tmpdir))
            service = DecisionGuidanceObservationService(repository=repository)

            result = service.persist_guidance_observation(
                {
                    "subject": "NVIDIA momentum rebound review",
                    "symbol": "NVDA",
                    "trade_date": "2026-05-20",
                    "recommendation": "keep_watch",
                }
            )

            self.assertFalse(result["persisted"])
            self.assertEqual("skipped", result["status"])

    def test_persist_guidance_observation_writes_processed_record(self) -> None:
        with TemporaryDirectory() as tmpdir:
            repository = KnowledgeRepository(data_root=Path(tmpdir))
            service = DecisionGuidanceObservationService(repository=repository)

            decision_result = {
                "subject": "NVIDIA momentum rebound review",
                "symbol": "NVDA",
                "trade_date": "2026-05-20",
                "decision_summary": "The setup remains watchful because the rebound still looks fragile.",
                "recommendation": "keep_watch",
                "confidence": "medium",
                "rationale": "A retrieved postmortem lesson argues for stronger confirmation first.",
                "case_fit_assessment": "A medium-fit postmortem offered bounded caution.",
                "reference_cases": [
                    {
                        "title": "Momentum Rebound Postmortem",
                        "fit": "medium",
                        "memory_type": "decision_postmortem",
                        "why_relevant": "Similar rebound profile.",
                    }
                ],
                "applied_postmortem_guidance": [
                    "Require stronger confirmation before adding to an extended move."
                ],
            }

            result = service.persist_guidance_observation(decision_result)

            self.assertTrue(result["persisted"])
            self.assertEqual("persisted", result["status"])
            record_path = Path(result["path"])
            self.assertTrue(record_path.exists())

            payload = json.loads(record_path.read_text(encoding="utf-8"))
            self.assertEqual("decision_guidance_observation", payload["metadata"]["category"])
            self.assertEqual("dynamic", payload["metadata"]["dataset"])
            self.assertIn("Applied postmortem guidance:", payload["text"])

            manifest = repository.load_manifest()
            processed_entries = manifest["datasets"]["dynamic"]["processed"]
            self.assertTrue(
                any(entry["name"] == result["record_name"] for entry in processed_entries)
            )

    def test_summarize_observations_reports_top_guidance_and_breakdowns(self) -> None:
        with TemporaryDirectory() as tmpdir:
            repository = KnowledgeRepository(data_root=Path(tmpdir))
            persistence = DecisionGuidanceObservationService(repository=repository)
            analytics = DecisionGuidanceObservationAnalyticsService(repository=repository)

            persistence.persist_guidance_observation(
                {
                    "subject": "NVIDIA momentum rebound review",
                    "symbol": "NVDA",
                    "trade_date": "2026-05-20",
                    "decision_summary": "Watchful stance while rebound remains fragile.",
                    "recommendation": "keep_watch",
                    "confidence": "medium",
                    "rationale": "A postmortem lesson argues for stronger confirmation.",
                    "case_fit_assessment": "A medium-fit postmortem offered caution.",
                    "reference_cases": [{"title": "Momentum Rebound Postmortem"}],
                    "applied_postmortem_guidance": [
                        "Require stronger confirmation before adding to an extended move."
                    ],
                }
            )
            persistence.persist_guidance_observation(
                {
                    "subject": "NVIDIA constructive reset",
                    "symbol": "NVDA",
                    "trade_date": "2026-05-21",
                    "decision_summary": "Measured add only after confirmation.",
                    "recommendation": "consider_buy",
                    "confidence": "medium",
                    "rationale": "The same guidance still argues for staged sizing.",
                    "case_fit_assessment": "Partial fit to prior rebound lesson.",
                    "reference_cases": [{"title": "Momentum Rebound Postmortem"}],
                    "applied_postmortem_guidance": [
                        "Require stronger confirmation before adding to an extended move."
                    ],
                }
            )
            persistence.persist_guidance_observation(
                {
                    "subject": "Utilities defensive rotation",
                    "symbol": "XLU",
                    "trade_date": "2026-05-22",
                    "decision_summary": "Reduce exposure into weakening momentum.",
                    "recommendation": "consider_reduce",
                    "confidence": "medium",
                    "rationale": "A separate lesson warns that failed bounces fade quickly.",
                    "case_fit_assessment": "High-fit defensive postmortem.",
                    "reference_cases": [{"title": "Defensive Rotation Postmortem"}],
                    "applied_postmortem_guidance": [
                        "Treat failed rebound persistence as a reason to lower confidence."
                    ],
                }
            )

            summary = analytics.summarize_observations(top_n=3)

            self.assertEqual(3, summary["total_observations"])
            self.assertEqual(
                "Require stronger confirmation before adding to an extended move.",
                summary["top_guidance"][0]["label"],
            )
            self.assertEqual(2, summary["top_guidance"][0]["count"])
            self.assertTrue(
                any(
                    item["label"] == "NVDA" and item["count"] == 2
                    for item in summary["symbol_breakdown"]
                )
            )
            self.assertTrue(
                any(
                    item["label"] == "keep_watch" and item["count"] == 1
                    for item in summary["recommendation_breakdown"]
                )
            )
            self.assertTrue(
                any(
                    item["label"] == "Momentum Rebound Postmortem" and item["count"] == 2
                    for item in summary["top_reference_cases"]
                )
            )

    def test_summarize_guidance_priors_filters_by_symbol_and_builds_summary(self) -> None:
        with TemporaryDirectory() as tmpdir:
            repository = KnowledgeRepository(data_root=Path(tmpdir))
            persistence = DecisionGuidanceObservationService(repository=repository)
            analytics = DecisionGuidanceObservationAnalyticsService(repository=repository)

            persistence.persist_guidance_observation(
                {
                    "subject": "NVIDIA momentum rebound review",
                    "symbol": "NVDA",
                    "trade_date": "2026-05-20",
                    "decision_summary": "Watchful stance while rebound remains fragile.",
                    "recommendation": "keep_watch",
                    "confidence": "medium",
                    "rationale": "A postmortem lesson argues for stronger confirmation.",
                    "case_fit_assessment": "A medium-fit postmortem offered caution.",
                    "reference_cases": [{"title": "Momentum Rebound Postmortem"}],
                    "applied_postmortem_guidance": [
                        "Require stronger confirmation before adding to an extended move."
                    ],
                }
            )
            persistence.persist_guidance_observation(
                {
                    "subject": "NVIDIA constructive reset",
                    "symbol": "NVDA",
                    "trade_date": "2026-05-21",
                    "decision_summary": "Measured add only after confirmation.",
                    "recommendation": "consider_buy",
                    "confidence": "medium",
                    "rationale": "The same guidance still argues for staged sizing.",
                    "case_fit_assessment": "Partial fit to prior rebound lesson.",
                    "reference_cases": [{"title": "Momentum Rebound Postmortem"}],
                    "applied_postmortem_guidance": [
                        "Require stronger confirmation before adding to an extended move."
                    ],
                }
            )
            persistence.persist_guidance_observation(
                {
                    "subject": "Utilities defensive rotation",
                    "symbol": "XLU",
                    "trade_date": "2026-05-22",
                    "decision_summary": "Reduce exposure into weakening momentum.",
                    "recommendation": "consider_reduce",
                    "confidence": "medium",
                    "rationale": "A separate lesson warns that failed bounces fade quickly.",
                    "case_fit_assessment": "High-fit defensive postmortem.",
                    "reference_cases": [{"title": "Defensive Rotation Postmortem"}],
                    "applied_postmortem_guidance": [
                        "Treat failed rebound persistence as a reason to lower confidence."
                    ],
                }
            )

            priors = analytics.summarize_guidance_priors(
                datasets=("dynamic",),
                symbol="NVDA",
                top_n=2,
            )

            self.assertEqual("NVDA", priors["symbol"])
            self.assertEqual(2, priors["total_observations"])
            self.assertEqual(
                "Require stronger confirmation before adding to an extended move.",
                priors["top_guidance"][0]["label"],
            )
            self.assertEqual(2, priors["top_guidance"][0]["count"])
            self.assertIn("For NVDA", priors["summary"])
            self.assertIn("keep_watch", priors["summary"])
            self.assertNotIn("XLU", priors["summary"])


if __name__ == "__main__":
    unittest.main()
