from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from backend.src.services.reflection import ReflectionPersistenceService, build_candidate_postmortem_record
from backend.src.knowledge.repository import KnowledgeRepository


class ReflectionPersistenceServiceTests(unittest.TestCase):
    def test_persist_reflection_result_skips_when_memory_should_not_persist(self) -> None:
        with TemporaryDirectory() as tmpdir:
            repository = KnowledgeRepository(data_root=Path(tmpdir))
            service = ReflectionPersistenceService(repository=repository)

            result = service.persist_reflection_result(
                {
                    "symbol": "NVDA",
                    "trade_date": "2026-05-20",
                    "memory_persistence": {
                        "should_persist": False,
                        "blocking_issues": ["outcome_label is still unknown"],
                    },
                    "candidate_memory": build_candidate_postmortem_record(
                        subject="NVIDIA event-driven rally watch",
                        symbol="NVDA",
                        trade_date="2026-05-20",
                        recommendation="keep_watch",
                        confidence="medium",
                        outcome_label="unknown",
                        reflection_summary="Need more evidence before persisting this case.",
                        what_worked=["Timing caution was directionally useful."],
                        what_failed_or_underweighted=["Outcome evidence is still incomplete."],
                        lessons=["Do not persist postmortems until the outcome is known."],
                        future_adjustments=["Wait for a known exit and final metrics."],
                    ),
                }
            )

            self.assertFalse(result["persisted"])
            self.assertEqual("skipped", result["status"])
            self.assertFalse((Path(tmpdir) / "dynamic" / "processed").exists())

    def test_persist_reflection_result_writes_candidate_memory_and_updates_manifest(self) -> None:
        with TemporaryDirectory() as tmpdir:
            repository = KnowledgeRepository(data_root=Path(tmpdir))
            service = ReflectionPersistenceService(repository=repository)
            reflection_result = {
                "symbol": "NVDA",
                "trade_date": "2026-05-20",
                "reflection_summary": "The setup partially worked, but execution and timing needed tighter controls.",
                "what_worked": ["Timing caution helped avoid a worse entry."],
                "what_failed_or_underweighted": ["The rebound lasted longer than expected before fading."],
                "lessons": ["Require stronger follow-through confirmation before scaling in."],
                "future_adjustments": ["Add an explicit rebound persistence check to the setup review."],
                "reference_cases": [],
                "memory_persistence": {
                    "should_persist": True,
                    "priority": "high",
                    "blocking_issues": [],
                    "supporting_reasons": ["outcome_label is mixed"],
                },
                "candidate_memory": build_candidate_postmortem_record(
                    subject="NVIDIA event-driven rally watch",
                    symbol="NVDA",
                    trade_date="2026-05-20",
                    recommendation="keep_watch",
                    confidence="medium",
                    outcome_label="mixed",
                    reflection_summary="The setup partially worked, but execution and timing needed tighter controls.",
                    what_worked=["Timing caution helped avoid a worse entry."],
                    what_failed_or_underweighted=["The rebound lasted longer than expected before fading."],
                    lessons=["Require stronger follow-through confirmation before scaling in."],
                    future_adjustments=["Add an explicit rebound persistence check to the setup review."],
                    execution_summary={"entry_date": "2026-05-21", "exit_date": "2026-05-28"},
                    outcome_metrics={"realized_pnl_pct": -1.4},
                    exit_context={"exit_reason": "trimmed into failed rebound"},
                    post_trade_notes="This should be reusable for later event-driven momentum reviews.",
                ),
            }

            result = service.persist_reflection_result(reflection_result)

            self.assertTrue(result["persisted"])
            self.assertEqual("persisted", result["status"])
            record_path = Path(result["path"])
            self.assertTrue(record_path.exists())

            payload = json.loads(record_path.read_text(encoding="utf-8"))
            self.assertEqual("dynamic", payload["metadata"]["dataset"])
            self.assertEqual("decision_postmortem", payload["metadata"]["memory_type"])

            manifest = repository.load_manifest()
            processed_entries = manifest["datasets"]["dynamic"]["processed"]
            self.assertTrue(
                any(entry["name"] == result["record_name"] for entry in processed_entries)
            )


if __name__ == "__main__":
    unittest.main()
