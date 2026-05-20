from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import unittest

from backend.src.agents.reflection import ReflectionTask
from backend.src.services.reflection import ReflectionContextService


TESTS_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT_DIR = TESTS_DIR.parent
FIXTURES_DIR = TESTS_DIR / "fixtures" / "decision_memory"
DATA_DIR = PROJECT_ROOT_DIR / "backend" / "data" / "dynamic" / "processed"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(slots=True)
class FakeDocument:
    page_content: str
    metadata: dict


class FakeRetriever:
    def __init__(self, documents: list[FakeDocument]) -> None:
        self.documents = documents

    def search(self, query: str, *, datasets, k: int, metadata_filter=None):
        return list(self.documents)

    def load_all_documents(self, datasets):
        return list(self.documents)


class ReflectionContextServiceTests(unittest.TestCase):
    def test_context_includes_outcome_and_candidate_memory_seed(self) -> None:
        valid_record = load_json(DATA_DIR / "sample_decision_case.json")
        invalid_record = load_json(FIXTURES_DIR / "invalid_decision_memory_record.json")
        retriever = FakeRetriever(
            [
                FakeDocument(valid_record["text"], valid_record["metadata"]),
                FakeDocument(invalid_record["text"], invalid_record["metadata"]),
            ]
        )
        service = ReflectionContextService(retriever=retriever, backend=object())
        task = ReflectionTask.from_decision_payload(
            {
                "subject": "NVIDIA event-driven rally watch",
                "symbol": "NVDA",
                "trade_date": "2026-05-20",
                "decision_summary": "The setup looked extended and the stance stayed cautious.",
                "recommendation": "keep_watch",
                "confidence": "medium",
                "portfolio_context_used": True,
            },
            analyst_payload={
                "subject": "NVIDIA event-driven rally watch",
                "overall_summary": "Catalyst remained active but timing looked stretched.",
                "key_signals": ["news catalyst remains active"],
                "portfolio_risks": ["crowded trade risk is rising"],
            },
            portfolio_context={
                "cash_pct": 8,
                "positions": [{"symbol": "NVDA", "weight_pct": 7.5}],
            },
            execution_summary={
                "entry_date": "2026-05-21",
                "exit_date": "2026-05-28",
                "holding_period_days": 7,
                "position_size_pct": 7.5,
            },
            outcome_metrics={
                "realized_pnl_pct": 1.8,
                "benchmark_relative_return_pct": -0.6,
                "max_drawdown_pct": -3.2,
            },
            exit_context={
                "exit_reason": "trimmed into strength after rebound",
                "summary": "Partial profit-taking after the fade stabilized.",
            },
            post_trade_notes="Timing caution helped, but the rebound persistence was still underweighted.",
            realized_outcome={
                "outcome_label": "mixed",
                "summary": "The stock initially extended higher before fading back.",
            },
            feedback_notes="The original timing caution helped, but upside persistence was underweighted.",
        )

        context = service.analyze(task)

        self.assertEqual(1, context["document_count"])
        self.assertEqual(2, context["validation_summary"]["total_candidates"])
        self.assertEqual(1, context["validation_summary"]["invalid_candidates"])
        self.assertEqual("mixed", context["realized_outcome"]["outcome_label"])
        self.assertEqual("mixed", context["candidate_memory_seed"]["outcome_label"])
        self.assertEqual("NVDA", context["candidate_memory_seed"]["symbol"])
        self.assertEqual(1.8, context["outcome_metrics"]["realized_pnl_pct"])
        self.assertTrue(context["post_trade_validation"]["is_valid"])
        self.assertEqual("complete", context["post_trade_completeness"]["status"])
        self.assertIn("trimmed into strength", context["candidate_memory_seed"]["exit_reason"])
        self.assertIn("realized_pnl_pct", context["realized_outcome"]["summary"])
        self.assertTrue(context["query"])


if __name__ == "__main__":
    unittest.main()
