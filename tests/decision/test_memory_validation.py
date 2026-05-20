from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from backend.src.agents.decision.base_agent import DecisionTask
from backend.src.knowledge.repository import KnowledgeRepository
from backend.src.services.decision import DecisionGuidanceObservationService
from backend.src.services.decision.memory import (
    DecisionKnowledgeService,
    summarize_decision_memory_validation,
    validate_decision_memory_record,
)


TESTS_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT_DIR = TESTS_DIR.parent

FIXTURES_DIR = TESTS_DIR / "fixtures" / "decision_memory"
DATA_DIR = (
    PROJECT_ROOT_DIR
    / "backend"
    / "data"
    / "dynamic"
    / "processed"
)


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


class DecisionMemoryValidationTests(unittest.TestCase):
    def test_valid_sample_record_passes_validation(self) -> None:
        record = load_json(DATA_DIR / "sample_decision_case.json")

        validation = validate_decision_memory_record(record, allowed_datasets=("dynamic",))

        self.assertTrue(validation["is_valid"])
        self.assertEqual([], validation["errors"])
        self.assertEqual("decision_memory", validation["normalized_metadata"]["category"])
        self.assertEqual("decision_case", validation["normalized_metadata"]["memory_type"])
        self.assertIn(
            "portfolio_state_tags",
            validation["normalized_metadata"],
        )

    def test_invalid_fixture_is_rejected_with_explicit_errors(self) -> None:
        record = load_json(FIXTURES_DIR / "invalid_decision_memory_record.json")

        validation = validate_decision_memory_record(record, allowed_datasets=("dynamic",))

        self.assertFalse(validation["is_valid"])
        self.assertIn("record.text must be a non-empty string", validation["errors"])
        self.assertIn("metadata.category must be 'decision_memory'", validation["errors"])
        self.assertTrue(
            any("metadata.recommendation must be one of" in error for error in validation["errors"])
        )
        self.assertTrue(
            any("metadata.signal_tags must be a list of strings" in error for error in validation["errors"])
        )

    def test_warning_fixture_stays_usable_but_reports_warnings(self) -> None:
        record = load_json(FIXTURES_DIR / "warning_decision_memory_record.json")

        validation = validate_decision_memory_record(record, allowed_datasets=("dynamic",))

        self.assertTrue(validation["is_valid"])
        self.assertEqual([], validation["errors"])
        self.assertTrue(
            any(
                "external decision memories should provide metadata.source" in warning
                for warning in validation["warnings"]
            )
        )
        self.assertTrue(
            any("metadata.created_at is recommended" in warning for warning in validation["warnings"])
        )

    def test_validation_summary_counts_invalid_and_warning_candidates(self) -> None:
        valid = validate_decision_memory_record(
            load_json(DATA_DIR / "sample_decision_case.json"),
            allowed_datasets=("dynamic",),
        )
        invalid = validate_decision_memory_record(
            load_json(FIXTURES_DIR / "invalid_decision_memory_record.json"),
            allowed_datasets=("dynamic",),
        )
        warning = validate_decision_memory_record(
            load_json(FIXTURES_DIR / "warning_decision_memory_record.json"),
            allowed_datasets=("dynamic",),
        )

        summary = summarize_decision_memory_validation([valid, invalid, warning], max_examples=2)

        self.assertEqual(3, summary["total_candidates"])
        self.assertEqual(2, summary["valid_candidates"])
        self.assertEqual(1, summary["invalid_candidates"])
        self.assertEqual(2, summary["warning_candidates"])
        self.assertEqual(1, summary["valid_warning_candidates"])
        self.assertEqual(1, summary["invalid_warning_candidates"])
        self.assertEqual("Invalid Decision Memory Case", summary["invalid_examples"][0]["title"])

    def test_retrieval_skips_invalid_documents_and_reports_validation_summary(self) -> None:
        valid_record = load_json(DATA_DIR / "sample_decision_case.json")
        invalid_record = load_json(FIXTURES_DIR / "invalid_decision_memory_record.json")
        warning_record = load_json(FIXTURES_DIR / "warning_decision_memory_record.json")

        retriever = FakeRetriever(
            [
                FakeDocument(valid_record["text"], valid_record["metadata"]),
                FakeDocument(invalid_record["text"], invalid_record["metadata"]),
                FakeDocument(warning_record["text"], warning_record["metadata"]),
            ]
        )
        service = DecisionKnowledgeService(retriever=retriever, backend=object())
        task = DecisionTask(
            subject="NVIDIA event-driven rally watch",
            symbol="NVDA",
            overall_summary="Catalyst and sentiment are strong but the setup looks extended.",
            key_signals=["news catalyst remains active"],
            portfolio_risks=["crowded trade risk is rising"],
            cross_analyst_observations=["Signals are constructive but timing looks stretched"],
        )

        context = service.analyze(task)

        self.assertEqual(2, context["document_count"])
        self.assertEqual(3, context["validation_summary"]["total_candidates"])
        self.assertEqual(1, context["validation_summary"]["invalid_candidates"])
        self.assertEqual(1, context["validation_summary"]["valid_warning_candidates"])
        self.assertTrue(
            any(
                document["title"] == "Warning Decision Memory Case"
                and document["metadata"]["validation_warnings"]
                for document in context["documents"]
            )
        )

    def test_portfolio_state_tags_influence_retrieval_order(self) -> None:
        shared_text = (
            "A prior internal decision on NVIDIA concluded that the setup required measured sizing "
            "because risk and reward were not perfectly balanced."
        )
        existing_position_record = {
            "text": shared_text,
            "metadata": {
                "source": "internal_decision_log",
                "source_type": "internal",
                "title": "Existing Position Near Limit Case",
                "created_at": "2026-05-16T10:00:00+08:00",
                "updated_at": "2026-05-16T10:00:00+08:00",
                "category": "decision_memory",
                "memory_type": "decision_case",
                "tags": ["semiconductors"],
                "symbol": "NVDA",
                "subject": "NVIDIA constructive setup",
                "topic": "decision-memory",
                "recommendation": "hold",
                "confidence": "medium",
                "market_regime": "event_driven",
                "analyst_alignment": "aligned",
                "signal_tags": ["news_catalyst", "momentum"],
                "risk_tags": ["valuation_risk"],
                "timing_tags": ["short_term"],
                "portfolio_state_tags": ["existing_position", "near_single_name_limit"],
                "outcome_label": "worked",
                "quality_score": 0.8,
                "dataset": "dynamic",
            },
        }
        fresh_entry_record = {
            "text": shared_text,
            "metadata": {
                "source": "internal_decision_log",
                "source_type": "internal",
                "title": "Fresh Entry With Cash Case",
                "created_at": "2026-05-17T10:00:00+08:00",
                "updated_at": "2026-05-17T10:00:00+08:00",
                "category": "decision_memory",
                "memory_type": "decision_case",
                "tags": ["semiconductors"],
                "symbol": "NVDA",
                "subject": "NVIDIA constructive setup",
                "topic": "decision-memory",
                "recommendation": "consider_buy",
                "confidence": "medium",
                "market_regime": "event_driven",
                "analyst_alignment": "aligned",
                "signal_tags": ["news_catalyst", "momentum"],
                "risk_tags": ["valuation_risk"],
                "timing_tags": ["short_term"],
                "portfolio_state_tags": ["no_position", "ample_cash"],
                "outcome_label": "worked",
                "quality_score": 0.8,
                "dataset": "dynamic",
            },
        }

        retriever = FakeRetriever(
            [
                FakeDocument(existing_position_record["text"], existing_position_record["metadata"]),
                FakeDocument(fresh_entry_record["text"], fresh_entry_record["metadata"]),
            ]
        )
        service = DecisionKnowledgeService(retriever=retriever, backend=object())

        near_limit_task = DecisionTask(
            subject="NVIDIA constructive setup",
            symbol="NVDA",
            overall_summary="Catalyst and trend are constructive.",
            overall_confidence="high",
            key_signals=["news catalyst remains active", "momentum is still strong"],
            portfolio_risks=["valuation risk is elevated"],
            cross_analyst_observations=["Signals are constructive and aligned"],
            portfolio_context={
                "cash_pct": 8,
                "max_single_name_pct": 10,
                "positions": [{"symbol": "NVDA", "weight_pct": 9.4}],
            },
        )
        fresh_entry_task = DecisionTask(
            subject="NVIDIA constructive setup",
            symbol="NVDA",
            overall_summary="Catalyst and trend are constructive.",
            overall_confidence="high",
            key_signals=["news catalyst remains active", "momentum is still strong"],
            portfolio_risks=["valuation risk is elevated"],
            cross_analyst_observations=["Signals are constructive and aligned"],
            portfolio_context={
                "cash_pct": 16,
                "max_single_name_pct": 10,
                "positions": [],
            },
        )

        near_limit_context = service.analyze(near_limit_task)
        fresh_entry_context = service.analyze(fresh_entry_task)

        self.assertEqual(
            "Existing Position Near Limit Case",
            near_limit_context["documents"][0]["title"],
        )
        self.assertEqual(
            "Fresh Entry With Cash Case",
            fresh_entry_context["documents"][0]["title"],
        )
        self.assertIn(
            "shared portfolio state tags",
            " ".join(near_limit_context["documents"][0]["match_reasons"]),
        )

    def test_postmortem_lessons_are_extracted_into_decision_context(self) -> None:
        with TemporaryDirectory() as tmpdir:
            repository = KnowledgeRepository(data_root=Path(tmpdir))
            observation_service = DecisionGuidanceObservationService(repository=repository)
            observation_service.persist_guidance_observation(
                {
                    "subject": "NVIDIA momentum rebound review",
                    "symbol": "NVDA",
                    "trade_date": "2026-05-19",
                    "decision_summary": "Watch the rebound until confirmation improves.",
                    "recommendation": "keep_watch",
                    "confidence": "medium",
                    "rationale": "A prior postmortem argued for stronger confirmation.",
                    "case_fit_assessment": "Partial fit to a failed rebound lesson.",
                    "reference_cases": [{"title": "Momentum Rebound Postmortem"}],
                    "applied_postmortem_guidance": [
                        "Require stronger confirmation before adding to an extended move."
                    ],
                }
            )

            postmortem_record = {
                "text": (
                    "Reflection summary: A prior momentum setup partially worked but became too dependent "
                    "on immediate follow-through.\n\n"
                    "Reusable lessons:\n"
                    "- Require stronger confirmation before adding to an extended move.\n"
                    "- Treat failed rebound persistence as a reason to lower confidence.\n\n"
                    "Future adjustments:\n"
                    "- Compare current setups against failed rebound postmortems before upgrading conviction.\n"
                ),
                "metadata": {
                    "source": "reflection_postmortem",
                    "source_type": "internal",
                    "title": "Momentum Rebound Postmortem",
                    "created_at": "2026-05-18T10:00:00+08:00",
                    "updated_at": "2026-05-18T10:00:00+08:00",
                    "category": "decision_memory",
                    "memory_type": "decision_postmortem",
                    "tags": ["momentum", "postmortem"],
                    "symbol": "NVDA",
                    "subject": "NVIDIA momentum rebound review",
                    "topic": "decision-memory",
                    "recommendation": "keep_watch",
                    "confidence": "medium",
                    "market_regime": "event_driven",
                    "analyst_alignment": "mixed",
                    "signal_tags": ["momentum"],
                    "risk_tags": ["event_fade"],
                    "timing_tags": ["short_term"],
                    "portfolio_state_tags": ["existing_position"],
                    "outcome_label": "mixed",
                    "quality_score": 0.85,
                    "dataset": "dynamic",
                },
            }
            retriever = FakeRetriever(
                [
                    FakeDocument(postmortem_record["text"], postmortem_record["metadata"]),
                ]
            )
            service = DecisionKnowledgeService(
                repository=repository,
                retriever=retriever,
                backend=object(),
            )
            task = DecisionTask(
                subject="NVIDIA momentum rebound review",
                symbol="NVDA",
                overall_summary="Catalyst stayed active but the rebound looked fragile.",
                overall_confidence="medium",
                key_signals=["momentum is still active"],
                portfolio_risks=["event fade risk is rising"],
                cross_analyst_observations=["Signals are constructive but timing looks stretched"],
                portfolio_context={
                    "cash_pct": 9,
                    "positions": [{"symbol": "NVDA", "weight_pct": 6.0}],
                },
            )

            context = service.analyze(task)

            self.assertEqual(1, context["document_count"])
            self.assertTrue(context["postmortem_lessons"])
            self.assertTrue(
                any(
                    "stronger confirmation" in item["lesson"].lower()
                    for item in context["postmortem_lessons"]
                )
            )
            self.assertEqual("NVDA", context["guidance_priors"]["symbol"])
            self.assertEqual(1, context["guidance_priors"]["total_observations"])
            self.assertTrue(context["guidance_priors"]["top_guidance"])
            self.assertIn("stronger confirmation", context["guidance_priors"]["summary"].lower())

    def test_guidance_priors_lightly_boost_aligned_documents_in_retrieval(self) -> None:
        with TemporaryDirectory() as tmpdir:
            repository = KnowledgeRepository(data_root=Path(tmpdir))
            observation_service = DecisionGuidanceObservationService(repository=repository)
            observation_service.persist_guidance_observation(
                {
                    "subject": "NVIDIA momentum rebound review",
                    "symbol": "NVDA",
                    "trade_date": "2026-05-19",
                    "decision_summary": "Watch the rebound until confirmation improves.",
                    "recommendation": "keep_watch",
                    "confidence": "medium",
                    "rationale": "A prior postmortem argued for stronger confirmation.",
                    "case_fit_assessment": "Partial fit to a failed rebound lesson.",
                    "reference_cases": [{"title": "Momentum Rebound Postmortem"}],
                    "applied_postmortem_guidance": [
                        "Require stronger confirmation before adding to an extended move."
                    ],
                }
            )
            observation_service.persist_guidance_observation(
                {
                    "subject": "NVIDIA momentum rebound follow-up",
                    "symbol": "NVDA",
                    "trade_date": "2026-05-20",
                    "decision_summary": "Stay patient until confirmation improves.",
                    "recommendation": "keep_watch",
                    "confidence": "medium",
                    "rationale": "The same lesson remains active.",
                    "case_fit_assessment": "The rebound still looks extended.",
                    "reference_cases": [{"title": "Momentum Rebound Postmortem"}],
                    "applied_postmortem_guidance": [
                        "Require stronger confirmation before adding to an extended move."
                    ],
                }
            )

            aligned_record = {
                "text": (
                    "This postmortem emphasized requiring stronger confirmation before adding to "
                    "an extended move after a fragile rebound."
                ),
                "metadata": {
                    "source": "reflection_postmortem",
                    "source_type": "internal",
                    "title": "Aligned Guidance Postmortem",
                    "created_at": "2026-05-18T10:00:00+08:00",
                    "updated_at": "2026-05-18T10:00:00+08:00",
                    "category": "decision_memory",
                    "memory_type": "decision_postmortem",
                    "tags": ["momentum", "extended"],
                    "symbol": "NVDA",
                    "subject": "NVIDIA momentum rebound review",
                    "topic": "decision-memory",
                    "recommendation": "keep_watch",
                    "confidence": "medium",
                    "market_regime": "event_driven",
                    "analyst_alignment": "mixed",
                    "signal_tags": ["momentum"],
                    "risk_tags": ["event_fade"],
                    "timing_tags": ["short_term"],
                    "portfolio_state_tags": ["existing_position"],
                    "outcome_label": "mixed",
                    "quality_score": 0.85,
                    "dataset": "dynamic",
                },
            }
            less_aligned_record = {
                "text": (
                    "This postmortem focused more on valuation risk and sector rotation than on "
                    "confirmation quality."
                ),
                "metadata": {
                    "source": "reflection_postmortem",
                    "source_type": "internal",
                    "title": "Less Aligned Guidance Postmortem",
                    "created_at": "2026-05-17T10:00:00+08:00",
                    "updated_at": "2026-05-17T10:00:00+08:00",
                    "category": "decision_memory",
                    "memory_type": "decision_postmortem",
                    "tags": ["valuation", "rotation"],
                    "symbol": "NVDA",
                    "subject": "NVIDIA momentum rebound review",
                    "topic": "decision-memory",
                    "recommendation": "keep_watch",
                    "confidence": "medium",
                    "market_regime": "event_driven",
                    "analyst_alignment": "mixed",
                    "signal_tags": ["momentum"],
                    "risk_tags": ["event_fade"],
                    "timing_tags": ["short_term"],
                    "portfolio_state_tags": ["existing_position"],
                    "outcome_label": "mixed",
                    "quality_score": 0.85,
                    "dataset": "dynamic",
                },
            }

            retriever = FakeRetriever(
                [
                    FakeDocument(less_aligned_record["text"], less_aligned_record["metadata"]),
                    FakeDocument(aligned_record["text"], aligned_record["metadata"]),
                ]
            )
            service = DecisionKnowledgeService(
                repository=repository,
                retriever=retriever,
                backend=object(),
            )
            task = DecisionTask(
                subject="NVIDIA momentum rebound review",
                symbol="NVDA",
                overall_summary="Catalyst stayed active but the rebound looked fragile.",
                overall_confidence="medium",
                key_signals=["momentum is still active"],
                portfolio_risks=["event fade risk is rising"],
                cross_analyst_observations=["Signals are constructive but timing looks stretched"],
                portfolio_context={
                    "cash_pct": 9,
                    "positions": [{"symbol": "NVDA", "weight_pct": 6.0}],
                },
            )

            context = service.analyze(task)

            self.assertEqual("Aligned Guidance Postmortem", context["documents"][0]["title"])
            self.assertIn(
                "aligned with recurring guidance priors for this symbol",
                context["documents"][0]["match_reasons"],
            )


if __name__ == "__main__":
    unittest.main()
