"""Decision-memory retrieval service for advisory synthesis."""

from __future__ import annotations

from collections.abc import Iterable
import re
from typing import TYPE_CHECKING, Any

from ....knowledge.indexing import KnowledgeIndexer
from ....knowledge.repository import DatasetName, KnowledgeRepository
from ....knowledge.retriever import KnowledgeRetriever, VectorRetrieverBackend
from ..observation_service import DecisionGuidanceObservationAnalyticsService
from .schema import (
    normalize_decision_memory_metadata,
    summarize_decision_memory_validation,
    validate_decision_memory_record,
)

if TYPE_CHECKING:
    from ....agents.decision.base_agent import DecisionTask


class DecisionKnowledgeService:
    """Retrieve dynamic decision-memory records used by the advisory layer."""

    agent_name = "decision_advisory"
    default_datasets: tuple[DatasetName, ...] = ("dynamic",)
    default_k = 3

    def __init__(
        self,
        repository: KnowledgeRepository | None = None,
        retriever: KnowledgeRetriever | None = None,
        backend: VectorRetrieverBackend | None = None,
        observation_analytics: DecisionGuidanceObservationAnalyticsService | None = None,
    ) -> None:
        self.repository = repository or KnowledgeRepository()
        self.indexer = KnowledgeIndexer(self.repository)
        resolved_backend = backend or self.indexer.build_local_index(self.default_datasets)
        self.retriever = retriever or KnowledgeRetriever(self.repository, backend=resolved_backend)
        self.observation_analytics = observation_analytics or DecisionGuidanceObservationAnalyticsService(
            self.repository
        )

    def default_metadata_filter(self) -> dict[str, Any]:
        """Restrict retrieval to decision-memory records by default."""
        return {"category": "decision_memory"}

    def build_query(self, task: "DecisionTask") -> str:
        """Build a retrieval query from the orchestrated analyst payload."""
        query_parts: list[str] = [task.subject.strip()]
        if task.symbol:
            query_parts.append(task.symbol.strip())
        if task.overall_summary:
            query_parts.append(task.overall_summary.strip())
        if task.key_signals:
            query_parts.append("signals " + " ".join(task.key_signals[:3]))
        if task.portfolio_risks:
            query_parts.append("risks " + " ".join(task.portfolio_risks[:3]))
        if task.cross_analyst_observations:
            query_parts.append("observations " + " ".join(task.cross_analyst_observations[:2]))
        if task.extra_context:
            query_parts.append(task.extra_context.strip())
        return " ".join(part for part in query_parts if part)

    def build_scenario_profile(self, task: "DecisionTask") -> dict[str, Any]:
        """Derive structured retrieval hints from the current analyst payload."""
        signal_texts = list(task.key_signals)
        risk_texts = list(task.portfolio_risks)
        for analyst_result in task.analyst_results:
            if not isinstance(analyst_result, dict):
                continue
            signal_texts.extend(str(item).strip() for item in analyst_result.get("signals", []))
            risk_texts.extend(str(item).strip() for item in analyst_result.get("risks", []))

        combined_text = " ".join(
            part
            for part in (
                task.subject,
                task.extra_context or "",
                task.overall_summary,
                " ".join(task.cross_analyst_observations),
            )
            if part
        )
        return {
            "symbol": task.symbol,
            "market_regime": self._infer_market_regime(combined_text),
            "analyst_alignment": self._infer_analyst_alignment(task),
            "signal_tags": self._extract_tags(
                signal_texts,
                tag_map={
                    "news_catalyst": ("guidance", "catalyst", "news", "headline"),
                    "sentiment_spike": ("sentiment", "hype", "attention", "buzz", "social"),
                    "momentum": ("momentum", "breakout", "trend", "acceleration"),
                    "ai_theme": ("ai", "artificial intelligence", "infrastructure"),
                    "price_extension": ("extended", "extension", "overbought", "high"),
                },
            ),
            "risk_tags": self._extract_tags(
                risk_texts,
                tag_map={
                    "crowded_trade": ("crowded", "overowned", "consensus", "crowded trade"),
                    "event_fade": ("event fade", "fade", "post catalyst", "post-catalyst"),
                    "valuation_risk": ("valuation", "expensive", "multiple", "rich"),
                    "execution_risk": ("execution", "delivery", "miss"),
                    "drawdown_risk": ("drawdown", "reversal", "pullback", "volatility"),
                },
            ),
            "timing_tags": self._extract_tags(
                [combined_text],
                tag_map={
                    "short_term": ("short-term", "near-term"),
                    "event_window": ("event", "earnings", "guidance", "catalyst"),
                    "near_local_high": ("high", "extended", "peak"),
                    "post_gap_up": ("gap", "gap-up"),
                },
            ),
            "portfolio_state_tags": self._infer_portfolio_state_tags(task),
        }

    def build_metadata_filter(
        self,
        *,
        metadata_filter: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Merge default and caller-specific metadata filters."""
        merged_filter = dict(self.default_metadata_filter())
        if metadata_filter:
            merged_filter.update(metadata_filter)
        return merged_filter

    def retrieve_context(
        self,
        task: "DecisionTask",
        *,
        query: str,
        scenario_profile: dict[str, Any],
        guidance_priors: dict[str, Any] | None = None,
        datasets: tuple[DatasetName, ...] | None = None,
        metadata_filter: dict[str, Any] | None = None,
        k: int | None = None,
    ) -> dict[str, Any]:
        """Fetch and rank decision-memory documents for the current task."""
        selected_datasets = datasets or self.default_datasets
        merged_filter = self.build_metadata_filter(metadata_filter=metadata_filter)
        candidate_count = max((k or self.default_k) * 4, 8)

        search_candidates = self.retriever.search(
            query,
            datasets=selected_datasets,
            k=candidate_count,
            metadata_filter=merged_filter or None,
        )
        fallback_candidates = [
            document
            for document in self.retriever.load_all_documents(selected_datasets)
            if self._matches_metadata_filter(getattr(document, "metadata", {}), merged_filter)
        ]
        candidates = self._dedupe_documents([*search_candidates, *fallback_candidates])
        validations: list[dict[str, Any]] = []
        validated_candidates: list[tuple[Any, dict[str, Any]]] = []
        for candidate in candidates:
            validation = self._validate_candidate(
                candidate,
                allowed_datasets=selected_datasets,
            )
            validations.append(validation)
            if not validation["is_valid"]:
                continue
            validated_candidates.append((candidate, validation))

        ranked_documents = [
            self._build_ranked_document(
                document,
                query=query,
                scenario_profile=scenario_profile,
                guidance_priors=guidance_priors or {},
                task=task,
                validation=validation,
            )
            for document, validation in validated_candidates
        ]
        ranked_documents.sort(
            key=lambda item: (
                item["score"],
                item["fit_rank"],
                item["metadata_quality"],
            ),
            reverse=True,
        )
        return {
            "ranked_documents": ranked_documents[: k or self.default_k],
            "validation_summary": summarize_decision_memory_validation(validations),
        }

    def analyze(
        self,
        task: "DecisionTask",
        *,
        datasets: tuple[DatasetName, ...] | None = None,
        metadata_filter: dict[str, Any] | None = None,
        k: int | None = None,
    ) -> dict[str, Any]:
        """Return a structured decision-memory payload for the advisory agent."""
        selected_datasets = datasets or self.default_datasets
        query = self.build_query(task)
        scenario_profile = self.build_scenario_profile(task)
        guidance_priors = self.collect_guidance_priors(task, datasets=selected_datasets)
        retrieval_context = self.retrieve_context(
            task,
            query=query,
            scenario_profile=scenario_profile,
            guidance_priors=guidance_priors,
            datasets=selected_datasets,
            metadata_filter=metadata_filter,
            k=k,
        )
        return self.build_context(
            task,
            query=query,
            scenario_profile=scenario_profile,
            datasets=selected_datasets,
            ranked_documents=retrieval_context["ranked_documents"],
            validation_summary=retrieval_context["validation_summary"],
            guidance_priors=guidance_priors,
        )

    def build_context(
        self,
        task: "DecisionTask",
        *,
        query: str,
        scenario_profile: dict[str, Any],
        datasets: tuple[DatasetName, ...],
        ranked_documents: list[dict[str, Any]],
        validation_summary: dict[str, Any],
        guidance_priors: dict[str, Any],
    ) -> dict[str, Any]:
        """Build an agent-friendly context payload from retrieved decision records."""
        serialized_documents = [self.serialize_document(item) for item in ranked_documents]
        return {
            "agent": self.agent_name,
            "subject": task.subject,
            "symbol": task.symbol,
            "trade_date": task.trade_date,
            "query": query,
            "scenario_profile": scenario_profile,
            "datasets": list(datasets),
            "document_count": len(serialized_documents),
            "validation_summary": validation_summary,
            "documents": serialized_documents,
            "evidence": self.collect_evidence(serialized_documents),
            "postmortem_lessons": self.collect_postmortem_lessons(serialized_documents),
            "guidance_priors": guidance_priors,
        }

    def serialize_document(self, ranked_document: dict[str, Any]) -> dict[str, Any]:
        """Convert a ranked document into a decision-service friendly payload."""
        document = ranked_document["document"]
        metadata = normalize_decision_memory_metadata(getattr(document, "metadata", {}))
        metadata["retrieval_score"] = ranked_document["score"]
        metadata["fit"] = ranked_document["fit"]
        metadata["match_reasons"] = ranked_document["match_reasons"]
        metadata["validation_warnings"] = ranked_document["validation"].get("warnings", [])
        return {
            "title": metadata.get("title", ""),
            "text": getattr(document, "page_content", ""),
            "metadata": metadata,
            "fit": ranked_document["fit"],
            "match_reasons": ranked_document["match_reasons"],
        }

    def collect_evidence(self, documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Normalize serialized documents into compact evidence entries."""
        evidence: list[dict[str, Any]] = []
        for document in documents:
            metadata = dict(document.get("metadata", {}))
            evidence.append(
                {
                    "source_type": metadata.get("source_type", "internal"),
                    "title": document.get("title", ""),
                    "content": self.build_excerpt(document.get("text", "")),
                    "metadata": metadata,
                    "fit": document.get("fit", metadata.get("fit", "low")),
                }
            )
        return evidence

    def collect_postmortem_lessons(self, documents: list[dict[str, Any]]) -> list[dict[str, str]]:
        """Extract reusable lesson snippets from retrieved postmortem memories."""
        lessons: list[dict[str, str]] = []
        for document in documents:
            metadata = dict(document.get("metadata", {}))
            memory_type = str(metadata.get("memory_type", "")).strip().lower()
            if memory_type != "decision_postmortem":
                continue
            for lesson in self._extract_postmortem_sections(document):
                lessons.append(
                    {
                        "title": str(document.get("title", "")).strip() or "Untitled postmortem",
                        "fit": str(document.get("fit", metadata.get("fit", "medium"))).strip().lower()
                        or "medium",
                        "lesson": lesson,
                    }
                )
        return lessons[:4]

    def collect_guidance_priors(
        self,
        task: "DecisionTask",
        *,
        datasets: tuple[DatasetName, ...],
    ) -> dict[str, Any]:
        """Summarize recurring guidance usage for the current symbol as bounded priors."""
        if not task.symbol:
            return {
                "datasets": list(datasets),
                "symbol": None,
                "total_observations": 0,
                "top_guidance": [],
                "recommendation_breakdown": [],
                "top_reference_cases": [],
                "summary": "",
            }
        return self.observation_analytics.summarize_guidance_priors(
            datasets=datasets,
            symbol=task.symbol,
            top_n=3,
        )

    def build_excerpt(self, text: str, *, limit: int = 280) -> str:
        """Return a compact evidence excerpt suitable for prompts."""
        compact_text = " ".join(text.split())
        if len(compact_text) <= limit:
            return compact_text
        return compact_text[: limit - 3].rstrip() + "..."

    def _extract_tags(
        self,
        texts: list[str],
        *,
        tag_map: dict[str, tuple[str, ...]],
    ) -> list[str]:
        """Infer normalized tags from free-form analyst text."""
        combined = " ".join(text.lower() for text in texts if text)
        return [
            tag
            for tag, keywords in tag_map.items()
            if any(keyword in combined for keyword in keywords)
        ]

    def _infer_market_regime(self, text: str) -> str:
        """Infer a coarse market-regime tag from the current task text."""
        lowered = text.lower()
        if any(
            keyword in lowered
            for keyword in ("earnings", "guidance", "catalyst", "event-driven", "event")
        ):
            return "event_driven"
        if any(keyword in lowered for keyword in ("risk-off", "defensive", "drawdown", "de-risk")):
            return "risk_off"
        if any(keyword in lowered for keyword in ("range", "sideways", "mean reversion")):
            return "range_bound"
        if any(keyword in lowered for keyword in ("momentum", "trend", "breakout")):
            return "trend_following"
        return "mixed"

    def _infer_analyst_alignment(self, task: "DecisionTask") -> str:
        """Infer the degree of cross-analyst agreement."""
        conflict_markers = ("disagree", "conflict", "mixed", "diverge", "uncertain")
        observations = " ".join(task.cross_analyst_observations).lower()
        if any(marker in observations for marker in conflict_markers):
            return "conflicted"
        if str(task.overall_confidence).strip().lower() == "high":
            return "aligned"
        return "mixed"

    def _infer_portfolio_state_tags(self, task: "DecisionTask") -> list[str]:
        """Infer coarse portfolio-state tags from current holdings and limits."""
        portfolio_context = task.portfolio_context or {}
        tags: list[str] = []
        positions = portfolio_context.get("positions", [])
        if isinstance(positions, list) and positions:
            tags.append("has_positions")

        current_position = self._find_symbol_position(portfolio_context, task.symbol)
        current_weight = self._extract_position_weight(current_position)
        max_weight = self._extract_max_single_name_pct(portfolio_context)
        cash_pct = self._extract_percent(portfolio_context.get("cash_pct"))

        if current_position is not None:
            tags.append("existing_position")
        else:
            tags.append("no_position")

        if current_weight is not None and max_weight is not None:
            if current_weight > max_weight:
                tags.append("above_single_name_limit")
            elif current_weight >= max_weight * 0.9:
                tags.append("near_single_name_limit")

        if cash_pct is not None:
            if cash_pct >= 10:
                tags.append("ample_cash")
            elif cash_pct < 5:
                tags.append("limited_cash")

        return tags

    def _dedupe_documents(self, documents: Iterable[Any]) -> list[Any]:
        """Remove duplicate candidate documents while preserving first-seen order."""
        deduped: list[Any] = []
        seen_keys: set[tuple[str, str]] = set()
        for document in documents:
            metadata = dict(getattr(document, "metadata", {}))
            key = (
                str(metadata.get("title", "")).strip(),
                str(metadata.get("created_at", "")).strip(),
            )
            if key in seen_keys:
                continue
            seen_keys.add(key)
            deduped.append(document)
        return deduped

    def _build_ranked_document(
        self,
        document: Any,
        *,
        query: str,
        scenario_profile: dict[str, Any],
        guidance_priors: dict[str, Any],
        task: "DecisionTask",
        validation: dict[str, Any],
    ) -> dict[str, Any]:
        """Score a candidate decision-memory document against the current task."""
        metadata = dict(validation.get("normalized_metadata", {}))
        score = 0.0
        match_reasons: list[str] = []
        primary_identity_match = False
        structured_overlap_count = 0

        text_score = self._text_overlap_score(query=query, document=document)
        score += text_score
        if text_score > 0:
            match_reasons.append("textual overlap with the current analyst synthesis")

        if task.symbol and str(metadata.get("symbol", "")).strip().upper() == task.symbol.upper():
            score += 6.0
            match_reasons.append("same symbol")
            primary_identity_match = True

        metadata_subject = str(metadata.get("subject", "")).lower()
        if task.subject and task.subject.lower() in metadata_subject:
            score += 3.0
            match_reasons.append("similar subject framing")
            primary_identity_match = True

        market_regime = str(metadata.get("market_regime", "")).strip().lower()
        if market_regime and market_regime == str(scenario_profile.get("market_regime", "")).lower():
            score += 3.0
            match_reasons.append("matching market regime")
            structured_overlap_count += 1

        analyst_alignment = str(metadata.get("analyst_alignment", "")).strip().lower()
        if analyst_alignment and analyst_alignment == str(
            scenario_profile.get("analyst_alignment", "")
        ).lower():
            score += 2.0
            match_reasons.append("similar analyst alignment")
            structured_overlap_count += 1

        signal_overlap = self._metadata_overlap(
            scenario_profile.get("signal_tags", []),
            metadata,
            field_names=("signal_tags", "tags"),
        )
        if signal_overlap:
            score += 2.0 * len(signal_overlap)
            match_reasons.append(f"shared signal tags: {', '.join(signal_overlap)}")
            structured_overlap_count += len(signal_overlap)

        risk_overlap = self._metadata_overlap(
            scenario_profile.get("risk_tags", []),
            metadata,
            field_names=("risk_tags", "tags"),
        )
        if risk_overlap:
            score += 2.0 * len(risk_overlap)
            match_reasons.append(f"shared risk tags: {', '.join(risk_overlap)}")
            structured_overlap_count += len(risk_overlap)

        timing_overlap = self._metadata_overlap(
            scenario_profile.get("timing_tags", []),
            metadata,
            field_names=("timing_tags", "tags"),
        )
        if timing_overlap:
            score += 1.5 * len(timing_overlap)
            match_reasons.append(f"shared timing tags: {', '.join(timing_overlap)}")
            structured_overlap_count += len(timing_overlap)

        portfolio_state_overlap = self._metadata_overlap(
            scenario_profile.get("portfolio_state_tags", []),
            metadata,
            field_names=("portfolio_state_tags", "tags"),
        )
        if portfolio_state_overlap:
            score += 2.5 * len(portfolio_state_overlap)
            match_reasons.append(
                f"shared portfolio state tags: {', '.join(portfolio_state_overlap)}"
            )
            structured_overlap_count += len(portfolio_state_overlap)

        source_type = str(metadata.get("source_type", "")).strip().lower()
        if source_type == "internal":
            score += 0.5

        outcome_label = str(metadata.get("outcome_label", "")).strip().lower()
        if outcome_label == "worked":
            score += 0.5

        memory_type = str(metadata.get("memory_type", "")).strip().lower()
        if memory_type == "decision_postmortem":
            score += 0.75
            match_reasons.append("postmortem memory with reusable review lessons")

        guidance_alignment_score = self._guidance_prior_alignment_score(
            document=document,
            guidance_priors=guidance_priors,
        )
        if guidance_alignment_score > 0:
            score += guidance_alignment_score
            match_reasons.append("aligned with recurring guidance priors for this symbol")

        metadata_quality = self._safe_float(metadata.get("quality_score")) or 0.0
        score += min(metadata_quality, 1.0) * 2.0

        fit = self._score_to_fit(
            score,
            primary_identity_match=primary_identity_match,
            structured_overlap_count=structured_overlap_count,
        )
        return {
            "document": document,
            "score": round(score, 3),
            "fit": fit,
            "fit_rank": {"high": 3, "medium": 2, "low": 1}[fit],
            "match_reasons": match_reasons or ["fallback decision-memory reference"],
            "metadata_quality": metadata_quality,
            "validation": validation,
        }

    def _validate_candidate(
        self,
        document: Any,
        *,
        allowed_datasets: Iterable[DatasetName] | None = None,
    ) -> dict[str, Any]:
        """Validate a candidate document before it participates in ranking."""
        record = {
            "text": getattr(document, "page_content", ""),
            "metadata": dict(getattr(document, "metadata", {})),
        }
        return validate_decision_memory_record(
            record,
            allowed_datasets=allowed_datasets,
        )

    def _text_overlap_score(self, *, query: str, document: Any) -> float:
        """Compute a small lexical overlap score for the current query."""
        stopwords = {
            "current",
            "analyst",
            "analysis",
            "evidence",
            "supports",
            "stance",
            "subject",
            "symbol",
            "setup",
            "with",
            "from",
            "that",
            "this",
            "into",
        }
        query_terms = {
            term
            for term in self._tokenize(query)
            if len(term) > 3 and term not in stopwords
        }
        metadata = dict(getattr(document, "metadata", {}))
        haystack = " ".join(
            [
                getattr(document, "page_content", ""),
                str(metadata.get("title", "")),
                str(metadata.get("subject", "")),
                " ".join(str(tag) for tag in metadata.get("tags", [])),
                " ".join(str(tag) for tag in metadata.get("signal_tags", [])),
                " ".join(str(tag) for tag in metadata.get("risk_tags", [])),
                " ".join(str(tag) for tag in metadata.get("timing_tags", [])),
            ]
        ).lower()
        haystack_terms = set(self._tokenize(haystack))
        return float(min(sum(1 for term in query_terms if term in haystack_terms), 4))

    def _guidance_prior_alignment_score(
        self,
        *,
        document: Any,
        guidance_priors: dict[str, Any],
    ) -> float:
        """Apply a small boost when a document aligns with recurring guidance priors."""
        top_guidance = guidance_priors.get("top_guidance", [])
        if not isinstance(top_guidance, list) or not top_guidance:
            return 0.0

        metadata = dict(getattr(document, "metadata", {}))
        haystack = " ".join(
            [
                getattr(document, "page_content", ""),
                str(metadata.get("title", "")),
                str(metadata.get("subject", "")),
                " ".join(str(tag) for tag in metadata.get("tags", [])),
                " ".join(str(tag) for tag in metadata.get("signal_tags", [])),
                " ".join(str(tag) for tag in metadata.get("risk_tags", [])),
            ]
        ).lower()
        haystack_terms = set(self._tokenize(haystack))

        overlap_count = 0
        for item in top_guidance[:2]:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label", "")).strip().lower()
            if not label:
                continue
            guidance_terms = {
                term
                for term in self._tokenize(label)
                if len(term) > 4 and term not in {"before", "after", "prior", "decisions"}
            }
            overlap_count += sum(1 for term in guidance_terms if term in haystack_terms)

        if overlap_count >= 4:
            return 1.5
        if overlap_count >= 2:
            return 0.75
        return 0.0

    def _metadata_overlap(
        self,
        target_tags: list[str],
        metadata: dict[str, Any],
        *,
        field_names: tuple[str, ...],
    ) -> list[str]:
        """Return overlapping tags between the scenario profile and document metadata."""
        metadata_tags: set[str] = set()
        for field_name in field_names:
            raw_value = metadata.get(field_name, [])
            if isinstance(raw_value, list):
                metadata_tags.update(
                    str(item).strip().lower() for item in raw_value if str(item).strip()
                )
        return [tag for tag in target_tags if tag.lower() in metadata_tags]

    def _score_to_fit(
        self,
        score: float,
        *,
        primary_identity_match: bool,
        structured_overlap_count: int,
    ) -> str:
        """Convert a numeric retrieval score into a fit label."""
        if score >= 11 and primary_identity_match and structured_overlap_count >= 2:
            return "high"
        if score >= 5:
            return "medium"
        return "low"

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text into lowercase alphanumeric terms."""
        return re.findall(r"[a-z0-9_+-]+", text.lower())

    def _extract_postmortem_sections(self, document: dict[str, Any]) -> list[str]:
        """Extract lesson-like bullet points from a serialized postmortem document."""
        text = str(document.get("text", "")).strip()
        if not text:
            return []

        sections: list[str] = []
        for heading in ("Reusable lessons:", "Future adjustments:"):
            pattern = rf"{re.escape(heading)}\n((?:- .+\n?){{1,6}})"
            match = re.search(pattern, text)
            if not match:
                continue
            block = match.group(1)
            for line in block.splitlines():
                normalized = line.removeprefix("- ").strip()
                if normalized:
                    sections.append(normalized)
        return sections[:3]

    def _matches_metadata_filter(
        self,
        metadata: dict[str, Any],
        metadata_filter: dict[str, Any] | None,
    ) -> bool:
        """Apply exact-match filtering over document metadata."""
        if not metadata_filter:
            return True
        for key, expected_value in metadata_filter.items():
            if metadata.get(key) != expected_value:
                return False
        return True

    def _safe_float(self, value: Any) -> float | None:
        """Convert optional numeric-like values into floats."""
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

    def _find_symbol_position(
        self,
        portfolio_context: dict[str, Any],
        symbol: str | None,
    ) -> dict[str, Any] | None:
        """Return the current position object for the requested symbol if present."""
        if not symbol:
            return None
        positions = portfolio_context.get("positions", [])
        if not isinstance(positions, list):
            return None
        target_symbol = symbol.strip().upper()
        for position in positions:
            if not isinstance(position, dict):
                continue
            if str(position.get("symbol", "")).strip().upper() == target_symbol:
                return position
        return None

    def _extract_position_weight(self, position: dict[str, Any] | None) -> float | None:
        """Extract a normalized percent weight from a position object."""
        if not isinstance(position, dict):
            return None
        for field_name in ("weight_pct", "weight"):
            value = self._extract_percent(position.get(field_name))
            if value is not None:
                return value
        return None

    def _extract_max_single_name_pct(self, portfolio_context: dict[str, Any]) -> float | None:
        """Extract the active single-name limit if available."""
        direct_value = self._extract_percent(portfolio_context.get("max_single_name_pct"))
        if direct_value is not None:
            return direct_value
        position_limits = portfolio_context.get("position_limits")
        if isinstance(position_limits, dict):
            return self._extract_percent(position_limits.get("max_single_name_pct"))
        return None

    def _extract_percent(self, value: Any) -> float | None:
        """Parse numeric percent-like values into floats."""
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
