# Decision Memory Template

Decision-memory records live as processed knowledge JSON files and use the normal project shape:

```json
{
  "text": "Historical decision narrative, recommendation, reasoning, risks, and lessons.",
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
    "dataset": "dynamic"
  }
}
```

Required metadata fields:
- `title`
- `category`
- `memory_type`
- `source_type`
- `subject`
- `recommendation`
- `confidence`
- `dataset`

Recommended metadata fields:
- `symbol`
- `market_regime`
- `analyst_alignment`
- `signal_tags`
- `risk_tags`
- `timing_tags`
- `portfolio_state_tags`
- `outcome_label`
- `quality_score`

Field intent:
- `memory_type`: `decision_case`, `decision_postmortem`, or `external_reference_decision`
- `source_type`: `internal` or `external`
- `signal_tags`: what constructive setup elements were present
- `risk_tags`: what decision-relevant risks mattered
- `timing_tags`: when the setup happened or what timing window mattered
- `portfolio_state_tags`: what the portfolio looked like when the advice was formed, such as `no_position`, `existing_position`, `near_single_name_limit`, `ample_cash`, or `limited_cash`
- `quality_score`: optional 0-1 prior on how much this memory should influence later retrieval

Authoring guidance:
- Put the reusable narrative in `text`, not only in metadata.
- Keep tags normalized and lowercase with underscores where possible.
- Use `quality_score` to down-rank weak or noisy cases instead of deleting everything.
- Prefer `internal` records over `external` ones when both exist for a similar setup.

Validation behavior:
- Invalid decision-memory records are skipped during retrieval instead of being silently normalized into use.
- Retrieval context now carries a `validation_summary` so upstream debugging can see how many candidates were filtered out and why.
- Validation warnings do not block retrieval, but they are attached to serialized document metadata as `validation_warnings`.
- `validation_summary` separates `valid_warning_candidates` from `invalid_warning_candidates` so review can distinguish usable-but-noisy records from fully rejected ones.
