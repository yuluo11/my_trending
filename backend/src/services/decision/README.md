# Decision Services

This package is intentionally narrow.

The decision layer keeps agent reasoning outside this directory. Services here
exist to support the agent with reusable infrastructure, especially around
decision-memory retrieval quality.

## Current Scope

- `memory/knowledge_service.py`
  Retrieves and ranks historical decision-memory documents.
- `memory/schema.py`
  Standardizes and validates decision-memory records.
- `memory/README.md`
  Documents the processed record shape for authoring and review.

## Boundary

Keep these concerns here:
- memory schema
- memory validation
- retrieval filters
- retrieval ranking
- retrieval diagnostics

Keep these concerns out of here:
- recommendation logic
- fallback decision text
- prompt assembly
- final decision-output formatting

Those belong in `backend/src/agents/decision/`.
