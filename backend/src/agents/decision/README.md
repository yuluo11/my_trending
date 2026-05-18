# Decision Agents

This package owns the advisory decision layer itself.

What belongs here:
- `DecisionTask` and decision-facing runtime state
- prompt assembly for decision agents
- advisory reasoning and fallback logic
- normalization of the final decision output contract

What does not belong here:
- knowledge-record schema rules
- decision-memory retrieval heuristics
- validation of processed memory documents

Those responsibilities stay in `backend/src/services/decision/memory/`.

## Files

- `base_agent.py`
  Shared decision-agent contract and the end-to-end advisory flow.
- `advisory_agent.py`
  Concrete advisory agent wired to the decision-memory service.
- `__init__.py`
  Stable import surface for app wiring.

## Debug Path

If the final recommendation looks wrong:
1. Check the `DecisionTask` values entering the agent.
2. Inspect `decision_context` in the returned payload.
3. Confirm prompt files under `backend/src/prompts/decision/`.
4. Only then inspect memory retrieval and validation behavior.

## Review Rule of Thumb

If a change affects how the agent reasons or what it returns, it belongs here.
If a change affects which historical cases are retrieved or whether they are valid, it belongs in `services/decision/memory`.
