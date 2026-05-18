# Decision Prompts

This directory holds only prompt assets for the decision layer.

## Layout

- `shared/`
  Cross-agent prompt rules that apply to every decision agent.
- `roles/`
  Role-specific prompt bodies such as `decision_advisory.txt`.

## Intended Use

Prompt edits here should change tone, framing, or behavioral guidance.
They should not redefine:
- memory schema
- retrieval ranking
- Python output normalization

If a prompt change needs a new field or a new validation rule, update the
relevant Python contract in `agents/decision` or `services/decision/memory`
instead of encoding structure only in text.
