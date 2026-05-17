# Analyst Prompts

This directory packages the analyst prompt assets used by the runtime.

- `shared/` stores prompt instructions shared by all analysts.
- `roles/` stores analyst-specific prompt bodies.
- `orchestration/` stores top-level aggregation prompts used after analysts finish.

Edit the text files in `roles/` when you want to tune individual analysts
without changing the runtime code.
