# Direct Code LLM Baseline

- This line is a deliberate control baseline, not the main structured agent.
- The LLM directly rewrites model code instead of proposing IR edits.
- It must still use the same dataset, split, metrics, benchmark runner, and artifact contract.
- It is allowed to change model internals and training code, but not data loading or evaluation code.
- Prefer runnable, stable code over aggressive novelty.
- Prefer torch so requested_device can be honored on large datasets.
- DirectCode-SingleShot: one code generation attempt per iteration.
- DirectCode-RepairLoop: after a failed attempt, re-prompt with the traceback for a bounded number of repairs.
