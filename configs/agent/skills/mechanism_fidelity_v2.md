# Mechanism-Faithful Search V2

- Search over structured biological modeling hypotheses, not free-form code.
- Prefer edits that match the dataset family:
  - single-cell CRISPR: baseline anchoring, delta prediction, zero-init stabilization
  - dose/time drug response: conditioning operator, delta-vs-response choice, loss mixing
  - paired clinical response: baseline anchoring, paired-state conditioning, conservative optimization
- In early search stages, prioritize mechanism and capacity edits before optimizer-only edits.
- Use cross-dataset memory only as a prior, not as a hard constraint.
- Favor compact edit programs that change one mechanistic claim at a time.
