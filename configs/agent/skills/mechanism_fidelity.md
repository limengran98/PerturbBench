# Mechanism Fidelity Skill

Search over structural modeling hypotheses, not arbitrary implementation details.

Prioritize edits that correspond to an interpretable scientific claim:

- baseline identity preservation
- delta prediction versus direct response prediction
- conditioning operator choice
- residual refinement depth
- trunk capacity and expressivity
- SE-style channel reweighting
- zero-init stabilization
- loss mixing between response and delta
- optimizer regularization only when attached to a structural hypothesis

Reject proposals that:

- change data semantics
- change split semantics
- change metric definitions
- perform optimizer-only hacking without a modeling hypothesis
- introduce many unrelated edits in one step
- weaken auditability or explanation quality
