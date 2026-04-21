# Runtime Environment Definitions

These files are now **fallback environment definitions**, not the primary execution plan.

The primary goal of this repo is:

- one shared runtime
- one unified benchmark interface
- one execution substrate for future agent-based hypothesis search

So these env files exist only for:

- strict-upstream reproduction
- debugging
- temporary comparison while shared-runtime integrations are still incomplete

## Primary Runtime

| Runtime | Role |
|---|---|
| `shared` | default benchmark runtime, default product runtime, and the target runtime for adapted specialist baselines |

## Fallback Runtime Map

| Env group | Baselines | Interpreter target | Setup file | Why it is fallback-only |
|---|---|---|---|---|
| `sc_gears_cpa` | `gears`, `cpa` | Python `3.10` | `envs/sc_gears_cpa.requirements.txt` | useful for strict-upstream comparison, but not intended as the permanent user-facing runtime |
| `sc_cellot` | `cellot` | Python `3.9.5` | `envs/sc_cellot.requirements.txt` | old stack; better long-term target is in-framework implementation |
| `drug_xpert` | `xpert` | Python `3.9` | `envs/drug_xpert.requirements.txt` | temporary upstream fallback while shared-runtime adaptation is incomplete |
| `drug_transigen` | `transigen` | Python `3.6.13` | `envs/drug_transigen.environment.yml` | legacy stack; strong candidate for reimplementation rather than permanent inheritance |

## Setup Commands

Dry-run a fallback setup plan:

```bash
python3 scripts/setup_runtime_env.py --env-group sc_gears_cpa --python-executable /path/to/python3.10
python3 scripts/setup_runtime_env.py --env-group drug_xpert --python-executable /path/to/python3.9
python3 scripts/setup_runtime_env.py --env-group drug_transigen
```

Check current runtime resolution:

```bash
python3 scripts/check_runtime_envs.py
python3 scripts/check_runtime_envs.py --run-preflight
```

## Important Reading

- If a baseline can be integrated into the shared runtime, prefer that.
- If a baseline cannot be integrated cleanly, decide whether it deserves:
  - a temporary fallback env
  - or an in-framework reimplementation
- Do not assume that preserving an upstream environment is always the right long-term choice.
