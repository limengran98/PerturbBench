#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

COMMAND_TO_SCRIPT = {
    "prepare": "prepare_dataset.py",
    "split": "make_splits.py",
    "benchmark": "run_benchmark.py",
    "baseline": "run_baseline.py",
    "experiment": "run_experiment.py",
    "specialist": "run_specialist.py",
    "agent": "run_agent.py",
}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Unified entrypoint for data, baselines, models, and agent workflows.")
    parser.add_argument("command", choices=sorted(COMMAND_TO_SCRIPT.keys()))
    parsed, passthrough = parser.parse_known_args()

    script_path = ROOT / "scripts" / COMMAND_TO_SCRIPT[parsed.command]
    command = [sys.executable, str(script_path), *passthrough]
    completed = subprocess.run(command, cwd=str(ROOT))
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
