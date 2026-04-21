#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sci_response.final_results import refresh_all_final_results
from sci_response.pathing import repo_relative_str


def main() -> None:
    refreshed = refresh_all_final_results(repo_root=ROOT)
    print(f"final_metric_result={repo_relative_str(refreshed['final_metric_result'])}")
    print(f"final_budget_result={repo_relative_str(refreshed['final_budget_result'])}")


if __name__ == "__main__":
    main()
