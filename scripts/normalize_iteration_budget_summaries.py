#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(ROOT / "src"))

from sci_response.agent.formal import normalize_agent_result_row
from sci_response.data.io import dump_json, load_json
from sci_response.final_results import refresh_all_final_results


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _rewrite_rows_file(json_path: Path, *, max_iteration: int = 10) -> bool:
    payload = load_json(json_path)
    rows = list(payload.get("rows", []))
    if not rows:
        return False
    normalized_rows = [normalize_agent_result_row(dict(row), max_iteration=max_iteration) for row in rows]
    if normalized_rows == rows:
        return False
    dump_json(json_path, {"rows": normalized_rows})
    csv_path = json_path.with_suffix(".csv")
    _write_csv(csv_path, normalized_rows)
    return True


def _candidate_json_paths(root: Path) -> Iterable[Path]:
    for pattern in (
        "datasets/*/latest/agent_pack_summary.json",
        "datasets/*/latest/per_seed_rows.json",
        "datasets/*/pack_runs/*/agent_pack_summary.json",
        "datasets/*/pack_runs/*/per_seed_rows.json",
        "global/agent_summary.json",
        "global/history/*/agent_summary.json",
        "direct_code_llm/global/direct_code_summary.json",
        "direct_code_llm/global/history/*/direct_code_summary.json",
    ):
        yield from root.glob(pattern)


def main() -> None:
    roots = [
        path
        for pattern in ("agent_runs*", "direct_code_runs*", "random_edit_runs*", "hpo_runs*")
        for path in sorted(ROOT.glob(pattern))
        if path.is_dir()
    ]
    touched_roots: set[Path] = set()
    rewritten = 0
    for root in roots:
        for json_path in _candidate_json_paths(root):
            if _rewrite_rows_file(json_path):
                rewritten += 1
                touched_roots.add(root)
                print(f"rewrote {json_path}", flush=True)
    for root in sorted(touched_roots):
        refresh_all_final_results(repo_root=ROOT, updated_branch_root=root)
        print(f"refreshed final results for {root}", flush=True)
    print(f"rewritten_files={rewritten}", flush=True)


if __name__ == "__main__":
    main()
