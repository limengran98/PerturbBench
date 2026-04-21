from __future__ import annotations

from typing import Dict

from . import cellot_wrapper, cpa_wrapper, gears_wrapper, gperturb_wrapper, transigen_wrapper, xpert_wrapper


WRAPPERS: Dict[str, object] = {
    "cellot": cellot_wrapper,
    "cpa": cpa_wrapper,
    "gears": gears_wrapper,
    "gperturb": gperturb_wrapper,
    "transigen": transigen_wrapper,
    "xpert": xpert_wrapper,
}


def normalize_baseline_name(name: str) -> str:
    return str(name).strip().lower().replace("-", "").replace("_", "")


def get_wrapper(name: str) -> object:
    normalized = normalize_baseline_name(name)
    aliases = {
        "gears": "gears",
        "cpa": "cpa",
        "cellot": "cellot",
        "gperturb": "gperturb",
        "xpert": "xpert",
        "transigen": "transigen",
    }
    if normalized not in aliases:
        raise KeyError(f"Unsupported specialist baseline: {name}")
    return WRAPPERS[aliases[normalized]]
