from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from sci_response.data.io import load_yaml


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def agent_config_root() -> Path:
    return repo_root() / "configs" / "agent"


def default_llm_config_path() -> Path:
    return agent_config_root() / "llm.yaml"


def default_harness_config_path() -> Path:
    return agent_config_root() / "harness.yaml"


def default_harness_config_v2_path() -> Path:
    return agent_config_root() / "harness_v2.yaml"


def default_prompt_config_path() -> Path:
    return agent_config_root() / "prompts" / "mechanism_fidelity.yaml"


def default_prompt_config_v2_path() -> Path:
    return agent_config_root() / "prompts" / "mechanism_fidelity_v2.yaml"


def default_direct_code_prompt_config_path() -> Path:
    return agent_config_root() / "prompts" / "direct_code_llm.yaml"


def default_mechanism_template_path() -> Path:
    return agent_config_root() / "mechanisms" / "template_library.yaml"


def default_mechanism_template_v2_path() -> Path:
    return agent_config_root() / "mechanisms" / "template_library_v2.yaml"


def default_skill_card_path() -> Path:
    return agent_config_root() / "skills" / "mechanism_fidelity.md"


def default_skill_card_v2_path() -> Path:
    return agent_config_root() / "skills" / "mechanism_fidelity_v2.md"


def default_direct_code_skill_card_path() -> Path:
    return agent_config_root() / "skills" / "direct_code_llm.md"


def default_agent_defaults_path() -> Path:
    return agent_config_root() / "defaults.yaml"


def load_harness_config(path: Path | None = None) -> Dict[str, Any]:
    target = path.resolve() if path is not None else default_harness_config_path().resolve()
    return load_yaml(target)


def load_agent_defaults(path: Path | None = None) -> Dict[str, Any]:
    target = path.resolve() if path is not None else default_agent_defaults_path().resolve()
    return load_yaml(target)
