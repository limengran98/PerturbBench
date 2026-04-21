"""Agent bridge modules for model-changing workflows."""

from sci_response.agent.codeevo import build_codeevo_command, list_codeevo_configs, resolve_codeevo_root
from sci_response.agent.direct_code import run_direct_code_model_session
from sci_response.agent.formal import (
    default_agent_root,
    default_session_id,
    resolve_dataset_keys,
    resolve_method_names,
    run_method_session,
    update_global_agent_summary,
    write_dataset_pack_summary,
)
from sci_response.agent.hpo import run_hpo_model_session
from sci_response.agent.random_edit import run_random_edit_model_session
from sci_response.agent.structured import run_structured_model_session
from sci_response.agent.structured_v2 import run_structured_model_session_v2
from sci_response.agent.structured_v3 import run_structured_model_session_v3
from sci_response.agent.structured_v4 import run_structured_model_session_v4
from sci_response.agent.structured_v5 import run_structured_model_session_v5
from sci_response.agent.structured_v6 import run_structured_model_session_v6
from sci_response.agent.structured_v7 import run_structured_model_session_v7
from sci_response.agent.structured_v8 import run_structured_model_session_v8

__all__ = [
    "build_codeevo_command",
    "list_codeevo_configs",
    "resolve_codeevo_root",
    "default_agent_root",
    "default_session_id",
    "resolve_dataset_keys",
    "resolve_method_names",
    "run_method_session",
    "run_direct_code_model_session",
    "run_hpo_model_session",
    "run_random_edit_model_session",
    "run_structured_model_session",
    "run_structured_model_session_v2",
    "run_structured_model_session_v3",
    "run_structured_model_session_v4",
    "run_structured_model_session_v5",
    "run_structured_model_session_v6",
    "run_structured_model_session_v7",
    "run_structured_model_session_v8",
    "update_global_agent_summary",
    "write_dataset_pack_summary",
]
