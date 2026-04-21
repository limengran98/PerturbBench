from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _safe_int(text: str) -> int:
    try:
        return int(text)
    except Exception as exc:
        raise ValueError(f"Invalid integer value: {text!r}") from exc


def parse_cuda_visible_devices(value: Optional[str]) -> Optional[Tuple[int, ...]]:
    if value is None:
        return None
    tokens = [token.strip() for token in str(value).split(",") if token.strip()]
    if not tokens:
        raise ValueError("--cuda-visible-devices must not be empty")
    parsed = tuple(_safe_int(token) for token in tokens)
    if any(item < 0 for item in parsed):
        raise ValueError("--cuda-visible-devices must contain non-negative GPU indices only")
    return parsed


def parse_device_request(device: str) -> Tuple[str, Optional[int]]:
    normalized = str(device).strip().lower()
    if normalized == "cpu":
        return "cpu", None
    if normalized == "cuda":
        return "cuda", None
    if normalized.startswith("cuda:"):
        index = _safe_int(normalized.split(":", 1)[1])
        if index < 0:
            raise ValueError("CUDA device index must be non-negative")
        return "cuda", index
    raise ValueError(f"Unsupported device request: {device!r}")


def _query_gpu_inventory() -> Tuple[List[Tuple[int, str]], Optional[str]]:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception as exc:
        return [], repr(exc)

    inventory: List[Tuple[int, str]] = []
    for line in completed.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = [part.strip() for part in stripped.split(",", 1)]
        if len(parts) != 2:
            continue
        inventory.append((_safe_int(parts[0]), parts[1]))
    return inventory, None


def _query_torch_cuda() -> Tuple[bool, int, List[str], Optional[str]]:
    try:
        import torch  # type: ignore
    except Exception as exc:
        return False, 0, [], repr(exc)

    try:
        available = bool(torch.cuda.is_available())
        count = int(torch.cuda.device_count()) if available else 0
        names = [str(torch.cuda.get_device_name(index)) for index in range(count)] if available else []
        return available, count, names, None
    except Exception as exc:
        return False, 0, [], repr(exc)


@dataclass(frozen=True)
class ResolvedDevice:
    requested_device: str
    resolved_device: str
    cuda_visible_devices: Optional[str]
    torch_cuda_available: bool
    gpu_name: Optional[str]
    gpu_count: int
    model_uses_gpu: bool
    model_supports_cuda: bool
    device_note: Optional[str]
    visible_gpu_indices: Tuple[int, ...]
    torch_cuda_error: Optional[str]
    nvidia_smi_error: Optional[str]

    def to_manifest_fields(self) -> Dict[str, Any]:
        return {
            "requested_device": self.requested_device,
            "resolved_device": self.resolved_device,
            "cuda_visible_devices": self.cuda_visible_devices,
            "torch_cuda_available": bool(self.torch_cuda_available),
            "gpu_name": self.gpu_name,
            "gpu_count": int(self.gpu_count),
            "model_uses_gpu": bool(self.model_uses_gpu),
            "model_supports_cuda": bool(self.model_supports_cuda),
            "device_note": self.device_note,
        }


def resolve_device(
    requested_device: str,
    cuda_visible_devices: Optional[str],
    *,
    model_supports_cuda: bool,
) -> ResolvedDevice:
    if cuda_visible_devices is not None:
        parsed_visible = parse_cuda_visible_devices(cuda_visible_devices)
        assert parsed_visible is not None
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(item) for item in parsed_visible)
        effective_cuda_visible_devices = os.environ["CUDA_VISIBLE_DEVICES"]
    else:
        effective_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        parsed_visible = parse_cuda_visible_devices(effective_cuda_visible_devices) if effective_cuda_visible_devices else None

    requested_kind, requested_index = parse_device_request(requested_device)
    gpu_inventory, nvidia_smi_error = _query_gpu_inventory()
    torch_cuda_available, torch_gpu_count, torch_gpu_names, torch_cuda_error = _query_torch_cuda()

    inventory_by_index = {index: name for index, name in gpu_inventory}
    if parsed_visible is not None:
        missing_visible = [index for index in parsed_visible if gpu_inventory and index not in inventory_by_index]
        if missing_visible:
            raise ValueError(
                f"CUDA_VISIBLE_DEVICES references unavailable GPU indices: {missing_visible}"
            )
        if gpu_inventory:
            visible_gpu_indices = parsed_visible
            visible_gpu_names = [inventory_by_index[index] for index in parsed_visible if index in inventory_by_index]
        elif torch_cuda_available:
            visible_gpu_indices = tuple(range(torch_gpu_count))
            visible_gpu_names = list(torch_gpu_names)
        else:
            visible_gpu_indices = tuple()
            visible_gpu_names = []
    elif gpu_inventory:
        visible_gpu_indices = tuple(index for index, _ in gpu_inventory)
        visible_gpu_names = [name for _, name in gpu_inventory]
    elif torch_cuda_available:
        visible_gpu_indices = tuple(range(torch_gpu_count))
        visible_gpu_names = list(torch_gpu_names)
    else:
        visible_gpu_indices = tuple()
        visible_gpu_names = []

    effective_gpu_count = len(visible_gpu_indices) if visible_gpu_indices else (torch_gpu_count if torch_cuda_available else 0)

    if requested_kind == "cpu":
        return ResolvedDevice(
            requested_device=requested_device,
            resolved_device="cpu",
            cuda_visible_devices=effective_cuda_visible_devices,
            torch_cuda_available=torch_cuda_available,
            gpu_name=None,
            gpu_count=int(effective_gpu_count),
            model_uses_gpu=False,
            model_supports_cuda=bool(model_supports_cuda),
            device_note="cpu_requested",
            visible_gpu_indices=visible_gpu_indices,
            torch_cuda_error=torch_cuda_error,
            nvidia_smi_error=nvidia_smi_error,
        )

    if effective_gpu_count <= 0:
        raise RuntimeError(
            "CUDA device was requested but no usable GPU is visible. "
            f"torch_cuda_available={torch_cuda_available} nvidia_smi_error={nvidia_smi_error}"
        )

    logical_index = 0 if requested_index is None else int(requested_index)
    if logical_index >= effective_gpu_count:
        raise ValueError(
            f"Requested CUDA device index {logical_index} is out of range for the current visibility mask; "
            f"visible_gpu_count={effective_gpu_count}"
        )

    gpu_name = visible_gpu_names[logical_index] if logical_index < len(visible_gpu_names) else None
    if model_supports_cuda:
        resolved_device = f"cuda:{logical_index}"
        model_uses_gpu = True
        note = "cuda_requested_and_model_can_use_gpu"
    else:
        resolved_device = "cpu"
        model_uses_gpu = False
        note = "cuda_requested_but_model_path_is_cpu_only"

    return ResolvedDevice(
        requested_device=requested_device,
        resolved_device=resolved_device,
        cuda_visible_devices=effective_cuda_visible_devices,
        torch_cuda_available=torch_cuda_available,
        gpu_name=gpu_name,
        gpu_count=int(effective_gpu_count),
        model_uses_gpu=model_uses_gpu,
        model_supports_cuda=bool(model_supports_cuda),
        device_note=note,
        visible_gpu_indices=visible_gpu_indices,
        torch_cuda_error=torch_cuda_error,
        nvidia_smi_error=nvidia_smi_error,
    )
