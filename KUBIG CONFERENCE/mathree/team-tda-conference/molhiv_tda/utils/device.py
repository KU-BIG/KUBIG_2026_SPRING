"""Device helpers for training scripts."""
from __future__ import annotations

import torch


def resolve_device(requested: str | None = None) -> torch.device:
    """
    Pick training device.

    - requested='cuda' / 'cuda:0' → use GPU (error if unavailable)
    - requested='cpu' → CPU
    - requested=None or 'auto' → CUDA if available, else CPU
    """
    if requested is None or requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but not available. "
            "Install CUDA PyTorch: bash scripts/setup_gpu.sh"
        )
    return device


def device_label(device: torch.device) -> str:
    if device.type != "cuda":
        return str(device)
    idx = device.index if device.index is not None else torch.cuda.current_device()
    name = torch.cuda.get_device_name(idx)
    return f"cuda:{idx} ({name})"
