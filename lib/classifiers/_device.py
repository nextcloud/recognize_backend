# SPDX-FileCopyrightText: 2026 Nextcloud GmbH and Nextcloud contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared device detection helpers."""

import os

from nc_py_api.ex_app import get_computation_device

# Set to True by :func:`force_cpu` once a CUDA/cuDNN runtime error is observed,
# so subsequent model loads stay on CPU for the rest of the process lifetime.
_FORCE_CPU = False


def force_cpu() -> None:
    """Permanently pin all subsequent model loads to CPU."""
    global _FORCE_CPU
    _FORCE_CPU = True


def torch_device() -> str:
    if _FORCE_CPU:
        return "cpu"
    requested = (get_computation_device() or os.environ.get("COMPUTE_DEVICE", "")).lower()
    if requested in ("cuda", "gpu"):
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
    return "cpu"


def onnx_providers() -> list[str]:
    if torch_device() == "cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


# Fraction of the currently free VRAM a model's inference batches may use.
VRAM_BUDGET = float(os.environ.get("RECOGNIZE_VRAM_BUDGET", "0.5"))


def batch_size_for(vram_per_item: int, max_batch: int) -> int:
    """Pick an inference batch size from the VRAM that is free right now.

    Call this after the model is loaded so its weights are already accounted
    for. ``RECOGNIZE_BATCH_SIZE`` overrides the heuristic; on CPU batching
    gains little, so we use 1.
    """
    override = os.environ.get("RECOGNIZE_BATCH_SIZE")
    if override:
        return max(1, int(override))
    if torch_device() != "cuda":
        return 1
    try:
        import torch
        free, _total = torch.cuda.mem_get_info()
    except Exception:
        return 1
    return max(1, min(max_batch, int(free * VRAM_BUDGET // vram_per_item)))


_OOM_TOKENS = (
    "OUT OF MEMORY",
    "FAILED TO ALLOCATE MEMORY",  # onnxruntime BFCArena
)


def is_cuda_oom(exc: BaseException) -> bool:
    """Best-effort match for torch/onnxruntime GPU out-of-memory errors."""
    msg = (str(exc) or "").upper()
    return any(tok in msg for tok in _OOM_TOKENS)


_CUDA_ERROR_TOKENS = (
    "CUDNN",
    "CUBLAS",
    "CUFFT",
    "CUSPARSE",
    "CUDA",
    "CUDAEXECUTIONPROVIDER",
    "GPU",
    "NVRTC",
    "NCCL",
)


def is_cuda_runtime_error(exc: BaseException) -> bool:
    """Best-effort match for CUDA/cuDNN/onnxruntime-GPU runtime failures."""
    msg = (str(exc) or "").upper()
    return any(tok in msg for tok in _CUDA_ERROR_TOKENS)
