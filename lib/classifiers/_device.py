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
