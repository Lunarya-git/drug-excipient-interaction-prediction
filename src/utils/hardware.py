"""Device selection, hardware reporting, GPU-memory helpers and seeding."""

from __future__ import annotations

import logging
import os
import platform
import random
import sys
from typing import Any, Mapping

import numpy as np
import torch


def configure_device(hw_cfg: Mapping[str, Any], logger: logging.Logger | None = None) -> torch.device:
    """Select the compute device and apply hardware settings.

    ``device: auto`` uses CUDA when available and otherwise falls back to CPU
    **with a logged warning**. An explicit ``cuda`` request without CUDA raises
    an error: the code never silently pretends to run on a GPU.
    """
    log = logger or logging.getLogger(__name__)
    alloc_conf = hw_cfg.get("cuda_alloc_conf")
    if alloc_conf:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = str(alloc_conf)
    deterministic = bool(hw_cfg.get("deterministic", True))
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    request = str(hw_cfg.get("device", "auto")).lower()
    cuda_ok = torch.cuda.is_available()
    if request == "auto":
        if cuda_ok:
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
            log.warning(
                "CUDA is NOT available (torch %s, built with CUDA: %s). Falling back to CPU.",
                torch.__version__,
                torch.version.cuda,
            )
    elif request.startswith("cuda"):
        if not cuda_ok:
            raise RuntimeError(
                f"hardware.device='{request}' was requested but torch.cuda.is_available() is False "
                f"(torch {torch.__version__}, built with CUDA: {torch.version.cuda}). Install a CUDA "
                "build of PyTorch (see README) or set hardware.device to 'auto' or 'cpu'."
            )
        device = torch.device(request)
    elif request == "cpu":
        device = torch.device("cpu")
    else:
        raise ValueError(f"Unsupported hardware.device value '{request}'")

    torch.backends.cudnn.benchmark = bool(hw_cfg.get("cudnn_benchmark", False))
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)

    if device.type == "cuda":
        fraction = hw_cfg.get("gpu_memory_fraction")
        if fraction:
            torch.cuda.set_per_process_memory_fraction(float(fraction), device.index or 0)
    elif hw_cfg.get("cpu_threads"):
        torch.set_num_threads(int(hw_cfg["cpu_threads"]))
    return device


def hardware_report(device: torch.device) -> dict[str, Any]:
    """Collect software / hardware facts for logs and metadata."""
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": str(device),
        "device_type": device.type,
        "cpu_count": os.cpu_count(),
        "gpu_name": None,
        "gpu_total_memory_gb": None,
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpu_compute_capability": None,
    }
    if device.type == "cuda":
        idx = device.index or 0
        props = torch.cuda.get_device_properties(idx)
        info["gpu_name"] = props.name
        info["gpu_total_memory_gb"] = round(props.total_memory / 2**30, 2)
        info["gpu_compute_capability"] = f"{props.major}.{props.minor}"
    return info


def log_hardware(info: Mapping[str, Any], logger: logging.Logger) -> None:
    """Log the hardware report; states clearly whether training runs on GPU or CPU."""
    logger.info("PyTorch %s | CUDA build: %s | cuDNN: %s", info["torch_version"], info["torch_cuda_build"], info["cudnn_version"])
    logger.info("CUDA available: %s | GPU count: %s", info["cuda_available"], info["gpu_count"])
    if info["device_type"] == "cuda":
        logger.info(
            "Running on GPU: %s (%s, %.2f GB, compute capability %s)",
            info["gpu_name"],
            info["device"],
            info["gpu_total_memory_gb"],
            info["gpu_compute_capability"],
        )
    else:
        logger.info("Running on CPU (device=%s). No GPU is used in this run.", info["device"])


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (CPU and CUDA)."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def reset_peak_memory(device: torch.device) -> None:
    """Reset CUDA peak-allocation statistics (no-op on CPU)."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_mb(device: torch.device) -> float | None:
    """Peak allocated CUDA memory in MiB since the last reset (None on CPU)."""
    if device.type == "cuda":
        return float(torch.cuda.max_memory_allocated(device)) / 2**20
    return None


def synchronize(device: torch.device) -> None:
    """Synchronise CUDA so that timings are accurate (no-op on CPU)."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def release_gpu_memory(device: torch.device) -> None:
    """Free cached CUDA memory (no-op on CPU)."""
    import gc

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
