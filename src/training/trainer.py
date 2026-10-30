"""Core training loop for a single (fold, seed) run.

Responsibilities
----------------
* Build the model from config.
* Compute class weights from the training fold labels.
* Carve out an inner validation split for early stopping.
* Train with AdamW + optional LR scheduler.
* Collect per-epoch metrics (train loss, inner val loss/AP, held-out loss/AP).
* Display a tqdm progress bar with live metrics per epoch.
* Save the best checkpoint and return a :class:`FoldResult`.

Nothing here touches the CV validation fold for stopping decisions.
"""

from __future__ import annotations

import copy
import logging
import math
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from tqdm import tqdm

from src.features.assemble import FoldScaler, RepresentationData, make_fold_arrays
from src.models.registry import build_model
from src.training.early_stopping import EarlyStopping
from src.training.results import EpochMetrics, FoldHistory, FoldResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _compute_class_weights(labels: np.ndarray, mode: str, device: torch.device) -> torch.Tensor | None:
    """Return [w_compatible, w_incompatible] or None if mode=='none'."""
    if mode == "none":
        return None
    n = len(labels)
    n_classes = 2
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    counts = np.maximum(counts, 1.0)          # avoid division by zero
    weights = n / (n_classes * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _make_optimizer(model: nn.Module, cfg_training: Mapping[str, Any]) -> AdamW:
    opt = cfg_training["optimizer"]
    betas = tuple(float(b) for b in opt.get("betas", [0.9, 0.999]))
    return AdamW(
        model.parameters(),
        lr=float(opt.get("lr", 1e-3)),
        weight_decay=float(opt.get("weight_decay", 1e-4)),
        betas=betas,
    )


def _make_scheduler(
    optimizer: AdamW, cfg_training: Mapping[str, Any]
) -> ReduceLROnPlateau | None:
    sched = cfg_training.get("scheduler", {})
    name = str(sched.get("name", "none")).lower()
    if name == "none":
        return None
    if name == "plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(sched.get("factor", 0.5)),
            patience=int(sched.get("patience", 5)),
            min_lr=float(sched.get("min_lr", 1e-6)),
        )
    raise ValueError(f"Unknown scheduler '{name}'. Supported: none, plateau")


def _bce_loss(
    logits: torch.Tensor, labels: torch.Tensor, class_weights: torch.Tensor | None
) -> torch.Tensor:
    """Binary cross-entropy with optional per-sample class weights."""
    if class_weights is not None:
        sample_w = class_weights[labels.long()]
        return nn.functional.binary_cross_entropy_with_logits(
            logits, labels.float(), weight=sample_w
        )
    return nn.functional.binary_cross_entropy_with_logits(logits, labels.float())


def _ap_score(scores: np.ndarray, labels: np.ndarray) -> float:
    """Average precision (area under the precision-recall curve)."""
    from sklearn.metrics import average_precision_score

    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(average_precision_score(labels, scores))


# ---------------------------------------------------------------------------
# Inner-split helpers
# ---------------------------------------------------------------------------

def _inner_split(
    train_ids: np.ndarray, labels: np.ndarray, val_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Stratified inner split: returns (inner_train_ids, inner_val_ids)."""
    from sklearn.model_selection import StratifiedShuffleSplit

    if val_fraction <= 0.0:
        return train_ids, np.array([], dtype=np.int64)
    y = labels[train_ids]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    local_train, local_val = next(sss.split(train_ids, y))
    return train_ids[local_train], train_ids[local_val]


# ---------------------------------------------------------------------------
# Batch iteration
# ---------------------------------------------------------------------------

def _iter_batches(
    ids: np.ndarray, batch_size: int, shuffle: bool, rng: np.random.Generator
) -> list[np.ndarray]:
    if shuffle:
        ids = ids.copy()
        rng.shuffle(ids)
    n = len(ids)
    return [ids[i: i + batch_size] for i in range(0, n, batch_size)]


# ---------------------------------------------------------------------------
# Epoch functions
# ---------------------------------------------------------------------------

def _train_epoch(
    model: nn.Module,
    xd_all: torch.Tensor,
    xe_all: torch.Tensor,
    y_all: torch.Tensor,
    ids: np.ndarray,
    batch_size: int,
    optimizer: AdamW,
    class_weights: torch.Tensor | None,
    grad_clip: float | None,
    rng: np.random.Generator,
) -> float:
    model.train()
    total_loss = 0.0
    n_samples = 0
    for batch_ids in _iter_batches(ids, batch_size, shuffle=True, rng=rng):
        idx = torch.from_numpy(batch_ids).long().to(xd_all.device)
        xd = xd_all[idx]
        xe = xe_all[idx]
        y = y_all[idx]
        optimizer.zero_grad()
        logits = model(xd, xe)
        loss = _bce_loss(logits, y, class_weights)
        loss.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += float(loss.item()) * len(batch_ids)
        n_samples += len(batch_ids)
    return total_loss / max(n_samples, 1)


@torch.no_grad()
def _eval_epoch(
    model: nn.Module,
    xd_all: torch.Tensor,
    xe_all: torch.Tensor,
    y_all: torch.Tensor,
    ids: np.ndarray,
    batch_size: int,
    class_weights: torch.Tensor | None,
) -> tuple[float, float]:
    """Return (loss, average_precision) for ``ids``."""
    model.eval()
    all_logits: list[torch.Tensor] = []
    total_loss = 0.0
    n_samples = 0
    dummy_rng = np.random.default_rng(0)
    for batch_ids in _iter_batches(ids, batch_size, shuffle=False, rng=dummy_rng):
        idx = torch.from_numpy(batch_ids).long().to(xd_all.device)
        xd = xd_all[idx]
        xe = xe_all[idx]
        y = y_all[idx]
        logits = model(xd, xe)
        loss = _bce_loss(logits, y, class_weights)
        total_loss += float(loss.item()) * len(batch_ids)
        n_samples += len(batch_ids)
        all_logits.append(logits.cpu())
    avg_loss = total_loss / max(n_samples, 1)
    scores = torch.sigmoid(torch.cat(all_logits)).numpy()
    y_np = y_all[ids].cpu().numpy()
    return avg_loss, _ap_score(scores, y_np)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_fold(
    experiment_id: str,
    fold: int,
    seed: int,
    train_ids: np.ndarray,
    val_ids: np.ndarray,
    labels: np.ndarray,
    data: RepresentationData,
    cfg: Mapping[str, Any],
    model_key: str,
    checkpoint_path: Path | None = None,
    dry_run: bool = False,
) -> FoldResult:
    """Train one (fold, seed) and return a :class:`FoldResult`.

    Args:
        experiment_id: Human-readable ID used in logging.
        fold: CV fold index (0-based).
        seed: Initialisation seed.
        train_ids: Sample IDs in the training fold (indices into the cleaned dataset).
        val_ids: Sample IDs in the held-out CV fold (for diagnostic metrics only).
        labels: Integer labels (0/1) for ALL samples in the cleaned dataset.
        data: Feature matrices for the current representation.
        cfg: Merged configuration dictionary.
        model_key: Key into ``cfg['models']`` selecting the architecture.
        checkpoint_path: If not None, the best model weights are saved here.
        dry_run: If True, log the training plan and return immediately without
            executing any training. Useful for validating the configuration.
    """
    cfg_training = cfg["training"]
    cfg_hw = cfg["hardware"]
    cfg_eval = cfg["evaluation"]

    # ------------------------------------------------------------------ device
    device = _resolve_device(cfg_hw)

    if dry_run:
        logger.info(
            "[DRY RUN] %s | fold=%d seed=%d | model=%s | device=%s | "
            "train=%d val=%d | max_epochs=%d",
            experiment_id, fold, seed, model_key, device,
            len(train_ids), len(val_ids),
            int(cfg_training.get("max_epochs", 200)),
        )
        return FoldResult(
            experiment_id=experiment_id, fold=fold, seed=seed,
            best_epoch=-1,
            history=FoldHistory(fold=fold, seed=seed),
            oof_sample_ids=list(map(int, val_ids)),
            oof_labels=list(map(int, labels[val_ids])),
            oof_scores=[float("nan")] * len(val_ids),
        )

    _set_seed(seed)
    if cfg_hw.get("deterministic"):
        torch.use_deterministic_algorithms(True, warn_only=True)
    if not cfg_hw.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = False

    # ------------------------------------------------------------------ model
    model_cfg = cfg["models"][model_key]
    model: nn.Module = build_model(model_cfg, data.spec)
    model.to(device)
    logger.info(
        "%s | fold=%d seed=%d | model=%s | device=%s | params=%d | train=%d val=%d",
        experiment_id, fold, seed, model_key, device,
        sum(p.numel() for p in model.parameters() if p.requires_grad),
        len(train_ids), len(val_ids),
    )

    # ------------------------------------------------------------------ inner split
    es_cfg = cfg_training.get("early_stopping", {})
    inner_val_frac = float(es_cfg.get("inner_val_fraction", 0.10))
    inner_train_ids, inner_val_ids = _inner_split(train_ids, labels, inner_val_frac, seed)
    logger.info(
        "%s | fold=%d seed=%d | inner train=%d inner val=%d",
        experiment_id, fold, seed, len(inner_train_ids), len(inner_val_ids),
    )

    # ------------------------------------------------------------------ features on device
    scaler = FoldScaler(data.spec)
    scaler.fit(data, inner_train_ids)
    xd_np, xe_np = make_fold_arrays(data, scaler, np.arange(len(labels)))

    xd_all = torch.from_numpy(xd_np)
    xe_all = torch.from_numpy(xe_np)
    y_all = torch.from_numpy(labels.astype(np.float32))
    if cfg_hw.get("data_on_device", True):
        xd_all = xd_all.to(device)
        xe_all = xe_all.to(device)
        y_all = y_all.to(device)

    # ------------------------------------------------------------------ class weights
    cw_mode = str(cfg_training.get("class_weights", {}).get("mode", "balanced"))
    class_weights = _compute_class_weights(labels[inner_train_ids], cw_mode, device)

    # ------------------------------------------------------------------ optimizer / scheduler
    optimizer = _make_optimizer(model, cfg_training)
    scheduler = _make_scheduler(optimizer, cfg_training)

    # ------------------------------------------------------------------ early stopping
    stopper = EarlyStopping(
        monitor=str(es_cfg.get("monitor", "val_loss")),
        patience=int(es_cfg.get("patience", 20)),
        min_delta=float(es_cfg.get("min_delta", 0.0)),
        min_epochs=int(es_cfg.get("min_epochs", 10)),
    )

    batch_size = int(cfg_training.get("batch_size", 128))
    max_epochs = int(cfg_training.get("max_epochs", 200))
    grad_clip = cfg_training.get("grad_clip_norm")
    grad_clip = float(grad_clip) if grad_clip is not None else None
    log_heldout = bool(cfg_training.get("log_heldout_curves", True))
    save_best = bool(cfg_training.get("checkpoint", {}).get("save_best", True))
    rng = np.random.default_rng(seed)

    history = FoldHistory(fold=fold, seed=seed)
    best_state: dict[str, Any] | None = None

    bar_desc = f"{experiment_id} fold={fold} seed={seed}"
    pbar = tqdm(
        range(1, max_epochs + 1),
        desc=bar_desc,
        unit="ep",
        dynamic_ncols=True,
        leave=True,
    )

    for epoch in pbar:
        train_loss = _train_epoch(
            model, xd_all, xe_all, y_all,
            inner_train_ids, batch_size, optimizer, class_weights, grad_clip, rng,
        )
        inner_val_loss, inner_val_ap = _eval_epoch(
            model, xd_all, xe_all, y_all,
            inner_val_ids, batch_size, class_weights,
        ) if len(inner_val_ids) else (float("nan"), float("nan"))

        heldout_loss: float | None = None
        heldout_ap: float | None = None
        if log_heldout and len(val_ids):
            heldout_loss, heldout_ap = _eval_epoch(
                model, xd_all, xe_all, y_all,
                val_ids, batch_size, class_weights,
            )

        current_lr = float(optimizer.param_groups[0]["lr"])
        em = EpochMetrics(
            epoch=epoch,
            train_loss=train_loss,
            inner_val_loss=inner_val_loss,
            inner_val_ap=inner_val_ap,
            heldout_loss=heldout_loss,
            heldout_ap=heldout_ap,
            lr=current_lr,
        )
        history.append(em)

        monitored = inner_val_loss if stopper.monitor == "val_loss" else inner_val_ap
        if not math.isnan(monitored):
            should_stop = stopper.step(epoch, monitored)
        else:
            should_stop = False

        if save_best and (best_state is None or stopper.best_epoch == epoch):
            best_state = copy.deepcopy(model.state_dict())

        if scheduler is not None and not math.isnan(inner_val_loss):
            scheduler.step(inner_val_loss)

        # Update the progress bar with live metrics.
        patience_left = stopper.patience - (epoch - stopper.best_epoch)
        iv_ap_str = f"{inner_val_ap:.3f}" if not math.isnan(inner_val_ap) else "n/a"
        hap_str = (
            f"{heldout_ap:.3f}" if (heldout_ap is not None and not math.isnan(heldout_ap)) else "-"
        )
        pbar.set_postfix(
            trn=f"{train_loss:.4f}",
            iv_loss=f"{inner_val_loss:.4f}" if not math.isnan(inner_val_loss) else "n/a",
            iv_ap=iv_ap_str,
            heldout_ap=hap_str,
            best=stopper.best_epoch,
            pat=max(patience_left, 0),
        )

        if epoch % 20 == 0 or should_stop:
            logger.info(
                "%s | fold=%d seed=%d | ep=%d/%d | "
                "trn_loss=%.4f iv_loss=%.4f iv_ap=%.4f | best_ep=%d",
                experiment_id, fold, seed, epoch, max_epochs,
                train_loss, inner_val_loss, inner_val_ap, stopper.best_epoch,
            )

        if should_stop:
            pbar.set_description(f"{bar_desc} [STOPPED ep={epoch}]")
            logger.info(
                "%s | fold=%d seed=%d | early stopping at epoch %d (best=%d)",
                experiment_id, fold, seed, epoch, stopper.best_epoch,
            )
            break

    pbar.close()

    # ------------------------------------------------------------------ restore best
    if save_best and best_state is not None:
        model.load_state_dict(best_state)

    # ------------------------------------------------------------------ save checkpoint
    if checkpoint_path is not None and save_best and best_state is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "experiment_id": experiment_id,
                "fold": fold,
                "seed": seed,
                "best_epoch": stopper.best_epoch,
                "model_key": model_key,
                "model_state_dict": best_state,
                "scaler": scaler.to_dict(),
                "representation": data.spec.keyword,
            },
            checkpoint_path,
        )
        logger.info("Saved checkpoint: %s", checkpoint_path)

    # ------------------------------------------------------------------ OOF predictions
    oof_scores, oof_labels = _predict(model, xd_all, xe_all, y_all, val_ids, batch_size)

    return FoldResult(
        experiment_id=experiment_id,
        fold=fold,
        seed=seed,
        best_epoch=stopper.best_epoch,
        history=history,
        oof_sample_ids=list(map(int, val_ids)),
        oof_labels=list(map(int, oof_labels)),
        oof_scores=list(map(float, oof_scores)),
    )


@torch.no_grad()
def _predict(
    model: nn.Module,
    xd_all: torch.Tensor,
    xe_all: torch.Tensor,
    y_all: torch.Tensor,
    ids: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_logits: list[torch.Tensor] = []
    dummy_rng = np.random.default_rng(0)
    for batch_ids in _iter_batches(ids, batch_size, shuffle=False, rng=dummy_rng):
        idx = torch.from_numpy(batch_ids).long().to(xd_all.device)
        all_logits.append(model(xd_all[idx], xe_all[idx]).cpu())
    scores = torch.sigmoid(torch.cat(all_logits)).numpy() if all_logits else np.array([])
    labels_np = y_all[ids].cpu().numpy().astype(int) if len(ids) else np.array([], dtype=int)
    return scores, labels_np


def _resolve_device(cfg_hw: Mapping[str, Any]) -> torch.device:
    setting = str(cfg_hw.get("device", "auto")).lower()
    if setting == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not torch.cuda.is_available():
            logger.warning("CUDA not available; falling back to CPU.")
        return dev
    if setting == "cpu":
        return torch.device("cpu")
    if setting.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"hardware.device={setting!r} was requested but CUDA is not available."
            )
        return torch.device(setting)
    raise ValueError(f"hardware.device must be 'auto', 'cpu', 'cuda', or 'cuda:N'; got '{setting}'")
