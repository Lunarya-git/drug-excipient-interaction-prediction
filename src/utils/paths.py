"""Output directory layout and run identifiers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from src.utils.config import resolve_path
from src.utils.io_utils import ensure_dir


@dataclass(frozen=True)
class OutputPaths:
    """Resolved output directories (all below ``experiments.output_root``)."""

    root: Path
    logs: Path
    preprocessing: Path
    experiments: Path
    checkpoints: Path
    metrics: Path
    predictions: Path
    plots: Path
    reports: Path

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "OutputPaths":
        exp = cfg["experiments"]
        root = resolve_path(cfg, exp["output_root"])
        sub = exp["subdirs"]
        return cls(
            root=root,
            logs=root / sub["logs"],
            preprocessing=root / sub["preprocessing"],
            experiments=root / sub["experiments"],
            checkpoints=root / sub["checkpoints"],
            metrics=root / sub["metrics"],
            predictions=root / sub["predictions"],
            plots=root / sub["plots"],
            reports=root / sub["reports"],
        )

    def ensure_all(self) -> None:
        for p in (
            self.logs,
            self.preprocessing,
            self.experiments,
            self.checkpoints,
            self.metrics,
            self.predictions,
            self.plots,
            self.reports,
        ):
            ensure_dir(p)


@dataclass(frozen=True)
class StageRun:
    """Directories of one run of one stage.

    Layout (``<root>`` = ``outputs``)::

        <root>/experiments/<stage>/<run_id>/<EXP_ID>/seed<S>/fold<K>/   per-fold results
        <root>/experiments/<stage>/<run_id>/<EXP_ID>/seed<S>/            per-seed OOF results
        <root>/checkpoints/<stage>/<run_id>/<EXP_ID>/seed<S>/fold<K>/    best checkpoints
        <root>/metrics|predictions|plots|reports/<stage>/<run_id>/       aggregate artefacts
    """

    stage: str
    run_id: str
    experiments: Path
    checkpoints: Path
    metrics: Path
    predictions: Path
    plots: Path
    reports: Path

    @classmethod
    def create(cls, paths: OutputPaths, stage: str, run_id: str) -> "StageRun":
        return cls(
            stage=stage,
            run_id=run_id,
            experiments=paths.experiments / stage / run_id,
            checkpoints=paths.checkpoints / stage / run_id,
            metrics=paths.metrics / stage / run_id,
            predictions=paths.predictions / stage / run_id,
            plots=paths.plots / stage / run_id,
            reports=paths.reports / stage / run_id,
        )

    def exists(self) -> bool:
        return self.experiments.exists()

    def ensure_all(self) -> None:
        for p in (
            self.experiments,
            self.checkpoints,
            self.metrics,
            self.predictions,
            self.plots,
            self.reports,
        ):
            ensure_dir(p)

    def seed_dir(self, exp_id: str, seed: int) -> Path:
        return self.experiments / exp_id / f"seed{seed}"

    def fold_dir(self, exp_id: str, seed: int, fold: int) -> Path:
        return self.seed_dir(exp_id, seed) / f"fold{fold}"

    def checkpoint_dir(self, exp_id: str, seed: int, fold: int) -> Path:
        return self.checkpoints / exp_id / f"seed{seed}" / f"fold{fold}"
