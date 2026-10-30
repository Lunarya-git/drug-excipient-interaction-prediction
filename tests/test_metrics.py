"""Tests for the evaluation metrics module."""

from __future__ import annotations

import math
import numpy as np
import pytest


def _labels_scores():
    rng = np.random.default_rng(42)
    labels = rng.integers(0, 2, size=200)
    # Noisy scores correlated with labels.
    scores = labels * 0.6 + rng.uniform(0, 0.4, size=200)
    scores = np.clip(scores, 0, 1)
    return labels, scores


class TestAP:
    def test_perfect_score(self):
        from src.evaluation.metrics import compute_ap
        labels = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        assert compute_ap(labels, scores) == pytest.approx(1.0)

    def test_nan_on_single_class(self):
        from src.evaluation.metrics import compute_ap
        labels = np.array([0, 0, 0])
        scores = np.array([0.1, 0.5, 0.9])
        assert math.isnan(compute_ap(labels, scores))


class TestAUROC:
    def test_perfect_score(self):
        from src.evaluation.metrics import compute_auroc
        labels = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        assert compute_auroc(labels, scores) == pytest.approx(1.0)

    def test_random_score(self):
        from src.evaluation.metrics import compute_auroc
        rng = np.random.default_rng(0)
        labels = rng.integers(0, 2, size=100)
        scores = rng.random(100)
        auc = compute_auroc(labels, scores)
        assert 0.0 <= auc <= 1.0


class TestThresholdMetrics:
    def test_known_values(self):
        from src.evaluation.metrics import compute_threshold_metrics
        labels = np.array([1, 1, 0, 0])
        scores = np.array([0.9, 0.8, 0.3, 0.2])
        m = compute_threshold_metrics(labels, scores, threshold=0.5)
        assert m.tp == 2
        assert m.fp == 0
        assert m.fn == 0
        assert m.tn == 2
        assert m.precision == pytest.approx(1.0)
        assert m.recall == pytest.approx(1.0)

    def test_f_beta_with_beta2(self):
        from src.evaluation.metrics import compute_threshold_metrics
        labels = np.array([1, 1, 1, 0])
        scores = np.array([0.9, 0.8, 0.4, 0.7])
        m = compute_threshold_metrics(labels, scores, threshold=0.5, beta=2.0)
        assert m.beta == 2.0
        assert 0.0 <= m.f_beta <= 1.0


class TestSelectThreshold:
    def test_returns_float_in_range(self):
        from src.evaluation.metrics import select_threshold
        labels, scores = _labels_scores()
        t = select_threshold(labels, scores, metric="f2", beta=2.0)
        assert 0.0 <= t <= 1.0

    def test_f1_threshold_in_range(self):
        from src.evaluation.metrics import select_threshold
        labels, scores = _labels_scores()
        t = select_threshold(labels, scores, metric="f1")
        assert 0.0 <= t <= 1.0


class TestCalibration:
    def test_perfect_calibration(self):
        from src.evaluation.metrics import compute_calibration
        # Scores and labels perfectly aligned.
        labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        scores = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.95, 0.95, 0.95, 0.95, 0.95])
        cal = compute_calibration(labels, scores, n_bins=2)
        assert cal.ece < 0.1

    def test_n_bins_respected(self):
        from src.evaluation.metrics import compute_calibration
        labels, scores = _labels_scores()
        cal = compute_calibration(labels, scores, n_bins=5)
        assert cal.n_bins == 5
        assert len(cal.bin_counts) == 5


class TestComputeExperimentMetrics:
    def test_full_run(self):
        from src.evaluation.metrics import compute_experiment_metrics
        labels, scores = _labels_scores()
        cfg = {
            "evaluation": {
                "threshold_selection": {"metric": "f2", "beta": 2.0,
                                        "grid": {"start": 0.1, "stop": 0.9, "step": 0.1}},
                "calibration_bins": 5,
            }
        }
        m = compute_experiment_metrics("test_exp", labels, scores, cfg)
        assert m.n_samples == len(labels)
        assert 0.0 <= m.average_precision <= 1.0
        assert 0.0 <= m.auroc <= 1.0
        d = m.to_dict()
        assert d["experiment_id"] == "test_exp"
