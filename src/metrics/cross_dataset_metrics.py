"""Metrics for evaluating cross-dataset generalization."""

from __future__ import annotations

import numpy as np
from scipy.stats import wasserstein_distance


def _validate_accuracy(value: float, name: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1. Received {value}.")


def calculate_generalization_gap(
    source_accuracy: float, target_accuracy: float
) -> float:
    """Return the non-negative generalization gap."""

    _validate_accuracy(source_accuracy, "source_accuracy")
    _validate_accuracy(target_accuracy, "target_accuracy")
    return float(max(0.0, source_accuracy - target_accuracy))


def calculate_relative_performance_drop(
    source_accuracy: float, target_accuracy: float
) -> float:
    """Compute the percentage drop in performance when transferring domains."""

    gap = calculate_generalization_gap(source_accuracy, target_accuracy)
    if source_accuracy == 0:
        return 0.0
    drop = (gap / source_accuracy) * 100.0
    return float(min(100.0, max(0.0, drop)))


def calculate_transfer_ratio(source_accuracy: float, target_accuracy: float) -> float:
    """Return the capped ratio between target and source accuracy."""

    _validate_accuracy(source_accuracy, "source_accuracy")
    _validate_accuracy(target_accuracy, "target_accuracy")
    if source_accuracy == 0:
        return 0.0
    ratio = target_accuracy / source_accuracy
    return float(min(1.0, max(0.0, ratio)))


def _ensure_2d(data: np.ndarray) -> np.ndarray:
    if data.ndim == 1:
        return data.reshape(-1, 1)
    return data


def compute_domain_divergence(
    source_features: np.ndarray, target_features: np.ndarray
) -> float:
    """Estimate Wasserstein distance between feature distributions."""

    source = _ensure_2d(np.asarray(source_features, dtype=float))
    target = _ensure_2d(np.asarray(target_features, dtype=float))
    if source.shape[1] != target.shape[1]:
        raise ValueError("Source and target must have the same number of columns")

    distances = []
    for idx in range(source.shape[1]):
        distances.append(wasserstein_distance(source[:, idx], target[:, idx]))
    divergence = float(np.mean(distances))
    return max(0.0, divergence)


__all__ = [
    "calculate_generalization_gap",
    "calculate_relative_performance_drop",
    "calculate_transfer_ratio",
    "compute_domain_divergence",
]
