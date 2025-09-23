"""Unit tests for feature alignment and cross-dataset metrics."""

import numpy as np
import numpy.testing as npt
import pytest

from src.features import FeatureAligner
from src.metrics.cross_dataset_metrics import (
    calculate_generalization_gap,
    calculate_relative_performance_drop,
    calculate_transfer_ratio,
    compute_domain_divergence,
)


def test_feature_alignment_shapes() -> None:
    aligner = FeatureAligner()
    source = np.random.RandomState(42).randn(100, 10)
    target = np.random.RandomState(13).randn(120, 15)

    # Statistical alignment should keep dimensionality of the source dataset
    stat = aligner.statistical_alignment(source, target[:, :10])
    assert stat.source.shape == source.shape
    assert stat.target.shape == target[:, :10].shape

    # PCA alignment should produce identical dimensionality for both datasets
    pca = aligner.pca_alignment(source, target[:, :10], n_components=5)
    assert pca.source.shape[1] == 5
    assert pca.target.shape[1] == 5


def test_alignment_requires_matching_dimensions() -> None:
    aligner = FeatureAligner()
    source = np.random.RandomState(0).randn(10, 5)
    target = np.random.RandomState(1).randn(12, 6)

    with pytest.raises(ValueError):
        aligner.statistical_alignment(source, target)

    with pytest.raises(ValueError):
        aligner.pca_alignment(source, target)


def test_transform_dataset_respects_feature_order() -> None:
    aligner = FeatureAligner()
    data = np.arange(15, dtype=float).reshape(5, 3)
    feature_names = ["a", "b", "c"]
    selected = ["c", "a"]

    transformed = aligner.transform_dataset(data, feature_names, selected)
    assert transformed.shape == (5, 2)
    npt.assert_array_equal(transformed[:, 0], data[:, 2])
    npt.assert_array_equal(transformed[:, 1], data[:, 0])


def test_cross_dataset_metrics_are_positive() -> None:
    gap = calculate_generalization_gap(0.92, 0.71)
    drop = calculate_relative_performance_drop(0.92, 0.71)
    ratio = calculate_transfer_ratio(0.92, 0.71)

    assert gap >= 0
    assert 0 <= drop <= 100
    assert 0 <= ratio <= 1

    source = np.random.RandomState(7).randn(50, 3)
    target = np.random.RandomState(11).randn(60, 3)
    divergence = compute_domain_divergence(source, target)
    assert divergence >= 0
