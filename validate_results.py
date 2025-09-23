"""Validate scientific correctness of cross-dataset evaluation results."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RESULTS_FILE = Path("data/results/cross_dataset_evaluation_fixed.csv")


def main() -> None:
    if not RESULTS_FILE.exists():
        raise FileNotFoundError(
            "Cross-dataset results not found. Run experiments/05_cross_dataset_evaluation.py first."
        )

    results = pd.read_csv(RESULTS_FILE)

    if not (results["Relative_Drop_%"] >= 0).all():
        raise AssertionError("Found negative performance drops in results table.")
    if not (results["Relative_Drop_%"] <= 100).all():
        raise AssertionError("Relative performance drops exceed 100%.")
    if not (results["Transfer_Ratio"].between(0, 1)).all():
        raise AssertionError("Transfer ratios must be in [0, 1].")
    if not (results["Target_Accuracy"] <= results["Source_Accuracy"]).all():
        raise AssertionError("Cross-dataset accuracy should not exceed within-dataset accuracy.")

    print("✅ All results are scientifically valid")


if __name__ == "__main__":
    main()
