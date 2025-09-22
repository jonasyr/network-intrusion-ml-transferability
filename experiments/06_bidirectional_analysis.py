#!/usr/bin/env python3
"""Bidirectional analysis of NSL-KDD and CIC-IDS-2017 cross-dataset experiments."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

FORWARD_RESULTS = Path("data/results/cross_dataset_evaluation_fixed.csv")
REVERSE_RESULTS = Path("data/results/reverse_cross_dataset_evaluation_fixed.csv")
COMBINED_PATH = Path("data/results/bidirectional_cross_dataset_analysis.csv")


def create_bidirectional_analysis() -> bool:
    if not FORWARD_RESULTS.exists() or not REVERSE_RESULTS.exists():
        print("❌ Required result files not found. Run the cross-dataset experiments first.")
        return False

    forward_df = pd.read_csv(FORWARD_RESULTS)
    reverse_df = pd.read_csv(REVERSE_RESULTS)

    forward_df = forward_df.rename(
        columns={
            "Source_Accuracy": "NSL_Test_Accuracy",
            "Source_F1": "NSL_Test_F1",
            "Target_Accuracy": "CIC_Test_Accuracy",
            "Target_F1": "CIC_Test_F1",
            "Generalization_Gap": "NSL_to_CIC_Gap",
            "Relative_Drop_%": "NSL_to_CIC_Relative_Drop",
            "Transfer_Ratio": "NSL_to_CIC_Transfer_Ratio",
        }
    )

    reverse_df = reverse_df.rename(
        columns={
            "Source_Accuracy": "CIC_Validation_Accuracy",
            "Source_F1": "CIC_Validation_F1",
            "Target_Accuracy": "NSL_Test_Accuracy_From_CIC",
            "Target_F1": "NSL_Test_F1_From_CIC",
            "Generalization_Gap": "CIC_to_NSL_Gap",
            "Relative_Drop_%": "CIC_to_NSL_Relative_Drop",
            "Transfer_Ratio": "CIC_to_NSL_Transfer_Ratio",
        }
    )

    combined = forward_df.merge(reverse_df, on="Model", suffixes=("_forward", "_reverse"))

    combined["Avg_Gap"] = (
        combined["NSL_to_CIC_Gap"] + combined["CIC_to_NSL_Gap"]
    ) / 2
    combined["Avg_Relative_Drop"] = (
        combined["NSL_to_CIC_Relative_Drop"] + combined["CIC_to_NSL_Relative_Drop"]
    ) / 2
    combined["Avg_Transfer_Ratio"] = (
        combined["NSL_to_CIC_Transfer_Ratio"] + combined["CIC_to_NSL_Transfer_Ratio"]
    ) / 2
    combined["Transfer_Asymmetry"] = (
        combined["NSL_to_CIC_Transfer_Ratio"] - combined["CIC_to_NSL_Transfer_Ratio"]
    ).abs()

    COMBINED_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(COMBINED_PATH, index=False)

    print("📊 BIDIRECTIONAL CROSS-DATASET ANALYSIS")
    print("=" * 80)
    print(combined.round(4).to_string(index=False))

    best_transfer = combined.sort_values("Avg_Transfer_Ratio", ascending=False).iloc[0]
    print("\n🔍 Key Insights")
    print(
        f"   • Best average transfer: {best_transfer['Model']} (ratio {best_transfer['Avg_Transfer_Ratio']:.3f})"
    )
    print(
        f"   • Mean generalization gap: {combined['Avg_Gap'].mean():.4f}"
        f" | Mean relative drop: {combined['Avg_Relative_Drop'].mean():.2f}%"
    )
    print(
        f"   • Most symmetric transfer: {combined.sort_values('Transfer_Asymmetry').iloc[0]['Model']}"
    )

    return True


def main() -> bool:
    success = create_bidirectional_analysis()
    if success:
        print("\n🎯 Bidirectional analysis complete!")
    return success


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
