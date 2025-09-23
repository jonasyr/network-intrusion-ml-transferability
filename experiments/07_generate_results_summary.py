#!/usr/bin/env python3
"""Generate a consolidated summary of all experimental results."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Dict, List, Optional

import pandas as pd

CURRENT_DIR = Path(__file__).parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

RESULTS_DIR = PROJECT_ROOT / "data/results"
BASELINE_PATH = RESULTS_DIR / "baseline_results.csv"
ADVANCED_PATH = RESULTS_DIR / "advanced_results.csv"
CV_SUMMARY_PATH = RESULTS_DIR / "cross_validation/cv_summary_table.csv"
FORWARD_CROSS_DATASET_PATH = RESULTS_DIR / "cross_dataset_evaluation_fixed.csv"
REVERSE_CROSS_DATASET_PATH = RESULTS_DIR / "reverse_cross_dataset_evaluation_fixed.csv"
BIDIRECTIONAL_PATH = RESULTS_DIR / "bidirectional_cross_dataset_analysis.csv"
HARMONIZED_PATH = RESULTS_DIR / "harmonized_cross_validation.json"
SUMMARY_OUTPUT_PATH = RESULTS_DIR / "experiment_summary.csv"
SUMMARY_JSON_PATH = RESULTS_DIR / "experiment_summary.json"


def _load_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        print(f"⚠️ Missing expected results file: {path}")
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - unexpected IO error
        print(f"❌ Failed to read {path}: {exc}")
        return None


def _load_harmonized_results(path: Path) -> Optional[Dict]:
    if not path.exists():
        print(f"⚠️ Harmonized evaluation file not found: {path}")
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:  # pragma: no cover - malformed JSON
        print(f"❌ Failed to parse harmonized results: {exc}")
        return None


def _summarize_baseline(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    numeric_df = df.select_dtypes(include=["number"]).copy()
    best_accuracy_row = df.loc[numeric_df["accuracy"].idxmax()]
    best_f1_row = df.loc[numeric_df["f1_score"].idxmax()]
    return {
        "best_accuracy": {
            "model": str(best_accuracy_row["model_name"]),
            "accuracy": float(best_accuracy_row["accuracy"]),
            "f1": float(best_accuracy_row["f1_score"]),
        },
        "best_f1": {
            "model": str(best_f1_row["model_name"]),
            "accuracy": float(best_f1_row["accuracy"]),
            "f1": float(best_f1_row["f1_score"]),
        },
    }


def _summarize_advanced(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    numeric_df = df.select_dtypes(include=["number"]).copy()
    best_accuracy_row = df.loc[numeric_df["accuracy"].idxmax()]
    best_auc_row = df.loc[numeric_df["roc_auc"].idxmax()]
    return {
        "best_accuracy": {
            "model": str(best_accuracy_row["model_name"]),
            "accuracy": float(best_accuracy_row["accuracy"]),
            "precision": float(best_accuracy_row["precision"]),
            "recall": float(best_accuracy_row["recall"]),
        },
        "best_auc": {
            "model": str(best_auc_row["model_name"]),
            "roc_auc": float(best_auc_row["roc_auc"]),
            "accuracy": float(best_auc_row["accuracy"]),
        },
    }


def _summarize_cross_validation(df: pd.DataFrame) -> Dict[str, str]:
    if df.empty:
        return {}
    best_f1_idx = df["F1-Score"].str.split().str[0].astype(float).idxmax()
    best_row = df.loc[best_f1_idx]
    return {
        "best_model": str(best_row["Model"]),
        "f1_score": str(best_row["F1-Score"]),
        "accuracy": str(best_row["Accuracy"]),
    }


def _summarize_cross_dataset(df: pd.DataFrame, direction: str) -> Dict[str, float]:
    if df.empty:
        return {}
    best_row = df.loc[df["Transfer_Ratio"].idxmax()]
    return {
        "best_model": str(best_row["Model"]),
        "transfer_ratio": float(best_row["Transfer_Ratio"]),
        "relative_drop": float(best_row["Relative_Drop_%"]),
        "source_accuracy": float(best_row["Source_Accuracy"]),
        "target_accuracy": float(best_row["Target_Accuracy"]),
        "direction": direction,
    }


def _summarize_bidirectional(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty:
        return {}
    summary = {
        "mean_transfer_ratio": float(df["Avg_Transfer_Ratio"].mean()),
        "mean_generalization_gap": float(df["Avg_Gap"].mean()),
        "mean_relative_drop": float(df["Avg_Relative_Drop"].mean()),
    }
    best_row = df.sort_values("Avg_Transfer_Ratio", ascending=False).iloc[0]
    summary.update(
        {
            "best_model": str(best_row["Model"]),
            "best_avg_transfer_ratio": float(best_row["Avg_Transfer_Ratio"]),
            "transfer_asymmetry": float(best_row["Transfer_Asymmetry"]),
        }
    )
    return summary


def _summarize_harmonized(data: Dict) -> List[Dict[str, float]]:
    if not data:
        return []
    results = []
    for entry in data.get("results", []):
        results.append(
            {
                "source": entry.get("source"),
                "target": entry.get("target"),
                "cv_f1_mean": float(entry.get("cv_f1_mean", 0)),
                "cv_f1_std": float(entry.get("cv_f1_std", 0)),
                "target_accuracy": float(entry.get("target_accuracy", 0)),
                "target_f1": float(entry.get("target_f1", 0)),
                "threshold_tuned_f1": float(entry.get("threshold_tuned_f1", 0)),
            }
        )
    return results


def create_summary() -> pd.DataFrame:
    summary_rows: List[Dict[str, str]] = []
    detailed_summary: Dict[str, object] = {}

    baseline_df = _load_csv(BASELINE_PATH)
    if baseline_df is not None and not baseline_df.empty:
        baseline_summary = _summarize_baseline(baseline_df)
        detailed_summary["baseline"] = baseline_summary
        summary_rows.append(
            {
                "Experiment": "Baseline Models",
                "Metric": "Best Accuracy",
                "Value": f"{baseline_summary['best_accuracy']['model']} (acc={baseline_summary['best_accuracy']['accuracy']:.4f})",
            }
        )
        summary_rows.append(
            {
                "Experiment": "Baseline Models",
                "Metric": "Best F1",
                "Value": f"{baseline_summary['best_f1']['model']} (F1={baseline_summary['best_f1']['f1']:.4f})",
            }
        )

    advanced_df = _load_csv(ADVANCED_PATH)
    if advanced_df is not None and not advanced_df.empty:
        advanced_summary = _summarize_advanced(advanced_df)
        detailed_summary["advanced"] = advanced_summary
        summary_rows.append(
            {
                "Experiment": "Advanced Models",
                "Metric": "Best Accuracy",
                "Value": f"{advanced_summary['best_accuracy']['model']} (acc={advanced_summary['best_accuracy']['accuracy']:.4f})",
            }
        )
        summary_rows.append(
            {
                "Experiment": "Advanced Models",
                "Metric": "Best ROC-AUC",
                "Value": f"{advanced_summary['best_auc']['model']} (AUC={advanced_summary['best_auc']['roc_auc']:.4f})",
            }
        )

    cv_df = _load_csv(CV_SUMMARY_PATH)
    if cv_df is not None and not cv_df.empty:
        cv_summary = _summarize_cross_validation(cv_df)
        detailed_summary["cross_validation"] = cv_summary
        summary_rows.append(
            {
                "Experiment": "Cross-Validation",
                "Metric": "Best F1",
                "Value": f"{cv_summary['best_model']} (F1={cv_summary['f1_score']})",
            }
        )

    forward_df = _load_csv(FORWARD_CROSS_DATASET_PATH)
    if forward_df is not None and not forward_df.empty:
        forward_summary = _summarize_cross_dataset(forward_df, "NSL→CIC")
        detailed_summary.setdefault("cross_dataset", {})["forward"] = forward_summary
        summary_rows.append(
            {
                "Experiment": "Cross-Dataset",
                "Metric": "Best Transfer (NSL→CIC)",
                "Value": (
                    f"{forward_summary['best_model']} (ratio={forward_summary['transfer_ratio']:.3f}, "
                    f"Δ={forward_summary['relative_drop']:.2f}%)"
                ),
            }
        )

    reverse_df = _load_csv(REVERSE_CROSS_DATASET_PATH)
    if reverse_df is not None and not reverse_df.empty:
        reverse_summary = _summarize_cross_dataset(reverse_df, "CIC→NSL")
        detailed_summary.setdefault("cross_dataset", {})["reverse"] = reverse_summary
        summary_rows.append(
            {
                "Experiment": "Cross-Dataset",
                "Metric": "Best Transfer (CIC→NSL)",
                "Value": (
                    f"{reverse_summary['best_model']} (ratio={reverse_summary['transfer_ratio']:.3f}, "
                    f"Δ={reverse_summary['relative_drop']:.2f}%)"
                ),
            }
        )

    bidirectional_df = _load_csv(BIDIRECTIONAL_PATH)
    if bidirectional_df is not None and not bidirectional_df.empty:
        bidirectional_summary = _summarize_bidirectional(bidirectional_df)
        detailed_summary["cross_dataset_summary"] = bidirectional_summary
        summary_rows.append(
            {
                "Experiment": "Cross-Dataset",
                "Metric": "Mean Transfer Ratio",
                "Value": f"{bidirectional_summary['mean_transfer_ratio']:.3f}",
            }
        )

    harmonized_data = _load_harmonized_results(HARMONIZED_PATH)
    harmonized_summary = _summarize_harmonized(harmonized_data) if harmonized_data else []
    if harmonized_summary:
        detailed_summary["harmonized"] = harmonized_summary
        for entry in harmonized_summary:
            summary_rows.append(
                {
                    "Experiment": "Harmonized Evaluation",
                    "Metric": f"{entry['source']}→{entry['target']} Target F1",
                    "Value": f"{entry['target_f1']:.4f} (CV F1={entry['cv_f1_mean']:.4f})",
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        SUMMARY_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(SUMMARY_OUTPUT_PATH, index=False)
        print("\n📊 EXPERIMENT SUMMARY")
        print(summary_df.to_string(index=False))
        print(f"\n💾 Summary saved to {SUMMARY_OUTPUT_PATH}")
    else:
        print("⚠️ No summary rows generated. Ensure experiments have been executed.")

    SUMMARY_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_JSON_PATH.open("w", encoding="utf-8") as handle:
        json.dump(detailed_summary, handle, indent=2)
        print(f"💾 Detailed summary saved to {SUMMARY_JSON_PATH}")

    return summary_df


def main() -> bool:
    summary_df = create_summary()
    return not summary_df.empty


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
