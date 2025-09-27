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
# Dataset-specific result paths
NSL_BASELINE_PATH = RESULTS_DIR / "nsl_baseline_results.csv"
NSL_ADVANCED_PATH = RESULTS_DIR / "nsl_advanced_results.csv"
CIC_BASELINE_PATH = RESULTS_DIR / "cic_baseline_results.csv"
CIC_ADVANCED_PATH = RESULTS_DIR / "cic_advanced_results.csv"
# Cross-validation paths
NSL_CV_SUMMARY_PATH = RESULTS_DIR / "cross_validation/cv_summary_table.csv"
CIC_CV_SUMMARY_PATH = RESULTS_DIR / "cross_validation/cic/cv_summary_table.csv"
# Cross-dataset evaluation paths
NSL_TO_CIC_PATH = RESULTS_DIR / "nsl_trained_tested_on_cic.csv"
CIC_TO_NSL_PATH = RESULTS_DIR / "cic_trained_tested_on_nsl.csv"
BIDIRECTIONAL_PATH = RESULTS_DIR / "bidirectional_cross_dataset_analysis.csv"
HARMONIZED_PATH = RESULTS_DIR / "harmonized_cross_validation.json"
SUMMARY_OUTPUT_PATH = RESULTS_DIR / "experiment_summary.csv"
SUMMARY_JSON_PATH = RESULTS_DIR / "experiment_summary.json"

# Constants for repeated strings
BEST_ACCURACY = "Best Accuracy"
BEST_F1 = "Best F1"
BEST_ROC_AUC = "Best ROC-AUC"


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


def _load_dataset_results(dataset_name: str) -> Dict[str, Optional[pd.DataFrame]]:
    """Load all results for a specific dataset (NSL or CIC)."""
    results = {}
    
    if dataset_name.upper() == "NSL":
        baseline_path = NSL_BASELINE_PATH
        advanced_path = NSL_ADVANCED_PATH
        cv_path = NSL_CV_SUMMARY_PATH
    elif dataset_name.upper() == "CIC":
        baseline_path = CIC_BASELINE_PATH
        advanced_path = CIC_ADVANCED_PATH
        cv_path = CIC_CV_SUMMARY_PATH
    else:
        return {"baseline": None, "advanced": None, "cv": None}
    
    results["baseline"] = _load_csv(baseline_path)
    results["advanced"] = _load_csv(advanced_path)
    results["cv"] = _load_csv(cv_path)
    
    return results


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


def _parse_complex_harmonized_structure(data: Dict) -> List[Dict]:
    """Parse harmonized results from complex nested structure."""
    results = []
    for key, value in data.items():
        if key.endswith("_results") and isinstance(value, dict):
            source_target = key.replace("_results", "")
            if "→" in source_target:
                source, target = source_target.split("→")
                results.append({
                    "source": source,
                    "target": target,
                    "cv_f1_mean": float(value.get("cv_f1_mean", 0)),
                    "cv_f1_std": float(value.get("cv_f1_std", 0)),
                    "target_accuracy": float(value.get("target_accuracy", 0)),
                    "target_f1": float(value.get("target_f1", 0)),
                    "threshold_tuned_f1": float(value.get("threshold_tuned_f1", 0)),
                    "training_method": value.get("training_method", "regular")
                })
    return results


def _summarize_harmonized(data: Dict) -> List[Dict[str, float]]:
    if not data:
        return []
    
    # Handle the actual harmonized JSON structure
    harmonized_results = data.get("results", [])
    if not harmonized_results and "nsl_summary" in data:
        return _parse_complex_harmonized_structure(data)
    
    # Use results array if present
    results = []
    for entry in harmonized_results:
        results.append({
            "source": entry.get("source"),
            "target": entry.get("target"),
            "cv_f1_mean": float(entry.get("cv_f1_mean", 0)),
            "cv_f1_std": float(entry.get("cv_f1_std", 0)),
            "target_accuracy": float(entry.get("target_accuracy", 0)),
            "target_f1": float(entry.get("target_f1", 0)),
            "threshold_tuned_f1": float(entry.get("threshold_tuned_f1", 0)),
            "training_method": entry.get("training_method", "regular")
        })
    
    return results


def _add_dataset_summary_rows(dataset_name: str, results: Dict, summary_rows: List[Dict[str, str]], detailed_summary: Dict) -> None:
    """Add summary rows for a specific dataset."""
    prefix = f"{dataset_name} "
    
    if results["baseline"] is not None and not results["baseline"].empty:
        baseline_summary = _summarize_baseline(results["baseline"])
        detailed_summary[f"{dataset_name.lower()}_baseline"] = baseline_summary
        summary_rows.extend([
            {"Experiment": f"{prefix}Baseline", "Metric": BEST_ACCURACY,
             "Value": f"{baseline_summary['best_accuracy']['model']} (acc={baseline_summary['best_accuracy']['accuracy']:.4f})"},
            {"Experiment": f"{prefix}Baseline", "Metric": BEST_F1,
             "Value": f"{baseline_summary['best_f1']['model']} (F1={baseline_summary['best_f1']['f1']:.4f})"}
        ])

    if results["advanced"] is not None and not results["advanced"].empty:
        advanced_summary = _summarize_advanced(results["advanced"])
        detailed_summary[f"{dataset_name.lower()}_advanced"] = advanced_summary
        summary_rows.extend([
            {"Experiment": f"{prefix}Advanced", "Metric": BEST_ACCURACY,
             "Value": f"{advanced_summary['best_accuracy']['model']} (acc={advanced_summary['best_accuracy']['accuracy']:.4f})"},
            {"Experiment": f"{prefix}Advanced", "Metric": BEST_ROC_AUC,
             "Value": f"{advanced_summary['best_auc']['model']} (AUC={advanced_summary['best_auc']['roc_auc']:.4f})"}
        ])

    if results["cv"] is not None and not results["cv"].empty:
        cv_summary = _summarize_cross_validation(results["cv"])
        detailed_summary[f"{dataset_name.lower()}_cross_validation"] = cv_summary
        summary_rows.append({
            "Experiment": f"{prefix}Cross-Validation", "Metric": BEST_F1,
            "Value": f"{cv_summary['best_model']} (F1={cv_summary['f1_score']})"
        })


def create_summary() -> pd.DataFrame:
    summary_rows: List[Dict[str, str]] = []
    detailed_summary: Dict[str, object] = {}

    # Process NSL-KDD results
    print("📊 Processing NSL-KDD results...")
    nsl_results = _load_dataset_results("NSL")
    _add_dataset_summary_rows("NSL-KDD", nsl_results, summary_rows, detailed_summary)

    # Process CIC-IDS-2017 results
    print("📊 Processing CIC-IDS-2017 results...")
    cic_results = _load_dataset_results("CIC")
    _add_dataset_summary_rows("CIC-IDS-2017", cic_results, summary_rows, detailed_summary)

    # Process cross-dataset evaluation
    print("📊 Processing cross-dataset evaluation...")
    nsl_to_cic_df = _load_csv(NSL_TO_CIC_PATH)
    if nsl_to_cic_df is not None and not nsl_to_cic_df.empty:
        forward_summary = _summarize_cross_dataset(nsl_to_cic_df, "NSL→CIC")
        detailed_summary.setdefault("cross_dataset", {})["nsl_to_cic"] = forward_summary
        summary_rows.append({
            "Experiment": "Cross-Dataset Transfer",
            "Metric": "Best Transfer (NSL→CIC)",
            "Value": f"{forward_summary['best_model']} (ratio={forward_summary['transfer_ratio']:.3f}, Δ={forward_summary['relative_drop']:.2f}%)",
        })

    cic_to_nsl_df = _load_csv(CIC_TO_NSL_PATH)
    if cic_to_nsl_df is not None and not cic_to_nsl_df.empty:
        reverse_summary = _summarize_cross_dataset(cic_to_nsl_df, "CIC→NSL")
        detailed_summary.setdefault("cross_dataset", {})["cic_to_nsl"] = reverse_summary
        summary_rows.append({
            "Experiment": "Cross-Dataset Transfer",
            "Metric": "Best Transfer (CIC→NSL)",
            "Value": f"{reverse_summary['best_model']} (ratio={reverse_summary['transfer_ratio']:.3f}, Δ={reverse_summary['relative_drop']:.2f}%)",
        })

    # Process bidirectional analysis
    bidirectional_df = _load_csv(BIDIRECTIONAL_PATH)
    if bidirectional_df is not None and not bidirectional_df.empty:
        bidirectional_summary = _summarize_bidirectional(bidirectional_df)
        detailed_summary["cross_dataset_summary"] = bidirectional_summary
        summary_rows.append({
            "Experiment": "Cross-Dataset Analysis",
            "Metric": "Mean Transfer Ratio",
            "Value": f"{bidirectional_summary['mean_transfer_ratio']:.3f}",
        })
        summary_rows.append({
            "Experiment": "Cross-Dataset Analysis",
            "Metric": "Best Model Overall",
            "Value": f"{bidirectional_summary['best_model']} (avg ratio={bidirectional_summary['best_avg_transfer_ratio']:.3f})",
        })

    # Process harmonized evaluation with incremental training support
    print("📊 Processing harmonized evaluation...")
    harmonized_data = _load_harmonized_results(HARMONIZED_PATH)
    harmonized_summary = _summarize_harmonized(harmonized_data) if harmonized_data else []
    if harmonized_summary:
        detailed_summary["harmonized_evaluation"] = harmonized_summary
        for entry in harmonized_summary:
            training_info = f" ({entry.get('training_method', 'regular')} training)" if entry.get('training_method') else ""
            summary_rows.append({
                "Experiment": "Harmonized Evaluation",
                "Metric": f"{entry['source']}→{entry['target']} Target F1",
                "Value": f"{entry['target_f1']:.4f} (CV F1={entry['cv_f1_mean']:.4f}){training_info}",
            })
    
    # Add metadata from harmonized results
    if harmonized_data:
        metadata = {
            "schema_version": harmonized_data.get("schema_version", "unknown"),
            "memory_mode": harmonized_data.get("memory_mode", "unknown"),
            "training_method": harmonized_data.get("training_method", "unknown"),
            "max_samples": harmonized_data.get("max_samples", "unknown")
        }
        detailed_summary["harmonized_metadata"] = metadata
        summary_rows.append({
            "Experiment": "Harmonized Evaluation",
            "Metric": "Training Configuration",
            "Value": f"Mode: {metadata['memory_mode']}, Method: {metadata['training_method']}",
        })

    # Generate final summary
    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        SUMMARY_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(SUMMARY_OUTPUT_PATH, index=False)
        print("\n📊 EXPERIMENT SUMMARY")
        print(summary_df.to_string(index=False))
        print(f"\n💾 Summary saved to {SUMMARY_OUTPUT_PATH}")
    else:
        print("⚠️ No summary rows generated. Ensure experiments have been executed.")

    # Save detailed JSON summary
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
