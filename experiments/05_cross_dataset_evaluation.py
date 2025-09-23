#!/usr/bin/env python3
"""Unified cross-dataset evaluation for NSL-KDD and CIC-IDS-2017."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

current_dir = Path(__file__).parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.features import FeatureAligner
from src.metrics.cross_dataset_metrics import (  # noqa: E402
    calculate_generalization_gap,
    calculate_relative_performance_drop,
    calculate_transfer_ratio,
    compute_domain_divergence,
)
from src.preprocessing import (  # noqa: E402
    CICIDSPreprocessor,
    NSLKDDAnalyzer,
    NSLKDDPreprocessor,
)

try:  # Optional dependencies
    import xgboost as xgb
except ImportError:  # pragma: no cover - optional component
    xgb = None

try:  # Optional dependencies
    import lightgbm as lgb
except ImportError:  # pragma: no cover - optional component
    lgb = None

RANDOM_STATE = 42
RESULTS_DIR = project_root / "data/results"
FORWARD_RESULTS_PATH = RESULTS_DIR / "cross_dataset_evaluation_fixed.csv"
REVERSE_RESULTS_PATH = RESULTS_DIR / "reverse_cross_dataset_evaluation_fixed.csv"
BIDIRECTIONAL_RESULTS_PATH = RESULTS_DIR / "bidirectional_cross_dataset_analysis.csv"


@dataclass
class DatasetBundle:
    """Container for aligned datasets used during transfer evaluation."""

    train_features: np.ndarray
    train_labels: np.ndarray
    source_eval_features: np.ndarray
    source_eval_labels: np.ndarray
    target_eval_features: np.ndarray
    target_eval_labels: np.ndarray
    domain_divergence: float
    aligned_feature_count: int


def _format_percentage(value: float) -> str:
    return f"{value * 100:.2f}%"


def _build_model_suite() -> Dict[str, object]:
    models: Dict[str, object] = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=25,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    }

    if xgb is not None:
        models["XGBoost"] = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            objective="binary:logistic",
            eval_metric="logloss",
        )
    else:
        print("   ⚠️ XGBoost not available – skipping")

    if lgb is not None:
        models["LightGBM"] = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    else:
        print("   ⚠️ LightGBM not available – skipping")

    return models


def _evaluate_model(
    model_name: str,
    model: object,
    bundle: DatasetBundle,
    source_label: str,
    target_label: str,
) -> Dict[str, float]:
    print(f"\n🤖 Training {model_name} on aligned {source_label} features…")
    start_time = time.time()
    model.fit(bundle.train_features, bundle.train_labels)
    training_time = time.time() - start_time
    print(f"   ✓ Training completed in {training_time:.2f}s")

    print(f"   📊 Evaluating on {source_label} hold-out split…")
    y_pred_source = model.predict(bundle.source_eval_features)
    source_accuracy = accuracy_score(bundle.source_eval_labels, y_pred_source)
    source_precision = precision_score(bundle.source_eval_labels, y_pred_source, zero_division=0)
    source_recall = recall_score(bundle.source_eval_labels, y_pred_source, zero_division=0)
    source_f1 = f1_score(bundle.source_eval_labels, y_pred_source, zero_division=0)

    print(f"   🔄 Evaluating on {target_label} dataset…")
    y_pred_target = model.predict(bundle.target_eval_features)
    target_accuracy = accuracy_score(bundle.target_eval_labels, y_pred_target)
    target_precision = precision_score(bundle.target_eval_labels, y_pred_target, zero_division=0)
    target_recall = recall_score(bundle.target_eval_labels, y_pred_target, zero_division=0)
    target_f1 = f1_score(bundle.target_eval_labels, y_pred_target, zero_division=0)

    gap = calculate_generalization_gap(source_accuracy, target_accuracy)
    relative_drop = calculate_relative_performance_drop(source_accuracy, target_accuracy)
    transfer_ratio = calculate_transfer_ratio(source_accuracy, target_accuracy)

    print(f"      Source accuracy: {_format_percentage(source_accuracy)}")
    print(f"      Target accuracy: {_format_percentage(target_accuracy)}")
    print(f"      Generalization gap: {gap:.4f}")
    print(f"      Relative drop: {relative_drop:.2f}%")
    print(f"      Transfer ratio: {transfer_ratio:.4f}")

    return {
        "Model": model_name,
        "Source_Accuracy": round(float(source_accuracy), 4),
        "Source_Precision": round(float(source_precision), 4),
        "Source_Recall": round(float(source_recall), 4),
        "Source_F1": round(float(source_f1), 4),
        "Target_Accuracy": round(float(target_accuracy), 4),
        "Target_Precision": round(float(target_precision), 4),
        "Target_Recall": round(float(target_recall), 4),
        "Target_F1": round(float(target_f1), 4),
        "Generalization_Gap": round(float(gap), 4),
        "Relative_Drop_%": round(float(relative_drop), 2),
        "Transfer_Ratio": round(float(transfer_ratio), 4),
        "Domain_Divergence": round(float(bundle.domain_divergence), 4),
        "Training_Time_s": round(training_time, 2),
        "Aligned_Features": int(bundle.aligned_feature_count),
    }


def _align_nsl_to_cic() -> DatasetBundle:
    print("🚀 CROSS-DATASET EVALUATION (NSL-KDD → CIC-IDS-2017)")
    print("=" * 80)

    analyzer = NSLKDDAnalyzer()
    nsl_preprocessor = NSLKDDPreprocessor(balance_method="smote")
    cic_preprocessor = CICIDSPreprocessor()

    print("\n📁 Loading datasets…")
    nsl_train = analyzer.load_data("KDDTrain+.txt")
    nsl_test = analyzer.load_data("KDDTest+.txt")
    cic_data = cic_preprocessor.load_data(use_full_dataset=False)

    if nsl_train is None or nsl_test is None or cic_data is None:
        raise RuntimeError("Failed to load one or more datasets")

    print("\n🔄 Preprocessing datasets…")
    X_nsl_train, X_nsl_val, y_nsl_train, y_nsl_val = nsl_preprocessor.fit_transform(nsl_train)
    X_nsl_test, y_nsl_test = nsl_preprocessor.transform(nsl_test)
    X_cic, y_cic = cic_preprocessor.fit_transform(cic_data)

    X_nsl_full = np.vstack([X_nsl_train, X_nsl_val])
    y_nsl_full = np.hstack([y_nsl_train, y_nsl_val])

    print("\n📐 Aligning feature spaces…")
    aligner = FeatureAligner()

    common = aligner.extract_common_features(
        X_nsl_full,
        X_cic,
        nsl_preprocessor.feature_names,
        cic_preprocessor.feature_names,
    )

    feature_pairs = common.metadata.get("feature_pairs", [])
    if feature_pairs:
        sample_pair = feature_pairs[0]
        print(
            "   • Example mapping: "
            f"{sample_pair['source_feature']} ↔ {sample_pair['target_feature']} "
            f"({sample_pair['semantic_feature']})"
        )

    statistical_alignment = aligner.statistical_alignment(common.source, common.target)
    domain_divergence = compute_domain_divergence(
        statistical_alignment.source, statistical_alignment.target
    )
    print(f"   • Domain divergence (Wasserstein distance): {domain_divergence:.4f}")

    n_components = min(20, common.source.shape[1])
    pca_alignment = aligner.pca_alignment(
        common.source, common.target, n_components=n_components
    )
    scaler = pca_alignment.metadata["scaler"]
    pca = pca_alignment.metadata["pca"]

    selected_nsl_features = common.metadata["source_features"]
    X_nsl_test_common = aligner.transform_dataset(
        X_nsl_test,
        nsl_preprocessor.feature_names,
        selected_nsl_features,
    )
    X_nsl_test_aligned = pca.transform(scaler.transform(X_nsl_test_common))
    X_cic_aligned = pca_alignment.target
    X_nsl_aligned = pca_alignment.source

    print(
        f"   • PCA alignment to {X_nsl_aligned.shape[1]} dimensions "
        f"(explained variance: {pca_alignment.metadata['explained_variance_ratio'].sum():.2f})"
    )

    return DatasetBundle(
        train_features=X_nsl_aligned,
        train_labels=y_nsl_full,
        source_eval_features=X_nsl_test_aligned,
        source_eval_labels=y_nsl_test,
        target_eval_features=X_cic_aligned,
        target_eval_labels=y_cic,
        domain_divergence=domain_divergence,
        aligned_feature_count=X_nsl_aligned.shape[1],
    )


def _align_cic_to_nsl() -> DatasetBundle:
    print("\n🔄 CROSS-DATASET EVALUATION (CIC-IDS-2017 → NSL-KDD)")
    print("=" * 80)

    analyzer = NSLKDDAnalyzer()
    nsl_preprocessor = NSLKDDPreprocessor(balance_method="smote")
    cic_preprocessor = CICIDSPreprocessor()

    print("\n📁 Loading datasets…")
    cic_data = cic_preprocessor.load_data(use_full_dataset=False)
    nsl_train = analyzer.load_data("KDDTrain+.txt")
    nsl_test = analyzer.load_data("KDDTest+.txt")

    if cic_data is None or nsl_train is None or nsl_test is None:
        raise RuntimeError("Failed to load one or more datasets")

    print("\n🔄 Preprocessing datasets…")
    X_cic, y_cic = cic_preprocessor.fit_transform(cic_data)
    X_cic_train, X_cic_val, y_cic_train, y_cic_val = train_test_split(
        X_cic,
        y_cic,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_cic,
    )

    _ = nsl_preprocessor.fit_transform(nsl_train)
    X_nsl_test, y_nsl_test = nsl_preprocessor.transform(nsl_test)

    print("\n📐 Aligning feature spaces…")
    aligner = FeatureAligner()

    common = aligner.extract_common_features(
        X_nsl_test,
        X_cic_train,
        nsl_preprocessor.feature_names,
        cic_preprocessor.feature_names,
    )

    feature_pairs = common.metadata.get("feature_pairs", [])
    if feature_pairs:
        sample_pair = feature_pairs[0]
        print(
            "   • Example mapping: "
            f"{sample_pair['source_feature']} ↔ {sample_pair['target_feature']} "
            f"({sample_pair['semantic_feature']})"
        )

    cic_train_common = common.target
    nsl_test_common = common.source

    statistical_alignment = aligner.statistical_alignment(
        cic_train_common, nsl_test_common
    )
    domain_divergence = compute_domain_divergence(
        statistical_alignment.source, statistical_alignment.target
    )
    print(f"   • Domain divergence (Wasserstein distance): {domain_divergence:.4f}")

    n_components = min(20, cic_train_common.shape[1])
    pca_alignment = aligner.pca_alignment(
        cic_train_common, nsl_test_common, n_components=n_components
    )
    scaler = pca_alignment.metadata["scaler"]
    pca = pca_alignment.metadata["pca"]

    selected_cic_features = common.metadata["target_features"]
    X_cic_val_common = aligner.transform_dataset(
        X_cic_val,
        cic_preprocessor.feature_names,
        selected_cic_features,
    )
    X_cic_val_aligned = pca.transform(scaler.transform(X_cic_val_common))
    X_cic_train_aligned = pca_alignment.source
    X_nsl_test_aligned = pca_alignment.target

    print(
        f"   • PCA alignment to {X_cic_train_aligned.shape[1]} dimensions "
        f"(explained variance: {pca_alignment.metadata['explained_variance_ratio'].sum():.2f})"
    )

    return DatasetBundle(
        train_features=X_cic_train_aligned,
        train_labels=y_cic_train,
        source_eval_features=X_cic_val_aligned,
        source_eval_labels=y_cic_val,
        target_eval_features=X_nsl_test_aligned,
        target_eval_labels=y_nsl_test,
        domain_divergence=domain_divergence,
        aligned_feature_count=X_cic_train_aligned.shape[1],
    )


def _run_direction(
    bundle: DatasetBundle,
    source_label: str,
    target_label: str,
) -> pd.DataFrame:
    models = _build_model_suite()
    results: List[Dict[str, float]] = []

    for model_name, model in models.items():
        metrics = _evaluate_model(model_name, model, bundle, source_label, target_label)
        results.append(metrics)

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results)


def _create_bidirectional_summary(forward_df: pd.DataFrame, reverse_df: pd.DataFrame) -> pd.DataFrame:
    if forward_df.empty or reverse_df.empty:
        return pd.DataFrame()

    forward = forward_df.rename(
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

    reverse = reverse_df.rename(
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

    combined = forward.merge(reverse, on="Model", suffixes=("_forward", "_reverse"))

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

    return combined


def run_cross_dataset_pipeline() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    try:
        forward_bundle = _align_nsl_to_cic()
        forward_results = _run_direction(forward_bundle, "NSL-KDD", "CIC-IDS-2017")
    except Exception as exc:  # pragma: no cover - runtime failures
        print(f"❌ NSL→CIC evaluation failed: {exc}")
        forward_results = pd.DataFrame()

    if not forward_results.empty:
        FORWARD_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        forward_results.to_csv(FORWARD_RESULTS_PATH, index=False)
        print("\n📊 CROSS-DATASET RESULTS (NSL-KDD → CIC-IDS-2017)")
        print(forward_results.to_string(index=False))
        print(f"\n💾 Results saved to {FORWARD_RESULTS_PATH}")

    try:
        reverse_bundle = _align_cic_to_nsl()
        reverse_results = _run_direction(reverse_bundle, "CIC-IDS-2017", "NSL-KDD")
    except Exception as exc:  # pragma: no cover - runtime failures
        print(f"❌ CIC→NSL evaluation failed: {exc}")
        reverse_results = pd.DataFrame()

    if not reverse_results.empty:
        REVERSE_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        reverse_results.to_csv(REVERSE_RESULTS_PATH, index=False)
        print("\n📊 CROSS-DATASET RESULTS (CIC-IDS-2017 → NSL-KDD)")
        print(reverse_results.to_string(index=False))
        print(f"\n💾 Results saved to {REVERSE_RESULTS_PATH}")

    summary_df = _create_bidirectional_summary(forward_results, reverse_results)
    if not summary_df.empty:
        BIDIRECTIONAL_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(BIDIRECTIONAL_RESULTS_PATH, index=False)

        print("\n📊 BIDIRECTIONAL CROSS-DATASET ANALYSIS")
        print("=" * 80)
        print(summary_df.round(4).to_string(index=False))

        best_transfer = summary_df.sort_values("Avg_Transfer_Ratio", ascending=False).iloc[0]
        print("\n🔍 Key Insights")
        print(
            f"   • Best average transfer: {best_transfer['Model']} (ratio {best_transfer['Avg_Transfer_Ratio']:.3f})"
        )
        print(
            f"   • Mean generalization gap: {summary_df['Avg_Gap'].mean():.4f}"
            f" | Mean relative drop: {summary_df['Avg_Relative_Drop'].mean():.2f}%"
        )
        print(
            "   • Most symmetric transfer: "
            f"{summary_df.sort_values('Transfer_Asymmetry').iloc[0]['Model']}"
        )
        print(f"\n💾 Combined results saved to {BIDIRECTIONAL_RESULTS_PATH}")

    return forward_results, reverse_results, summary_df


def main() -> bool:
    forward_results, reverse_results, summary_df = run_cross_dataset_pipeline()
    success = not forward_results.empty and not reverse_results.empty

    if success:
        print("\n🎯 Cross-dataset evaluation pipeline complete!")
    else:
        print("\n⚠️ Cross-dataset evaluation finished with warnings. Check logs above.")

    if summary_df.empty:
        print("⚠️ Combined summary not generated.")

    return success


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
