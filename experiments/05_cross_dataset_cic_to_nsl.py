#!/usr/bin/env python3
"""Cross-dataset evaluation: train on CIC-IDS-2017 and test on NSL-KDD."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from src.features import FeatureAligner
from src.metrics.cross_dataset_metrics import (
    calculate_generalization_gap,
    calculate_relative_performance_drop,
    calculate_transfer_ratio,
    compute_domain_divergence,
)
from src.preprocessing import CICIDSPreprocessor, NSLKDDAnalyzer, NSLKDDPreprocessor

try:  # Optional dependencies
    import xgboost as xgb
except ImportError:  # pragma: no cover - optional component
    xgb = None

try:  # Optional dependencies
    import lightgbm as lgb
except ImportError:  # pragma: no cover - optional component
    lgb = None

RANDOM_STATE = 42
RESULTS_PATH = project_root / "data/results/reverse_cross_dataset_evaluation_fixed.csv"


def _format_percentage(value: float) -> str:
    return f"{value * 100:.2f}%"


def run_reverse_cross_dataset_evaluation() -> bool:
    """Execute the CIC→NSL cross-dataset experiment."""

    print("🔄 CROSS-DATASET EVALUATION (CIC-IDS-2017 → NSL-KDD)")
    print("=" * 80)

    analyzer = NSLKDDAnalyzer()
    nsl_preprocessor = NSLKDDPreprocessor(balance_method="smote")
    cic_preprocessor = CICIDSPreprocessor()

    print("\n📁 Loading datasets…")
    cic_data = cic_preprocessor.load_data(use_full_dataset=False)
    nsl_train = analyzer.load_data("KDDTrain+_20Percent.txt")
    nsl_test = analyzer.load_data("KDDTest+.txt")

    if cic_data is None or nsl_train is None or nsl_test is None:
        print("❌ Failed to load one or more datasets")
        return False

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

    try:
        common = aligner.extract_common_features(
            X_nsl_test,
            X_cic_train,
            nsl_preprocessor.feature_names,
            cic_preprocessor.feature_names,
        )
    except ValueError as exc:
        print(f"   ❌ Failed to identify overlapping features: {exc}")
        return False

    selected_nsl_features = common.metadata["source_features"]
    selected_cic_features = common.metadata["target_features"]
    print(
        "   • Identified "
        f"{len(selected_nsl_features)} semantically aligned feature pairs"
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

    X_cic_train_aligned = pca_alignment.source
    X_cic_val_common = aligner.transform_dataset(
        X_cic_val,
        cic_preprocessor.feature_names,
        selected_cic_features,
    )
    X_cic_val_aligned = pca.transform(scaler.transform(X_cic_val_common))
    X_nsl_test_aligned = pca_alignment.target

    print(
        f"   • PCA alignment to {X_cic_train_aligned.shape[1]} dimensions "
        f"(explained variance: {pca_alignment.metadata['explained_variance_ratio'].sum():.2f})"
    )

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

    results: List[Dict[str, float]] = []

    for model_name, model in models.items():
        print(f"\n🤖 Training {model_name} on aligned CIC-IDS-2017 features…")
        start_time = time.time()
        model.fit(X_cic_train_aligned, y_cic_train)
        training_time = time.time() - start_time
        print(f"   ✓ Training completed in {training_time:.2f}s")

        print("   📊 Evaluating within CIC-IDS-2017…")
        y_pred_cic = model.predict(X_cic_val_aligned)
        source_accuracy = accuracy_score(y_cic_val, y_pred_cic)
        source_precision = precision_score(y_cic_val, y_pred_cic, zero_division=0)
        source_recall = recall_score(y_cic_val, y_pred_cic, zero_division=0)
        source_f1 = f1_score(y_cic_val, y_pred_cic, zero_division=0)

        print("   🔄 Evaluating on NSL-KDD…")
        y_pred_nsl = model.predict(X_nsl_test_aligned)
        target_accuracy = accuracy_score(y_nsl_test, y_pred_nsl)
        target_precision = precision_score(y_nsl_test, y_pred_nsl, zero_division=0)
        target_recall = recall_score(y_nsl_test, y_pred_nsl, zero_division=0)
        target_f1 = f1_score(y_nsl_test, y_pred_nsl, zero_division=0)

        gap = calculate_generalization_gap(source_accuracy, target_accuracy)
        relative_drop = calculate_relative_performance_drop(source_accuracy, target_accuracy)
        transfer_ratio = calculate_transfer_ratio(source_accuracy, target_accuracy)

        print(f"      Source accuracy: {_format_percentage(source_accuracy)}")
        print(f"      Target accuracy: {_format_percentage(target_accuracy)}")
        print(f"      Generalization gap: {gap:.4f}")
        print(f"      Relative drop: {relative_drop:.2f}%")
        print(f"      Transfer ratio: {transfer_ratio:.4f}")

        results.append(
            {
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
                "Domain_Divergence": round(float(domain_divergence), 4),
                "Training_Time_s": round(training_time, 2),
                "Aligned_Features": X_cic_train_aligned.shape[1],
            }
        )

    if not results:
        print("❌ No models were evaluated")
        return False

    results_df = pd.DataFrame(results)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(RESULTS_PATH, index=False)

    print("\n📊 CROSS-DATASET RESULTS (CIC-IDS-2017 → NSL-KDD)")
    print(results_df.to_string(index=False))
    print(f"\n💾 Results saved to {RESULTS_PATH}")

    best_model = results_df.sort_values("Transfer_Ratio", ascending=False).iloc[0]
    print("\n🔍 Key Findings")
    print(
        f"   • Best transfer performance: {best_model['Model']} (transfer ratio {best_model['Transfer_Ratio']:.3f})"
    )
    print(
        f"   • Average relative performance drop: {results_df['Relative_Drop_%'].mean():.2f}%"
    )
    print(f"   • Domain divergence (lower is better): {domain_divergence:.4f}")

    return True


def main() -> bool:
    success = run_reverse_cross_dataset_evaluation()
    if success:
        print("\n🎯 Reverse cross-dataset evaluation complete!")
    return success


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
