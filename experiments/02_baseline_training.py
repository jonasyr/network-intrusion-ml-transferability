#!/usr/bin/env python3
"""Comprehensive baseline training for NSL-KDD and CIC-IDS-2017 datasets with memory adaptation."""

from __future__ import annotations

import sys
import traceback
import gc
from pathlib import Path

from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.utils import (
    get_memory_adaptive_config,
    MemoryMonitor,
    optimize_memory_usage,
    get_optimal_sample_size,
)


def _print_separator(title: str) -> None:
    print()
    print(title)
    print("=" * len(title))


def train_nsl_kdd_baseline() -> bool:
    """Run the baseline pipeline on the NSL-KDD dataset."""

    _print_separator("🚀 NSL-KDD Baseline Training")

    try:
        from src.preprocessing import NSLKDDAnalyzer, NSLKDDPreprocessor
        from src.models import BaselineModels
    except ImportError as exc:  # pragma: no cover - runtime feedback only
        print(f"❌ Import error: {exc}")
        return False

    try:
        analyzer = NSLKDDAnalyzer()
        train_data = analyzer.load_data("KDDTrain+.txt")
        test_data = analyzer.load_data("KDDTest+.txt")

        if train_data is None or test_data is None:
            print("❌ Training aborted: NSL-KDD data not found.")
            return False

        print("🔄 Preprocessing NSL-KDD data (undersampling for balance)...")
        preprocessor = NSLKDDPreprocessor(balance_method="undersample")
        X_train, X_val, y_train, y_val = preprocessor.fit_transform(train_data)
        X_test, y_test = preprocessor.transform(test_data)

        print("\n🤖 Training ALL baseline models on NSL-KDD...")
        print("   🔬 SCIENTIFIC MODE: Training ALL models including SVM and KNN")
        baseline = BaselineModels()
        exclude_models = []  # NO MODEL EXCLUSIONS for scientific paper
        baseline.train_all(X_train, y_train, exclude_models=exclude_models)

        print("\n📊 Validation performance on NSL-KDD...")
        val_results = baseline.evaluate_all(X_val, y_val)
        if val_results.empty:
            print("❌ No baseline models produced validation results for NSL-KDD.")
            return False

        print("\n🏆 NSL-KDD validation leaderboard:")
        print(val_results[["model_name", "accuracy", "f1_score", "precision", "recall"]].round(3))

        best_model_name = val_results.iloc[0]["model_name"]
        print(f"\n🎯 Evaluating best NSL-KDD model ({best_model_name}) on the official test set...")
        test_metrics = baseline.evaluate_model(best_model_name, X_test, y_test)
        for metric in ["accuracy", "f1_score", "precision", "recall"]:
            value = test_metrics.get(metric, float("nan"))
            print(f"   {metric.title():<10}: {value:.3f}")

        print("\n💾 Persisting NSL-KDD baseline artefacts...")
        models_dir = PROJECT_ROOT / "data" / "models" / "baseline"
        results_dir = PROJECT_ROOT / "data" / "results" / "nsl"
        baseline.save_models(str(models_dir), results_dir=str(results_dir), dataset_suffix="_nsl")
        preprocessor.save(str(models_dir / "nsl_preprocessor.pkl"))

        print("✅ NSL-KDD baseline training complete!")
        return True
    except Exception as exc:  # pragma: no cover - runtime feedback only
        print(f"❌ Error during NSL-KDD baseline training: {exc}")
        traceback.print_exc()
        return False


def train_cic_baseline() -> bool:
    """Run the baseline pipeline on the CIC-IDS-2017 dataset with FULL DATASET for scientific accuracy."""

    # FORCE FULL DATASET FOR SCIENTIFIC PAPER - Override memory configuration
    print("🔬 SCIENTIFIC MODE: Forcing full dataset usage for publication accuracy")
    use_full = True
    
    title = "🚀 CIC-IDS-2017 Baseline Training (FULL DATASET - Scientific Mode)"
    
    _print_separator(title)

    try:
        from src.preprocessing import CICIDSPreprocessor
        from src.models import BaselineModels
    except ImportError as exc:  # pragma: no cover - runtime feedback only
        print(f"❌ Import error: {exc}")
        return False

    preprocessor = CICIDSPreprocessor()

    try:
        with MemoryMonitor("Dataset Loading"):
            if use_full:
                print("📁 Loading FULL CIC-IDS-2017 dataset...")
                cic_data = preprocessor.load_data(use_full_dataset=True)
            else:
                print("📁 Loading CIC-IDS-2017 sample dataset...")
                cic_data = preprocessor.load_data(use_full_dataset=False)
                
        if cic_data is None:
            print("❌ CIC-IDS-2017 data could not be loaded.")
            return False

        with MemoryMonitor("Feature Preparation"):
            print("🔄 Preparing CIC-IDS-2017 features and labels...")
            X_full, y_full = preprocessor.fit_transform(cic_data)
            
            # Free up memory from raw data
            del cic_data
            optimize_memory_usage()

        # SCIENTIFIC MODE: Always use FULL dataset
        print("💪 Using FULL dataset for scientific accuracy")
        X_sample, y_sample = X_full, y_full
        
        # Split into train/val/test (60/20/20)
        with MemoryMonitor("Dataset Splitting"):
            X_train, X_temp, y_train, y_temp = train_test_split(
                X_sample, y_sample,
                test_size=0.4,
                random_state=42,
                stratify=y_sample,
            )
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp,
                test_size=0.5,
                random_state=42,
                stratify=y_temp,
            )
            
            # Free up memory from intermediate datasets
            del X_sample, y_sample, X_temp, y_temp
            if not use_full:
                del X_full, y_full
            optimize_memory_usage()
            
            print(f"📊 Final dataset sizes:")
            print(f"   Training: {X_train.shape}")
            print(f"   Validation: {X_val.shape}")
            print(f"   Test: {X_test.shape}")

        with MemoryMonitor("Model Training"):
            print("\n🤖 Training ALL baseline models on CIC-IDS-2017...")
            print("   🔬 SCIENTIFIC MODE: Training ALL models including SVM and KNN")
            baseline = BaselineModels()
            
            # NO MODEL EXCLUSIONS for scientific paper
            exclude_models = []
            
            baseline.train_all(X_train, y_train, exclude_models=exclude_models)

        print("\n📊 Validation performance on CIC-IDS-2017...")
        val_results = baseline.evaluate_all(X_val, y_val)
        if val_results.empty:
            print("❌ No baseline models produced validation results for CIC-IDS-2017.")
            return False

        print("\n🏆 CIC-IDS-2017 validation leaderboard:")
        print(val_results[["model_name", "accuracy", "f1_score", "precision", "recall"]].round(3))

        best_model_name = val_results.iloc[0]["model_name"]
        print(f"\n🎯 Evaluating best CIC baseline model ({best_model_name}) on the hold-out test split...")
        test_metrics = baseline.evaluate_model(best_model_name, X_test, y_test)
        for metric in ["accuracy", "f1_score", "precision", "recall"]:
            value = test_metrics.get(metric, float("nan"))
            print(f"   {metric.title():<10}: {value:.3f}")

        print("\n💾 Persisting CIC-IDS-2017 baseline artefacts...")
        cic_models_dir = PROJECT_ROOT / "data" / "models" / "cic_baseline"
        results_dir = PROJECT_ROOT / "data" / "results" / "cic"
        baseline.save_models(str(cic_models_dir), results_dir=str(results_dir), dataset_suffix="_cic")

        print("✅ CIC-IDS-2017 baseline training complete!")
        return True
    except Exception as exc:  # pragma: no cover - runtime feedback only
        print(f"❌ Error during CIC-IDS-2017 baseline training: {exc}")
        traceback.print_exc()
        return False


def main() -> bool:
    print("🚀 Baseline Training Pipeline")
    print("=" * 60)

    success_nsl = train_nsl_kdd_baseline()
    success_cic = train_cic_baseline()

    print("\n" + "=" * 60)
    if success_nsl and success_cic:
        print("✅ Baseline training completed for both NSL-KDD and CIC-IDS-2017!")
    elif success_nsl:
        print("⚠️ Baseline training completed for NSL-KDD only.")
    elif success_cic:
        print("⚠️ Baseline training completed for CIC-IDS-2017 only.")
    else:
        print("❌ Baseline training failed for all datasets.")

    return success_nsl and success_cic


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
