#!/usr/bin/env python3
"""Training entry point for advanced models on NSL-KDD and CIC-IDS-2017 with memory adaptation."""

from __future__ import annotations

import sys
import traceback
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


def train_nsl_kdd_advanced() -> bool:
    """Train the advanced model suite on the NSL-KDD dataset."""

    _print_separator("🚀 NSL-KDD Advanced Training")

    try:
        from src.preprocessing import NSLKDDAnalyzer, NSLKDDPreprocessor
        from src.models import AdvancedModels
    except ImportError as exc:  # pragma: no cover - runtime feedback only
        print(f"❌ Import error: {exc}")
        return False

    try:
        analyzer = NSLKDDAnalyzer()
        train_data = analyzer.load_data("KDDTrain+.txt")
        test_data = analyzer.load_data("KDDTest+.txt")

        if train_data is None or test_data is None:
            print("❌ Training aborted: required NSL-KDD dataset files are missing.")
            return False

        print("🔄 Preprocessing NSL-KDD data (SMOTE for balance)...")
        preprocessor = NSLKDDPreprocessor(balance_method="smote")
        X_train, X_val, y_train, y_val = preprocessor.fit_transform(
            train_data, target_type="binary"
        )
        X_test, y_test = preprocessor.transform(test_data, target_type="binary")

        print("\n🤖 Initialising advanced models for NSL-KDD...")
        advanced_models = AdvancedModels(random_state=42)
        available_models = list(advanced_models.models.keys())
        print(f"✅ Models available: {available_models}")

        train_results = advanced_models.train_all(X_train, y_train)
        failed_models = [
            name for name, result in train_results.items() if result.status != "success"
        ]
        if failed_models:
            print(f"⚠️ Models failed to train: {failed_models}")

        print("\n📊 Validation performance on NSL-KDD")
        val_results = advanced_models.evaluate_all(X_val, y_val, dataset="validation")
        if val_results.empty:
            print("❌ No validation results available for NSL-KDD advanced models.")
            return False

        summary_cols = [
            "model_name",
            "dataset",
            "accuracy",
            "f1_score",
            "precision",
            "recall",
            "roc_auc",
        ]
        print(val_results[summary_cols].round(4))

        best_model_name = val_results.iloc[0]["model_name"]
        print(f"\n🏆 Best validation model on NSL-KDD: {best_model_name}")

        print("\n🧪 Testing best NSL-KDD model on hold-out set")
        test_metrics = advanced_models.evaluate_model(
            best_model_name,
            X_test,
            y_test,
            dataset="test",
        )
        for metric in ["accuracy", "f1_score", "precision", "recall", "roc_auc"]:
            value = test_metrics.get(metric)
            if value is not None:
                print(f"   {metric.title():<10}: {value:.4f}")
            else:
                print(f"   {metric.title():<10}: n/a")

        print("\n💾 Persisting NSL-KDD advanced artefacts...")
        models_dir = PROJECT_ROOT / "data" / "models" / "advanced"
        results_dir = PROJECT_ROOT / "data" / "results" / "nsl"
        advanced_models.save_models(
            models_dir,
            results_dir=results_dir,
            results_filename="advanced_results.csv",
            dataset_suffix="_nsl",
        )

        print("✅ NSL-KDD advanced training pipeline complete!")
        return True
    except Exception as exc:  # pragma: no cover - runtime feedback only
        print(f"❌ Error during NSL-KDD advanced training: {exc}")
        traceback.print_exc()
        return False


def train_cic_advanced() -> bool:
    """Train the advanced model suite on the CIC-IDS-2017 dataset with FULL DATASET for scientific accuracy."""

    # FORCE FULL DATASET FOR SCIENTIFIC PAPER - Override memory configuration
    print("🔬 SCIENTIFIC MODE: Forcing full dataset usage for publication accuracy")
    use_full = True

    title = "🚀 CIC-IDS-2017 Advanced Training (FULL DATASET - Scientific Mode)"

    _print_separator(title)

    try:
        from src.preprocessing import CICIDSPreprocessor
        from src.models import AdvancedModels
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

        # Create train/validation/test splits (60/20/20)
        with MemoryMonitor("Dataset Splitting"):
            X_train, X_temp, y_train, y_temp = train_test_split(
                X_sample,
                y_sample,
                test_size=0.4,
                random_state=42,
                stratify=y_sample,
            )
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp,
                y_temp,
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
            print("\n🤖 Initialising advanced models for CIC-IDS-2017...")
            advanced_models = AdvancedModels(random_state=42)
            available_models = list(advanced_models.models.keys())
            print(f"✅ Models available: {available_models}")

            # SCIENTIFIC MODE: Using full model configurations
            print("💪 Using full model configurations for scientific accuracy")

            train_results = advanced_models.train_all(X_train, y_train)
            failed_models = [
                name
                for name, result in train_results.items()
                if result.status != "success"
            ]
            if failed_models:
                print(f"⚠️ Models failed to train: {failed_models}")

        with MemoryMonitor("Model Evaluation"):
            print("\n📊 Validation performance on CIC-IDS-2017")
            val_results = advanced_models.evaluate_all(
                X_val, y_val, dataset="validation"
            )
            if val_results.empty:
                print(
                    "❌ No validation results available for CIC-IDS-2017 advanced models."
                )
                return False

        summary_cols = [
            "model_name",
            "dataset",
            "accuracy",
            "f1_score",
            "precision",
            "recall",
            "roc_auc",
        ]
        print(val_results[summary_cols].round(4))

        best_model_name = val_results.iloc[0]["model_name"]
        print(f"\n🏆 Best validation model on CIC-IDS-2017: {best_model_name}")

        print("\n🧪 Testing best CIC model on hold-out split")
        test_metrics = advanced_models.evaluate_model(
            best_model_name,
            X_test,
            y_test,
            dataset="test",
        )
        for metric in ["accuracy", "f1_score", "precision", "recall", "roc_auc"]:
            value = test_metrics.get(metric)
            if value is not None:
                print(f"   {metric.title():<10}: {value:.4f}")
            else:
                print(f"   {metric.title():<10}: n/a")

        print("\n💾 Persisting CIC-IDS-2017 advanced artefacts...")
        models_dir = PROJECT_ROOT / "data" / "models" / "cic_advanced"
        results_dir = PROJECT_ROOT / "data" / "results" / "cic"
        advanced_models.save_models(
            models_dir,
            results_dir=results_dir,
            results_filename="cic_advanced_results.csv",
            dataset_suffix="_cic",
        )

        print("✅ CIC-IDS-2017 advanced training pipeline complete!")
        return True
    except Exception as exc:  # pragma: no cover - runtime feedback only
        print(f"❌ Error during CIC-IDS-2017 advanced training: {exc}")
        traceback.print_exc()
        return False


def main() -> bool:
    print("🚀 Advanced Model Training Pipeline")
    print("=" * 60)

    success_nsl = train_nsl_kdd_advanced()
    success_cic = train_cic_advanced()

    print("\n" + "=" * 60)
    if success_nsl and success_cic:
        print("✅ Advanced training completed for both NSL-KDD and CIC-IDS-2017!")
    elif success_nsl:
        print("⚠️ Advanced training completed for NSL-KDD only.")
    elif success_cic:
        print("⚠️ Advanced training completed for CIC-IDS-2017 only.")
    else:
        print("❌ Advanced training failed for all datasets.")

    return success_nsl and success_cic


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
