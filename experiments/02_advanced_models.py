#!/usr/bin/env python3
"""Training entry point for advanced NSL-KDD models."""

import sys
from pathlib import Path

# Ensure the src package is importable when running as a script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.append(str(PROJECT_ROOT / "src"))


def main() -> None:
    print("🚀 Advanced Model Training")
    print("=" * 60)

    try:
        from nsl_kdd_analyzer import NSLKDDAnalyzer
        from data.preprocessor import NSLKDDPreprocessor
        from models.advanced import AdvancedModels
    except ImportError as exc:  # pragma: no cover - runtime feedback only
        print(f"❌ Import error: {exc}")
        print("💡 Ensure the repository structure is intact and dependencies installed.")
        return

    analyzer = NSLKDDAnalyzer()
    train_data = analyzer.load_data("KDDTrain+_20Percent.txt")
    test_data = analyzer.load_data("KDDTest+.txt")

    if train_data is None or test_data is None:
        print("❌ Training aborted: required dataset files are missing.")
        return

    print("\n🔄 Preprocessing data (SMOTE for class balance)...")
    preprocessor = NSLKDDPreprocessor(balance_method="smote")
    X_train, X_val, y_train, y_val = preprocessor.fit_transform(train_data, target_type="binary")
    X_test, y_test = preprocessor.transform(test_data, target_type="binary")

    print("\n🤖 Initialising advanced models...")
    advanced_models = AdvancedModels(random_state=42)

    available_models = list(advanced_models.models.keys())
    print(f"✅ Models available: {available_models}")

    print("\n⚙️ Training phase")
    train_results = advanced_models.train_all(X_train, y_train)

    failed_models = [name for name, result in train_results.items() if result.status != "success"]
    if failed_models:
        print(f"⚠️ Models failed to train: {failed_models}")

    print("\n📊 Validation performance")
    val_results = advanced_models.evaluate_all(X_val, y_val, dataset="validation")
    if not val_results.empty:
        summary_cols = ["model_name", "dataset", "accuracy", "f1_score", "precision", "recall", "roc_auc"]
        print(val_results[summary_cols].round(4))

        best_model_name = val_results.iloc[0]["model_name"]
        print(f"\n🏆 Best validation model: {best_model_name}")

        print("\n🧪 Testing best model on hold-out set")
        test_metrics = advanced_models.evaluate_model(
            best_model_name,
            X_test,
            y_test,
            dataset="test",
        )
        for metric in ["accuracy", "f1_score", "precision", "recall", "roc_auc"]:
            value = test_metrics.get(metric)
            if value is not None:
                print(f"   {metric.title()}: {value:.4f}")
            else:
                print(f"   {metric.title()}: n/a")
    else:
        print("⚠️ No validation results available. Skipping test evaluation.")

    output_dir = PROJECT_ROOT / "data" / "models" / "advanced"
    print(f"\n💾 Saving models and validation results to {output_dir}")
    advanced_models.save_models(output_dir, results_filename="advanced_results.csv")

    print("\n✅ Advanced training pipeline complete!")


if __name__ == "__main__":
    main()
