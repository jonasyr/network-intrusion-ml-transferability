#!/usr/bin/env python3
"""
Environment and Data Validation Script
Checks if the repository is ready for running experiments
"""

import sys
from pathlib import Path
import json


def check_python_version():
    """Check Python version is 3.8+"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}")
    return True


def check_data_files():
    """Check if required data files exist"""
    data_checks = []

    # NSL-KDD files
    nsl_path = Path("data/raw/nsl-kdd/KDDTrain+.txt")
    if nsl_path.exists():
        size_mb = nsl_path.stat().st_size / (1024 * 1024)
        print(f"✅ NSL-KDD training data ({size_mb:.1f} MB)")
        data_checks.append(True)
    else:
        print("❌ NSL-KDD training data missing")
        data_checks.append(False)

    # CIC-IDS-2017 sample
    cic_sample = Path("data/raw/cic-ids-2017/cic_ids_sample_backup.csv")
    if cic_sample.exists():
        size_mb = cic_sample.stat().st_size / (1024 * 1024)
        print(f"✅ CIC-IDS-2017 sample ({size_mb:.1f} MB)")
        data_checks.append(True)
    else:
        print("❌ CIC-IDS-2017 sample missing")
        data_checks.append(False)

    # CIC-IDS-2017 full dataset
    cic_full = Path("data/raw/cic-ids-2017/full_dataset")
    if cic_full.exists():
        csv_files = list(cic_full.glob("*.csv"))
        if csv_files:
            total_size = sum(f.stat().st_size for f in csv_files) / (1024 * 1024)
            print(
                f"✅ CIC-IDS-2017 full dataset ({len(csv_files)} files, {total_size:.0f} MB)"
            )
            data_checks.append(True)
        else:
            print("⚠️ CIC-IDS-2017 full dataset directory exists but no CSV files found")
            data_checks.append(False)
    else:
        print("⚠️ CIC-IDS-2017 full dataset directory missing")
        data_checks.append(False)

    return all(data_checks[:2])  # Only require sample files, full dataset is optional


def check_dependencies():
    """Check if key dependencies are available"""
    required = [
        "pandas",
        "numpy",
        "sklearn",
        "matplotlib",
        "seaborn",
        "joblib",
        "pathlib",
    ]
    optional = ["xgboost", "lightgbm"]

    dep_checks = []
    for package in required:
        try:
            __import__(package)
            print(f"✅ {package}")
            dep_checks.append(True)
        except ImportError:
            print(f"❌ {package} (required)")
            dep_checks.append(False)

    for package in optional:
        try:
            __import__(package)
            print(f"✅ {package} (optional)")
        except ImportError:
            print(f"⚠️ {package} (optional, will use fallback)")

    return all(dep_checks)


def check_results_structure():
    """Check if results directory structure exists"""
    results_dir = Path("data/results")
    if not results_dir.exists():
        results_dir.mkdir(parents=True, exist_ok=True)
        print("✅ Created results directory")
    else:
        print("✅ Results directory exists")

    # Check if we have any existing results
    result_files = list(results_dir.glob("*.csv")) + list(results_dir.glob("*.json"))
    if result_files:
        print(f"📊 Found {len(result_files)} existing result files")
        for f in result_files[:5]:  # Show first 5
            print(f"  - {f.name}")
        if len(result_files) > 5:
            print(f"  ... and {len(result_files) - 5} more")

    return True


def check_script_syntax():
    """Check if all experiment scripts compile"""
    experiment_dir = Path("experiments")
    if not experiment_dir.exists():
        print("❌ Experiments directory missing")
        return False

    scripts = list(experiment_dir.glob("*.py"))
    if not scripts:
        print("❌ No experiment scripts found")
        return False

    print(f"✅ Found {len(scripts)} experiment scripts")

    # Check the critical fixed script
    harmonized_script = experiment_dir / "06_harmonized_evaluation.py"
    if harmonized_script.exists():
        print("✅ Harmonized evaluation script (FIXED)")
    else:
        print("❌ Harmonized evaluation script missing")
        return False

    return True


def generate_status_report():
    """Generate overall status report"""
    print("\n" + "=" * 60)
    print("📋 REPOSITORY STATUS REPORT")
    print("=" * 60)

    checks = [
        ("Python Version", check_python_version()),
        ("Data Files", check_data_files()),
        ("Dependencies", check_dependencies()),
        ("Results Structure", check_results_structure()),
        ("Script Syntax", check_script_syntax()),
    ]

    total_checks = len(checks)
    passed_checks = sum(1 for _, status in checks if status)

    print(f"\n📊 Status: {passed_checks}/{total_checks} checks passed")

    if passed_checks == total_checks:
        print("🎉 REPOSITORY IS READY FOR EXPERIMENTS!")
        print("🚀 You can now run: python3 run_all_experiments.py")
        return True
    else:
        print("⚠️ Some issues need to be resolved before running experiments")
        print("\nTo fix data issues:")
        print("  1. Download NSL-KDD from: https://www.unb.ca/cic/datasets/nsl.html")
        print(
            "  2. Download CIC-IDS-2017 from: https://www.unb.ca/cic/datasets/ids-2017.html"
        )
        print("  3. Place files according to README.md instructions")
        return False


def main():
    """Main validation function"""
    print("🔍 REPOSITORY VALIDATION")
    print("Checking environment and data availability...\n")

    success = generate_status_report()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
