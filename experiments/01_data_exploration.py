#!/usr/bin/env python3
"""Initial exploratory analysis for NSL-KDD and CIC-IDS-2017 datasets."""

from __future__ import annotations

import sys
from pathlib import Path

current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from src.preprocessing import CICIDSPreprocessor, NSLKDDAnalyzer


def run_data_exploration() -> bool:
    print("🧭 DATA EXPLORATION")
    print("=" * 60)

    analyzer = NSLKDDAnalyzer()
    nsl_data = analyzer.load_data("KDDTrain+_20Percent.txt")

    if nsl_data is None:
        print("❌ Failed to load NSL-KDD sample")
        return False

    analyzer.basic_info(nsl_data)
    analyzer.analyze_class_distribution(nsl_data)

    cic_preprocessor = CICIDSPreprocessor()
    cic_sample = cic_preprocessor.load_data(use_full_dataset=False)

    if cic_sample is None:
        print("❌ Failed to load CIC-IDS-2017 sample")
        return False

    print("\n📊 CIC-IDS-2017 Sample Overview")
    print(cic_sample.head())
    print(cic_sample["Label"].value_counts())

    return True


def main() -> bool:
    success = run_data_exploration()
    if success:
        print("\n🎯 Data exploration complete!")
    return success


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
