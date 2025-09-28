#!/usr/bin/env python3
# scripts/run_cross_validation.py
"""
Run comprehensive cross-validation analysis on all trained models
"""

import sys
import os
from pathlib import Path

# Add project root to path
current_dir = Path(__file__).parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))


def main():
    """Run cross-validation pipeline"""
    print("🚀 COMPREHENSIVE CROSS-VALIDATION ANALYSIS")
    print("=" * 60)

    overall_success = True

    try:
        # Import the cross-validation modules
        from src.metrics.cross_validation import (
            run_full_cross_validation,
            run_cic_cross_validation,
        )

        # Run NSL-KDD cross-validation
        print("\n🔬 NSL-KDD Cross-Validation")
        print("-" * 40)
        cv_framework, nsl_results = run_full_cross_validation()

        if nsl_results:
            print(f"\n🎯 NSL-KDD CROSS-VALIDATION COMPLETE!")
            print(f"✅ Analyzed {len(nsl_results)} NSL-KDD models")

            # Show top 3 NSL models
            print(f"\n🏆 TOP 3 NSL-KDD MODELS BY ACCURACY:")
            sorted_results = sorted(
                nsl_results, key=lambda x: x["accuracy_mean"], reverse=True
            )
            for i, result in enumerate(sorted_results[:3], 1):
                print(
                    f"  {i}. {result['model_name']}: {result['accuracy_mean']:.4f} ± {result['accuracy_std']:.4f}"
                )
        else:
            print("\n❌ No NSL-KDD cross-validation results generated")
            overall_success = False

        # Run CIC-IDS-2017 cross-validation
        print("\n� CIC-IDS-2017 Cross-Validation")
        print("-" * 40)
        cic_cv_framework, cic_results = run_cic_cross_validation()

        if cic_results:
            print(f"\n🎯 CIC-IDS-2017 CROSS-VALIDATION COMPLETE!")
            print(f"✅ Analyzed {len(cic_results)} CIC-IDS-2017 models")

            # Show top 3 CIC models
            print(f"\n🏆 TOP 3 CIC-IDS-2017 MODELS BY ACCURACY:")
            sorted_cic_results = sorted(
                cic_results, key=lambda x: x["accuracy_mean"], reverse=True
            )
            for i, result in enumerate(sorted_cic_results[:3], 1):
                print(
                    f"  {i}. {result['model_name']}: {result['accuracy_mean']:.4f} ± {result['accuracy_std']:.4f}"
                )
        else:
            print("\n⚠️ CIC-IDS-2017 cross-validation skipped (models not found)")

        return overall_success

    except Exception as e:
        print(f"❌ Cross-validation failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
