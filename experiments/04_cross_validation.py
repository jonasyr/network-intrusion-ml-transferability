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
    
    try:
        # Import the cross-validation module
        from src.metrics import run_full_cross_validation
        
        # Run the analysis
        cv_framework, results = run_full_cross_validation()
        
        if results:
            print(f"\n🎯 CROSS-VALIDATION COMPLETE!")
            print(f"✅ Analyzed {len(results)} models")
            print(f"📊 Results saved to data/results/cross_validation/")
            print(f"📈 Visualizations saved to data/results/")
            
            # Show top 3 models
            print(f"\n🏆 TOP 3 MODELS BY ACCURACY:")
            sorted_results = sorted(results, key=lambda x: x['accuracy_mean'], reverse=True)
            for i, result in enumerate(sorted_results[:3], 1):
                print(f"  {i}. {result['model_name']}: {result['accuracy_mean']:.4f} ± {result['accuracy_std']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Cross-validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)