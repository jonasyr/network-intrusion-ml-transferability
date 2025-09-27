#!/usr/bin/env python3
"""
Incremental CIC Cross-Validation Runner
Run this script to perform cross-validation with automatic saving and resume capability.
Safe to interrupt and restart - it will resume from where it left off.
"""

import sys
from pathlib import Path

# Add project root to path
current_dir = Path(__file__).parent
project_root = current_dir
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

def main():
    """Run incremental CIC cross-validation with same output as experiment 04"""
    print("🚀 COMPREHENSIVE CROSS-VALIDATION ANALYSIS")
    print("=" * 60)
    print("")
    print("� CIC-IDS-2017 Cross-Validation")
    print("-" * 40)
    
    try:
        from src.metrics.cross_validation import run_cic_cross_validation
        
        # Run with same output format as experiment 04
        cic_cv_framework, cic_results = run_cic_cross_validation()
        
        if cic_results:
            print(f"\n� CIC-IDS-2017 CROSS-VALIDATION COMPLETE!")
            print(f"✅ Analyzed {len(cic_results)} CIC-IDS-2017 models")
            
            # Show top 3 CIC models (same format as experiment 04)
            print(f"\n🏆 TOP 3 CIC-IDS-2017 MODELS BY ACCURACY:")
            sorted_cic_results = sorted(cic_results, key=lambda x: x['accuracy_mean'], reverse=True)
            for i, result in enumerate(sorted_cic_results[:3], 1):
                print(f"  {i}. {result['model_name']}: {result['accuracy_mean']:.4f} ± {result['accuracy_std']:.4f}")
                
            return True
        else:
            print("\n⚠️ CIC-IDS-2017 cross-validation skipped (models not found)")
            return False
        
    except KeyboardInterrupt:
        print("\n⏸️  Cross-validation interrupted by user")
        print("💾 Progress has been saved - you can restart this script to resume")
        return True
        
    except Exception as e:
        print(f"\n❌ Error during cross-validation: {e}")
        import traceback
        traceback.print_exc()
        print("\n💾 Any completed models have been saved")
        print("🔄 You can restart this script to retry/resume")
        return False

if __name__ == "__main__":
    # Show system info quietly like experiment 04
    try:
        from src.utils import get_memory_adaptive_config
        config = get_memory_adaptive_config()
    except Exception as e:
        print(f"⚠️ Could not load configuration: {e}")
        sys.exit(1)
    
    success = main()
    sys.exit(0 if success else 1)