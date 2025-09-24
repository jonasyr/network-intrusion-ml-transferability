#!/usr/bin/env python3
"""
Enable Scientific Mode for Full Dataset Training
This script sets the environment variable to force full dataset usage across all experiments.
"""

import os
import subprocess
import sys

def enable_scientific_mode():
    """Enable scientific mode for all experiments."""
    print("🔬 ENABLING SCIENTIFIC MODE")
    print("=" * 50)
    print("This will:")
    print("✅ Force full dataset usage on ALL experiments")
    print("✅ Train ALL models including memory-intensive SVM and KNN")
    print("✅ Ensure scientifically accurate results for publication")
    print("⚠️  WARNING: This will use maximum available memory!")
    
    # Set environment variable
    os.environ["SCIENTIFIC_MODE"] = "1"
    os.environ["FORCE_FULL_DATASET"] = "1"
    
    print("\n🎯 Environment configured:")
    print(f"   SCIENTIFIC_MODE = {os.environ.get('SCIENTIFIC_MODE', 'Not set')}")
    print(f"   FORCE_FULL_DATASET = {os.environ.get('FORCE_FULL_DATASET', 'Not set')}")
    
    # Check system memory
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024**3)
        available_gb = psutil.virtual_memory().available / (1024**3)
        print(f"\n💾 System Memory:")
        print(f"   Total: {memory_gb:.1f}GB")
        print(f"   Available: {available_gb:.1f}GB")
        
        if memory_gb < 16:
            print("\n⚠️  WARNING: System has <16GB RAM")
            print("   Training all models may cause memory issues")
            response = input("Continue anyway? [y/N]: ").lower().strip()
            if response not in ('y', 'yes'):
                print("❌ Scientific mode cancelled")
                return False
    except ImportError:
        print("⚠️  Could not check system memory")
    
    print("\n✅ Scientific mode enabled!")
    print("🚀 All experiments will now use full datasets and all models")
    return True

def run_all_experiments():
    """Run the complete experimental pipeline with scientific mode enabled."""
    print("\n🚀 Starting complete experimental pipeline...")
    
    # Set environment variables for the subprocess
    env = os.environ.copy()
    env["SCIENTIFIC_MODE"] = "1"
    env["FORCE_FULL_DATASET"] = "1"
    
    try:
        # Run the experiments
        result = subprocess.run([
            sys.executable, 
            "run_all_experiments.py"
        ], env=env, check=True)
        
        print("\n🎉 All experiments completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Experiments failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return False

def main():
    """Main function."""
    if enable_scientific_mode():
        print("\n" + "=" * 50)
        choice = input("Run all experiments now? [Y/n]: ").lower().strip()
        if choice in ('', 'y', 'yes'):
            success = run_all_experiments()
            sys.exit(0 if success else 1)
        else:
            print("✅ Scientific mode is enabled.")
            print("💡 Run experiments manually with: python run_all_experiments.py")
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()