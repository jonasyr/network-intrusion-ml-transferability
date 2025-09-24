#!/usr/bin/env python3
"""
Safe experiment runner with error handling and resource management
"""

import sys
import os
import subprocess
import signal
from pathlib import Path

# Set safe environment variables
os.environ.update({
    'OMP_NUM_THREADS': '2',
    'OPENBLAS_NUM_THREADS': '1', 
    'MKL_NUM_THREADS': '1',
    'NUMEXPR_NUM_THREADS': '1',
    'XGBOOST_NTHREAD': '1'
})

def run_experiment_safely(script_path, max_retries=2):
    """Run an experiment script with error handling"""
    print(f"🚀 Running {script_path}")
    
    for attempt in range(max_retries + 1):
        if attempt > 0:
            print(f"🔄 Retry attempt {attempt}/{max_retries}")
            
        try:
            result = subprocess.run([
                sys.executable, script_path
            ], timeout=7200, capture_output=True, text=True)  # 2 hour timeout
            
            if result.returncode == 0:
                print(f"✅ {script_path} completed successfully")
                return True
            else:
                print(f"❌ {script_path} failed with code {result.returncode}")
                print(f"Error: {result.stderr[:500]}")
                
        except subprocess.TimeoutExpired:
            print(f"⏰ {script_path} timed out after 2 hours")
        except Exception as e:
            print(f"💥 {script_path} crashed: {e}")
            
    print(f"🚫 {script_path} failed after {max_retries + 1} attempts")
    return False

def main():
    experiments = [
        "experiments/01_data_exploration.py",
        "experiments/02_baseline_training.py", 
        "experiments/03_advanced_training.py",
        "experiments/04_cross_validation.py",
        "experiments/05_cross_dataset_evaluation.py",
        "experiments/06_harmonized_evaluation.py", 
        "experiments/07_generate_results_summary.py",
        "experiments/08_generate_paper_figures.py",
        "experiments/09_enhance_repository.py"
    ]
    
    successful = 0
    failed = []
    
    for exp in experiments:
        if Path(exp).exists():
            if run_experiment_safely(exp):
                successful += 1
            else:
                failed.append(exp)
        else:
            print(f"⚠️ Experiment not found: {exp}")
            failed.append(exp)
    
    print(f"\n📊 EXPERIMENT SUMMARY")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {len(failed)}")
    
    if failed:
        print("Failed experiments:")
        for f in failed:
            print(f"  - {f}")
    
    return len(failed) == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
