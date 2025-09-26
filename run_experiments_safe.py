#!/usr/bin/env python3
"""
Safe experiment runner with error handling and resource management
"""

import sys
import os
import subprocess
import signal
import time
from pathlib import Path
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# Set optimized environment variables for 12-core system
os.environ.update({
    'OMP_NUM_THREADS': '6',        # Use half cores for OpenMP (conservative)
    'OPENBLAS_NUM_THREADS': '4',   # BLAS operations can use 4 threads safely
    'MKL_NUM_THREADS': '4',        # Intel MKL can handle 4 threads well
    'NUMEXPR_NUM_THREADS': '4',    # NumExpr benefits from multiple threads
    'XGBOOST_NTHREAD': '4'         # XGBoost with 4 threads (was causing segfaults with 1)
})

def set_experiment_specific_limits(script_path):
    """Set experiment-specific resource limits"""
    script_name = Path(script_path).stem
    
    if 'advanced_training' in script_name or 'cross_validation' in script_name:
        # Memory-intensive experiments: use more conservative settings
        os.environ.update({
            'OMP_NUM_THREADS': '4',
            'XGBOOST_NTHREAD': '2'  # Extra conservative for XGBoost
        })
        print(f"🔧 Applied conservative limits for {script_name}")
    elif 'data_exploration' in script_name or 'generate_' in script_name:
        # Light experiments: can use more threads
        os.environ.update({
            'OMP_NUM_THREADS': '8',
            'XGBOOST_NTHREAD': '6'
        })
        print(f"🚀 Applied high-performance limits for {script_name}")
    else:
        # Default balanced settings (already set globally)
        print(f"⚖️ Using balanced limits for {script_name}")

def run_experiment_safely(script_path, max_retries=2):
    """Run an experiment script with error handling"""
    print(f"🚀 Running {script_path}")
    
    # Set experiment-specific resource limits
    set_experiment_specific_limits(script_path)
    
    for attempt in range(max_retries + 1):
        if attempt > 0:
            print(f"🔄 Retry attempt {attempt}/{max_retries}")
            
        try:
            # Run with live output - same as running experiments individually
            result = subprocess.run([
                sys.executable, script_path
            ])  # NO TIMEOUT - let it run as long as it needs!
            
            if result.returncode == 0:
                print(f"✅ {script_path} completed successfully")
                return True
            else:
                print(f"❌ {script_path} failed with code {result.returncode}")
                print("📋 Check the output above for error details")
                
        except KeyboardInterrupt:
            print(f"🛑 {script_path} interrupted by user - stopping pipeline")
            return False
        except Exception as e:
            print(f"💥 {script_path} crashed: {e}")
            
    print(f"🚫 {script_path} failed after {max_retries + 1} attempts")
    return False

def monitor_system_resources():
    """Monitor and report system resource usage"""
    if not HAS_PSUTIL:
        return "Resource monitoring unavailable (psutil not installed)"
    
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    load_avg = os.getloadavg() if hasattr(os, 'getloadavg') else (0, 0, 0)
    
    return (f"💻 System Status: CPU={cpu_percent:.1f}% | "
            f"Memory={memory.percent:.1f}% | "
            f"Load={load_avg[0]:.2f}")

def main():
    print("🖥️  SYSTEM INFO")
    print(f"Available CPU cores: {os.cpu_count()}")
    if HAS_PSUTIL:
        print(f"Available memory: {psutil.virtual_memory().total // (1024**3)} GB")
    print(f"Current thread limits: OMP={os.environ.get('OMP_NUM_THREADS')}, "
          f"XGBoost={os.environ.get('XGBOOST_NTHREAD')}")
    print(f"Resource monitoring: {monitor_system_resources()}")
    print()
    
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
    
    for i, exp in enumerate(experiments, 1):
        print(f"\n{'='*60}")
        print(f"📊 EXPERIMENT {i}/{len(experiments)}: {exp}")
        print(f"⏰ Starting at: {time.strftime('%H:%M:%S')}")
        print(monitor_system_resources())
        print('='*60)
        
        if Path(exp).exists():
            start_time = time.time()
            if run_experiment_safely(exp):
                duration = time.time() - start_time
                print(f"✅ Completed in {duration/60:.1f} minutes")
                print(monitor_system_resources())
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
