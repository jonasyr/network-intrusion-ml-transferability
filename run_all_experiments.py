#!/usr/bin/env python3
"""
Run All Experiments Pipeline
Complete experimental pipeline for the research paper
"""

import subprocess
import sys
import time
from pathlib import Path

def get_python_executable():
    """Get the correct Python executable (virtual environment if available)"""
    venv_python = Path(".venv/bin/python")
    if venv_python.exists():
        return str(venv_python.absolute())
    return sys.executable

def run_experiment(script_path, description):
    """Run a single experiment script"""
    print(f"\n{'='*60}")
    print(f"🧪 {description}")
    print(f"{'='*60}")
    print(f"📜 Running: {script_path}")
    
    start_time = time.time()
    python_exe = get_python_executable()
    print(f"🐍 Using Python: {python_exe}")
    
    try:
        result = subprocess.run([python_exe, script_path], 
                              capture_output=True, text=True, check=True)
        duration = time.time() - start_time
        
        print(f"✅ Completed in {duration:.1f}s")
        if result.stdout:
            print("📤 Output:")
            print(result.stdout[-500:])  # Last 500 chars
        return True
        
    except subprocess.CalledProcessError as e:
        duration = time.time() - start_time
        print(f"❌ Failed after {duration:.1f}s")
        print(f"Error: {e}")
        if e.stderr:
            print("Error output:")
            print(e.stderr[-500:])
        return False

def main():
    """Run complete experimental pipeline"""
    print("🚀 COMPLETE EXPERIMENTAL PIPELINE")
    print("🎓 Machine Learning Models for Network Anomaly Detection")
    print("=" * 80)
    
    # Define experiment sequence
    experiments = [
        ("experiments/01_data_exploration.py", "Dataset Exploration"),
        ("experiments/02_baseline_training.py", "Baseline Models Training"),
        ("experiments/03_advanced_training.py", "Advanced Models Training"),
        ("experiments/04_cross_validation.py", "Cross-Validation Analysis"),
        ("experiments/05_cross_dataset_evaluation.py", "Cross-Dataset Evaluation Pipeline"),
        ("experiments/06_generate_figures.py", "Generate Publication Figures"),
    ]
    
    total_start_time = time.time()
    successful = 0
    failed = 0
    
    for script_path, description in experiments:
        if Path(script_path).exists():
            success = run_experiment(script_path, description)
            if success:
                successful += 1
            else:
                failed += 1
        else:
            print(f"⚠️ Script not found: {script_path}")
            failed += 1
    
    total_duration = time.time() - total_start_time
    
    print(f"\n{'='*80}")
    print(f"🎯 PIPELINE COMPLETE")
    print(f"{'='*80}")
    print(f"⏱️ Total Duration: {total_duration/60:.1f} minutes")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Success Rate: {successful/(successful+failed)*100:.1f}%")
    
    if failed == 0:
        print("\n🎉 ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
        print("📝 Ready for paper writing!")
    else:
        print(f"\n⚠️ {failed} experiments failed. Check logs above.")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)