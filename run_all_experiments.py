#!/usr/bin/env python3
"""
Run All Experiments Pipeline
Complete experimental pipeline for the research paper
"""

import subprocess
import sys
import time
import threading
from pathlib import Path
from datetime import datetime, timedelta

def get_python_executable():
    """Get the correct Python executable (virtual environment if available)"""
    venv_python = Path(".venv/bin/python")
    if venv_python.exists():
        return str(venv_python.absolute())
    return sys.executable

def format_duration(seconds):
    """Format duration in a human-readable way"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}min"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}min"

def progress_indicator(stop_event, experiment_name, start_time):
    """Show a progress indicator with elapsed time"""
    chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    idx = 0
    
    while not stop_event.is_set():
        elapsed = time.time() - start_time
        print(f"\r{chars[idx]} {experiment_name} - Elapsed: {format_duration(elapsed)}", end="", flush=True)
        idx = (idx + 1) % len(chars)
        time.sleep(0.2)

def run_experiment(script_path, description, step_num, total_steps):
    """Run a single experiment script with enhanced progress tracking"""
    print(f"\n{'='*80}")
    print(f"🧪 STEP {step_num}/{total_steps}: {description}")
    print(f"{'='*80}")
    print(f"📜 Script: {script_path}")
    print(f"⏰ Started at: {datetime.now().strftime('%H:%M:%S')}")
    
    start_time = time.time()
    python_exe = get_python_executable()
    print(f"🐍 Python: {python_exe}")
    
    # Start progress indicator in a separate thread
    stop_event = threading.Event()
    progress_thread = threading.Thread(
        target=progress_indicator, 
        args=(stop_event, description, start_time)
    )
    progress_thread.daemon = True
    progress_thread.start()
    
    try:
        result = subprocess.run([python_exe, script_path], 
                              capture_output=True, text=True, check=True)
        
        # Stop progress indicator
        stop_event.set()
        progress_thread.join(timeout=1)
        
        duration = time.time() - start_time
        end_time = datetime.now()
        
        print(f"\r✅ COMPLETED in {format_duration(duration)}")
        print(f"⏰ Finished at: {end_time.strftime('%H:%M:%S')}")
        
        if result.stdout:
            print("📤 Last 500 characters of output:")
            print("-" * 50)
            print(result.stdout[-500:])
            print("-" * 50)
        
        return True, duration
        
    except subprocess.CalledProcessError as e:
        # Stop progress indicator
        stop_event.set()
        progress_thread.join(timeout=1)
        
        duration = time.time() - start_time
        end_time = datetime.now()
        
        print(f"\r❌ FAILED after {format_duration(duration)}")
        print(f"⏰ Failed at: {end_time.strftime('%H:%M:%S')}")
        print(f"💥 Error code: {e.returncode}")
        
        if e.stderr:
            print("🚨 Error output:")
            print("-" * 50)
            print(e.stderr[-1000:])  # Show more error context
            print("-" * 50)
        
        if e.stdout:
            print("📤 Standard output:")
            print("-" * 50)
            print(e.stdout[-500:])
            print("-" * 50)
            
        return False, duration

def main():
    """Run complete experimental pipeline"""
    print("🚀 COMPLETE EXPERIMENTAL PIPELINE")
    print("🎓 Machine Learning Models for Network Anomaly Detection")
    print("=" * 80)
    print(f"⏰ Pipeline started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Define complete experiment sequence (ALL experiments)
    experiments = [
        ("experiments/01_data_exploration.py", "Dataset Exploration & Analysis"),
        ("experiments/02_baseline_training.py", "Baseline Models Training"),
        ("experiments/03_advanced_training.py", "Advanced Models Training"),
        ("experiments/04_cross_validation.py", "Cross-Validation Analysis"),
        ("experiments/05_cross_dataset_evaluation.py", "Cross-Dataset Evaluation Pipeline"),
        ("experiments/06_harmonized_evaluation.py", "Harmonized Feature Evaluation"),
        ("experiments/07_generate_results_summary.py", "Aggregate Experiment Summaries"),
        ("experiments/08_generate_paper_figures.py", "Generate Publication Figures"),
        ("experiments/09_enhance_repository.py", "Repository Enhancement & Scientific Value"),
        ("experiments/10_validate_enhancements.py", "Validate All Enhancements"),
    ]
    
    total_start_time = time.time()
    successful = 0
    failed = 0
    experiment_durations = []
    
    print(f"\n📋 EXPERIMENT OVERVIEW:")
    print(f"   Total experiments to run: {len(experiments)}")
    print(f"   Estimated duration: 2-4 hours (depending on dataset size)")
    
    for i, (script_path, description) in enumerate(experiments, 1):
        if Path(script_path).exists():
            print(f"\n🔄 Starting experiment {i}/{len(experiments)}...")
            success, duration = run_experiment(script_path, description, i, len(experiments))
            experiment_durations.append((description, duration, success))
            
            if success:
                successful += 1
                
                # Estimate remaining time based on average duration so far
                if i > 1:  # Only estimate after first experiment
                    avg_duration = sum(d[1] for d in experiment_durations if d[2]) / successful
                    remaining_experiments = len(experiments) - i
                    estimated_remaining = avg_duration * remaining_experiments
                    
                    print(f"⏱️ Average experiment duration: {format_duration(avg_duration)}")
                    print(f"📊 Estimated remaining time: {format_duration(estimated_remaining)}")
                    
                    if estimated_remaining > 0:
                        eta = datetime.now() + timedelta(seconds=estimated_remaining)
                        print(f"🎯 Estimated completion: {eta.strftime('%H:%M:%S')}")
            else:
                failed += 1
                print(f"⚠️ Experiment {i} failed. Continuing with next experiment...")
        else:
            print(f"⚠️ Script not found: {script_path}")
            failed += 1
    
    total_duration = time.time() - total_start_time
    end_time = datetime.now()
    
    print(f"\n{'='*80}")
    print(f"🎯 PIPELINE COMPLETE")
    print(f"{'='*80}")
    print(f"⏰ Finished at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱️ Total Duration: {format_duration(total_duration)}")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Success Rate: {successful/(successful+failed)*100:.1f}%")
    
    # Detailed timing breakdown
    if experiment_durations:
        print(f"\n📈 EXPERIMENT TIMING BREAKDOWN:")
        print(f"{'='*80}")
        for desc, duration, success in experiment_durations:
            status = "✅" if success else "❌"
            print(f"{status} {desc:<45} {format_duration(duration):>10}")
    
    if failed == 0:
        print("\n🎉 ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
        print("📝 Ready for paper writing!")
        print("📊 Check data/results/ for all generated outputs")
        print("📄 Review enhanced_results_report.md for insights")
    else:
        print(f"\n⚠️ {failed} experiments failed. Check logs above.")
        print(f"💡 You can re-run individual failed experiments manually")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)