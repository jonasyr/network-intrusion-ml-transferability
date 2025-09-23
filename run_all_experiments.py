#!/usr/bin/env python3
"""
Run All Experiments Pipeline
Complete experimental pipeline for the research paper
"""

import subprocess
import sys
import time
import threading
import os
import select
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

def parse_experiment_output(output_line):
    """Parse experiment output to extract meaningful progress information"""
    line = output_line.strip()
    line_lower = line.lower()
    
    # Common progress indicators to look for
    progress_indicators = [
        ("loading", "📂 Loading data"),
        ("preprocessing", "⚙️ Preprocessing"),
        ("training", "🏋️ Training model"),
        ("fitting", "🏋️ Fitting model"),
        ("evaluating", "📊 Evaluating"),
        ("testing", "🧪 Testing"),
        ("validating", "✅ Validating"),
        ("cross-validation", "🔄 Cross-validating"),
        ("feature", "🎯 Feature processing"),
        ("model selection", "🎯 Model selection"),
        ("hyperparameter", "⚙️ Tuning parameters"),
        ("saving", "💾 Saving results"),
        ("generating", "📋 Generating output"),
        ("computing", "🧮 Computing metrics"),
        ("analyzing", "🔍 Analyzing"),
        ("baseline", "📐 Baseline processing"),
        ("advanced", "🚀 Advanced processing"),
        ("random forest", "🌲 Random Forest"),
        ("gradient boosting", "⚡ Gradient Boosting"),
        ("neural network", "🧠 Neural Network"),
        ("svm", "🎯 Support Vector Machine"),
        ("logistic regression", "📈 Logistic Regression"),
        ("accuracy", "🎯 Accuracy"),
        ("precision", "🔍 Precision"),
        ("recall", "📡 Recall"),
        ("f1-score", "⚖️ F1-Score"),
        ("auc", "📊 AUC Score"),
        ("confusion matrix", "🔢 Confusion Matrix"),
    ]
    
    for keyword, icon in progress_indicators:
        if keyword in line_lower:
            # Try to preserve important parts of the original line
            if any(metric in line_lower for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']):
                # For metrics, show the actual values if present
                return f"{icon} {line[:80]}"
            else:
                return f"{icon} {line_lower.capitalize()[:60]}"
    
    # Check for percentage or numerical progress
    if any(indicator in line_lower for indicator in ["epoch", "fold", "iteration", "step"]):
        return f"📈 {line[:70]}"
    
    # Check for completion indicators
    completion_indicators = ["completed", "finished", "done", "saved", "wrote"]
    for indicator in completion_indicators:
        if indicator in line_lower:
            return f"✨ {line[:70]}"
    
    # Check for dataset information
    if any(keyword in line_lower for keyword in ["dataset", "samples", "features", "classes"]):
        return f"📋 {line[:70]}"
    
    # Check for model performance outputs
    if any(keyword in line_lower for keyword in ["best", "score", "performance", "results"]):
        return f"🏆 {line[:70]}"
    
    return None

def parse_experiment_output(line):
    """Parse experiment output for activity detection"""
    if not line:
        return None
    
    line = line.strip()
    
    # Skip empty lines, debug info, and warnings
    if (not line or 
        "DEBUG" in line.upper() or 
        "WARNING" in line.upper() or 
        "FutureWarning" in line or 
        "DeprecationWarning" in line or
        line.startswith("2024-") or  # Skip timestamp-only lines
        len(line) < 5):
        return None
    
    # Look for meaningful activity indicators
    activity_keywords = [
        ("Loading", ["loading", "reading", "importing", "load", "read"]),
        ("Preprocessing", ["preprocessing", "preprocess", "normalizing", "scaling", "encoding"]),
        ("Feature", ["feature", "selection", "extraction", "engineering"]),
        ("Training", ["training", "fit", "fitting", "epoch", "learning"]),
        ("Evaluating", ["evaluating", "evaluate", "testing", "validation", "score", "accuracy"]),
        ("Cross-validating", ["cross", "cv", "fold"]),
        ("Saving", ["saving", "save", "export", "writing"]),
        ("Generating", ["generating", "generate", "creating", "create"])
    ]
    
    lower_line = line.lower()
    for activity, keywords in activity_keywords:
        if any(keyword in lower_line for keyword in keywords):
            return f"{activity}: {line[:50]}{'...' if len(line) > 50 else ''}"
    
    return None


def format_duration(seconds):
    """Format duration in human-readable format"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours}:{minutes:02d}:{secs:02d}"


def run_experiment(script_path, description, step_num, total_steps, experiment_durations=None):
    """Run a single experiment script with real-time progress tracking"""
    print(f"\n{'='*80}")
    print(f"🧪 STEP {step_num}/{total_steps}: {description}")
    print(f"{'='*80}")
    print(f"📜 Script: {script_path}")
    print(f"⏰ Started at: {datetime.now().strftime('%H:%M:%S')}")
    
    start_time = time.time()
    python_exe = get_python_executable()
    print(f"🐍 Python: {python_exe}")
    print()  # Add space before progress indicator
    
    # Shared variables for threading
    output_lines = []
    try:
        # Use Popen for real-time output streaming
        process = subprocess.Popen(
            [python_exe, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Real-time output processing with periodic progress updates
        print("🔄 Live Output (filtering DEBUG messages):")
        print("-" * 80)
        
        # Show initial progress
        progress_percent = int((step_num - 1) / total_steps * 100)
        print(f"📊 [{step_num}/{total_steps}] {progress_percent}% | Starting | ⏱️ 0.0s")
        
        last_progress_time = time.time()
        line_count = 0
        
        while True:
            # Check if process has finished
            if process.poll() is not None:
                # Read any remaining output
                remaining = process.stdout.read()
                if remaining:
                    for line in remaining.split('\n'):
                        if line.strip():
                            output_lines.append(line + '\n')
                            clean_line = line.strip()
                            if clean_line and not any(keyword in clean_line for keyword in ['debug', 'DEBUG', 'TRACE', 'trace', 'UserWarning:', 'FutureWarning:', 'DeprecationWarning:', '/opt/conda/', '/usr/local/lib/', 'site-packages/', '__pycache__', '.pyc', 'pytest-warning']):
                                timestamp = datetime.now().strftime('%H:%M:%S')
                                print(f"[{timestamp}] {clean_line}")
                break
            
            # Check for progress update regardless of whether there's new output
            current_time = time.time()
            if current_time - last_progress_time > 60.0:
                elapsed = current_time - start_time
                progress_percent = int((step_num - 1) / total_steps * 100)
                
                # Analyze recent output for current phase
                current_phase = "Processing"
                if output_lines:
                    recent_lines = output_lines[-5:]
                    for line in reversed(recent_lines):
                        activity = parse_experiment_output(line.strip())
                        if activity:
                            if "Loading" in activity:
                                current_phase = "Data Loading"
                            elif "Preprocessing" in activity or "Feature" in activity:
                                current_phase = "Data Processing" 
                            elif "Training" in activity or "Fitting" in activity:
                                current_phase = "Model Training"
                            elif "Evaluating" in activity or "Testing" in activity:
                                current_phase = "Model Evaluation"
                            elif "Cross-validating" in activity:
                                current_phase = "Cross-Validation"
                            elif "Saving" in activity or "Generating" in activity:
                                current_phase = "Results Generation"
                            break
                
                # Calculate ETA
                eta_str = ""
                if len(experiment_durations or []) > 0 and step_num > 1:
                    completed_durations = [d[1] for d in experiment_durations if d[2]]
                    if completed_durations:
                        avg_duration = sum(completed_durations) / len(completed_durations)
                        remaining_experiments = total_steps - step_num
                        estimated_remaining = avg_duration * remaining_experiments
                        if elapsed < avg_duration:
                            estimated_remaining += (avg_duration - elapsed)
                        eta_time = datetime.now() + timedelta(seconds=estimated_remaining)
                        eta_str = f" | ETA: {eta_time.strftime('%H:%M')}"
                
                print(f"📊 [{step_num}/{total_steps}] {progress_percent}% | {current_phase} | ⏱️ {format_duration(elapsed)}{eta_str}")
                last_progress_time = current_time
            
            # Use select to check if there's data to read (non-blocking)
            if select.select([process.stdout], [], [], 0.1)[0]:  # 0.1 second timeout
                output = process.stdout.readline()
                if output:
                    output_lines.append(output)
                    line_count += 1
                    
                    # Filter out debug messages and other noise
                    clean_line = output.strip()
                    should_show = True
                    
                    # Filter criteria
                    filter_keywords = [
                        'debug', 'DEBUG', 'TRACE', 'trace',
                        'UserWarning:', 'FutureWarning:', 'DeprecationWarning:',
                        '/opt/conda/', '/usr/local/lib/', 'site-packages/',
                        '__pycache__', '.pyc', 'pytest-warning'
                    ]
                    
                    # Skip empty lines and filtered content
                    if not clean_line:
                        should_show = False
                    else:
                        for keyword in filter_keywords:
                            if keyword in clean_line:
                                should_show = False
                                break
                    
                    # Show relevant output in real-time
                    if should_show:
                        timestamp = datetime.now().strftime('%H:%M:%S')
                        print(f"[{timestamp}] {clean_line}")
            else:
                # Small sleep to prevent busy waiting when no data is available
                time.sleep(0.05)
        
        # Show final progress update
        final_elapsed = time.time() - start_time
        progress_percent = int((step_num) / total_steps * 100)
        print(f"📊 [{step_num}/{total_steps}] {progress_percent}% | Complete | ⏱️ {format_duration(final_elapsed)}")
        
        print("-" * 80)
        print("🔄 Experiment output complete, processing results...")
        print()
        
        # Real-time output processing
        current_phase = "Initializing"
        line_count = 0
        
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
                
            if output:
                output_lines.append(output)
                line_count += 1
                current_time = time.time()
                
                # Parse for meaningful progress information
                progress_info = parse_experiment_output(output.strip())
                
                # Update current phase based on output
                if progress_info:
                    if "Loading" in progress_info:
                        current_phase = "Data Loading"
                    elif "Preprocessing" in progress_info or "Feature" in progress_info:
                        current_phase = "Data Processing"
                    elif "Training" in progress_info or "Fitting" in progress_info:
                        current_phase = "Model Training"
                    elif "Evaluating" in progress_info or "Testing" in progress_info:
                        current_phase = "Model Evaluation"
                    elif "Cross-validating" in progress_info:
                        current_phase = "Cross-Validation"
                    elif "Saving" in progress_info or "Generating" in progress_info:
                        current_phase = "Results Generation"
                
                # Show progress updates every 1.5 seconds or for meaningful output
                if progress_info or (current_time - last_progress_update > 1.5):
                    elapsed = current_time - start_time
                    
                    if progress_info:
                        # Clear the line and show progress with phase
                        print(f"\r⏱️ {format_duration(elapsed):>8} | 📍 {current_phase:<18} | {progress_info:<60}", end="", flush=True)
                        last_progress_update = current_time
                    else:
                        # Show generic progress with current phase and line count
                        if line_count % 5 == 0:  # Update every 5 lines to reduce flickering
                            print(f"\r⏱️ {format_duration(elapsed):>8} | � {current_phase:<18} | 🔄 Processing... ({line_count} lines)", end="", flush=True)
                            last_progress_update = current_time
        
        # Wait for process to complete
        return_code = process.poll()
        duration = time.time() - start_time
        end_time = datetime.now()
        
        if return_code == 0:
            print(f"✅ EXPERIMENT COMPLETED in {format_duration(duration)}")
            print(f"⏰ Finished at: {end_time.strftime('%H:%M:%S')}")
            print(f"� Next: Starting experiment {step_num + 1}/{total_steps}" if step_num < total_steps else "🎯 All experiments complete!")
            return True, duration
        else:
            print(f"❌ EXPERIMENT FAILED after {format_duration(duration)} (exit code {return_code})")
            print(f"⏰ Failed at: {end_time.strftime('%H:%M:%S')}")
            return False, duration
            
    except Exception as e:
        duration = time.time() - start_time
        print(f"❌ EXPERIMENT EXCEPTION after {format_duration(duration)}: {str(e)}")
        return False, duration

def main():
    """Run complete experimental pipeline"""
    print("🚀 COMPLETE EXPERIMENTAL PIPELINE")
    print("🎓 Machine Learning Models for Network Anomaly Detection")
    print("=" * 80)
    print(f"⏰ Pipeline started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Import memory utilities to check configuration
    try:
        import sys
        from pathlib import Path
        PROJECT_ROOT = Path(__file__).parent
        sys.path.append(str(PROJECT_ROOT / "src"))
        from utils.memory_utils import get_memory_adaptive_config, get_system_memory_gb
        
        # Show memory configuration and scientific accuracy warnings
        print(f"\n🔬 SCIENTIFIC ACCURACY CHECK")
        print("=" * 50)
        
        system_memory = get_system_memory_gb()
        config = get_memory_adaptive_config()
        
        if config["use_full_dataset"]:
            print("✅ FULL DATASET MODE: All experiments will use complete datasets")
            print("✅ Results will be scientifically accurate for publication")
        else:
            print("⚠️  REDUCED DATASET MODE: Some experiments will use sampled data")
            print("⚠️  Results may not be suitable for publication without full dataset")
            print("\n💡 To ensure scientific accuracy:")
            print("   1. Run on a system with >16GB RAM, OR")
            print("   2. Set environment variable: export SCIENTIFIC_MODE=1")
            
            response = input("\n🤔 Continue anyway? [y/N]: ").lower().strip()
            if response not in ('y', 'yes'):
                print("🛑 Pipeline cancelled. Please ensure sufficient memory for full dataset.")
                return False
    except ImportError:
        print("⚠️ Could not check memory configuration. Proceeding...")
    
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
    print(f"   Estimated duration: 2-6 hours (depending on dataset size)")
    try:
        if config["use_full_dataset"]:
            print(f"   📊 Dataset mode: FULL (scientifically accurate)")
        else:
            print(f"   📊 Dataset mode: OPTIMIZED (may reduce accuracy)")
    except:
        print(f"   📊 Dataset mode: Unknown")
    
    for i, (script_path, description) in enumerate(experiments, 1):
        if Path(script_path).exists():
            print(f"\n🔄 Starting experiment {i}/{len(experiments)}...")
            success, duration = run_experiment(script_path, description, i, len(experiments), experiment_durations)
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
    
    # Final scientific accuracy summary
    try:
        if config["use_full_dataset"]:
            scientific_status = "✅ SCIENTIFICALLY ACCURATE"
            accuracy_note = "All experiments used complete datasets - results are publication-ready"
        else:
            scientific_status = "⚠️  POTENTIALLY INACCURATE"
            accuracy_note = "Some experiments used reduced datasets - verify results before publication"
        
        print(f"\n🔬 SCIENTIFIC ACCURACY STATUS:")
        print(f"{'='*80}")
        print(f"{scientific_status}")
        print(f"📄 {accuracy_note}")
    except:
        print(f"\n🔬 SCIENTIFIC ACCURACY: Unable to determine dataset completeness")
    
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