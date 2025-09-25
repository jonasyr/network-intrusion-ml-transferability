#!/usr/bin/env python3
"""
FAST CIC BASELINE RECOVERY - Skip the slow models, get results quickly
"""

import sys
from pathlib import Path

# Add project root to path  
PROJECT_ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from models.baseline_models import BaselineModels
from preprocessing.cic_ids_preprocessor import CICIDSPreprocessor
import numpy as np
from sklearn.model_selection import train_test_split
import time

def fast_cic_baseline_training():
    """Train CIC baseline models FAST - skip the slow ones"""
    
    print("⚡ FAST CIC BASELINE RECOVERY")
    print("=" * 50)
    print("Training only the FAST and EFFECTIVE models")
    print("Skipping: KNN (too slow), SVM (too slow)")
    
    start_time = time.time()
    
    try:
        # Load CIC data efficiently
        preprocessor = CICIDSPreprocessor()
        print("\n📁 Loading CIC-IDS-2017 data...")
        cic_data = preprocessor.load_data(use_full_dataset=True)
        
        if cic_data is None:
            print("❌ Failed to load CIC data")
            return False
            
        print(f"✅ Loaded CIC data: {cic_data.shape}")
        
        # Preprocess
        print("🔄 Preprocessing...")
        x_full, y_full = preprocessor.fit_transform(cic_data)
        print(f"✅ Preprocessed data: {x_full.shape}")
        
        # Create splits
        print("🔄 Creating train/test splits...")
        x_train, x_temp, y_train, y_temp = train_test_split(
            x_full, y_full, test_size=0.4, random_state=42, stratify=y_full
        )
        
        _, x_test, _, y_test = train_test_split(
            x_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
        
        print(f"📊 Training: {x_train.shape[0]:,} samples")
        print(f"📊 Testing: {x_test.shape[0]:,} samples")
        
        # Initialize models
        baseline = BaselineModels(random_state=42)
        
        # Train only FAST models
        fast_models = ['random_forest', 'logistic_regression', 'decision_tree', 'naive_bayes']
        exclude_models = ['knn', 'svm_linear']  # Skip the slow ones
        
        print(f"\n🚀 Training {len(fast_models)} FAST models...")
        
        # Train all fast models
        training_results = baseline.train_all(x_train, y_train, exclude_models=exclude_models)
        
        print(f"\n📊 Evaluating on test set...")
        
        # Evaluate only the trained models manually
        eval_results = {}
        print(f"\n🏆 CIC-IDS-2017 Baseline Results:")
        print("-" * 40)
        
        for model_name in fast_models:
            if model_name in baseline.trained_models:
                print(f"🔍 Evaluating {model_name}...")
                metrics = baseline.evaluate_model(model_name, x_test, y_test)
                eval_results[model_name] = metrics
                print(f"{model_name:20} F1={metrics['f1_score']:.3f} Acc={metrics['accuracy']:.3f}")
        
        # Save everything with dataset-specific naming
        models_dir = PROJECT_ROOT / "data" / "models" / "cic_baseline"
        results_dir = PROJECT_ROOT / "data" / "results"
        
        print(f"\n💾 Saving models and results...")
        baseline.save_models(str(models_dir), results_dir=str(results_dir), dataset_suffix="_cic")
        
        elapsed = time.time() - start_time
        print(f"\n✅ CIC BASELINE RECOVERY COMPLETE!")
        print(f"⏱️  Total time: {elapsed/60:.1f} minutes (much faster than 2+ hours!)")
        print(f"🎯 You now have CIC baseline results and can continue with advanced models")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = fast_cic_baseline_training()
    sys.exit(0 if success else 1)