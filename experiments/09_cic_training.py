#!/usr/bin/env python3
"""
Training script for CIC-IDS-2017 dataset
Creates models saved in cic_baseline/ and cic_advanced/ folders
"""

import sys
from pathlib import Path

# Add project root to path
current_dir = Path(__file__).parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

def train_cic_baseline():
    """Train baseline models on CIC-IDS-2017 dataset"""
    print("🚀 CIC-IDS-2017 Baseline Training")
    print("=" * 50)
    
    try:
        # Import modules
        from src.preprocessing import CICIDSPreprocessor
        from src.models import BaselineModels
        
        # Load and preprocess CIC data
        print("📁 Loading CIC-IDS-2017 data...")
        preprocessor = CICIDSPreprocessor()
        cic_data = preprocessor.load_data(use_full_dataset=False)  # Use sample for testing
        
        if cic_data is None:
            print("❌ Failed to load CIC-IDS-2017 data!")
            return False
        
        print("🔄 Preprocessing CIC data...")
        X_train, y_train = preprocessor.fit_transform(cic_data)
        
        print(f"✅ CIC data prepared:")
        print(f"   Training: {X_train.shape}")
        print(f"   Labels: {y_train.shape}")
        print(f"   Classes: {len(set(y_train))}")
        
        # Train baseline models
        print("\n🤖 Training baseline models on CIC data...")
        baseline = BaselineModels()
        
        # Train selected models (exclude slow ones for large datasets)
        exclude_models = ['svm_linear', 'knn'] if len(X_train) > 10000 else []
        training_results = baseline.train_all(X_train, y_train, exclude_models=exclude_models)
        
        # Evaluate models (using a portion for validation)
        from sklearn.model_selection import train_test_split
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        print("\n📊 Evaluating baseline models...")
        val_results = baseline.evaluate_all(X_val, y_val)
        
        if len(val_results) > 0:
            print("\n🏆 CIC Baseline Results:")
            print(val_results[['model_name', 'accuracy', 'f1_score', 'precision', 'recall']].round(3))
            
            # Save models to CIC baseline folder
            print("\n💾 Saving CIC baseline models...")
            baseline.save_models(
                "data/models/cic_baseline", 
                "data/results", 
                dataset_suffix="_cic_trained"
            )
            
            # Save preprocessor
            preprocessor.save("data/models/cic_preprocessor.pkl")
            
            print(f"\n✅ CIC baseline training complete!")
            print(f"📁 Models saved to: data/models/cic_baseline/")
            return True
        else:
            print("❌ No models were successfully trained")
            return False
            
    except Exception as e:
        print(f"❌ Error in CIC baseline training: {e}")
        import traceback
        traceback.print_exc()
        return False

def train_cic_advanced():
    """Train advanced models on CIC-IDS-2017 dataset"""
    print("\n🚀 CIC-IDS-2017 Advanced Training")
    print("=" * 50)
    
    try:
        # Import modules
        from src.preprocessing import CICIDSPreprocessor
        from src.models import AdvancedModels
        
        # Load and preprocess CIC data
        print("📁 Loading CIC-IDS-2017 data...")
        preprocessor = CICIDSPreprocessor()
        cic_data = preprocessor.load_data(use_full_dataset=False)  # Use sample for testing
        
        if cic_data is None:
            print("❌ Failed to load CIC-IDS-2017 data!")
            return False
        
        print("🔄 Preprocessing CIC data...")
        X_train, y_train = preprocessor.fit_transform(cic_data)
        
        print(f"✅ CIC data prepared:")
        print(f"   Training: {X_train.shape}")
        print(f"   Labels: {y_train.shape}")
        print(f"   Classes: {len(set(y_train))}")
        
        # Train advanced models
        print("\n🤖 Training advanced models on CIC data...")
        advanced = AdvancedModels()
        
        # Train selected models (you can include/exclude specific models)
        training_results = advanced.train_all(X_train, y_train)
        
        # Evaluate models (using a portion for validation)
        from sklearn.model_selection import train_test_split
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        print("\n📊 Evaluating advanced models...")
        val_results = advanced.evaluate_all(X_val, y_val)
        
        if len(val_results) > 0:
            print("\n🏆 CIC Advanced Results:")
            print(val_results.round(3))
            
            # Save models to CIC advanced folder
            print("\n💾 Saving CIC advanced models...")
            advanced.save_models(
                "data/models/cic_advanced", 
                results_dir="data/results", 
                results_filename="cic_advanced_results.csv",
                dataset_suffix="_cic_trained"
            )
            
            print(f"\n✅ CIC advanced training complete!")
            print(f"📁 Models saved to: data/models/cic_advanced/")
            return True
        else:
            print("❌ No advanced models were successfully trained")
            return False
            
    except Exception as e:
        print(f"❌ Error in CIC advanced training: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run CIC training pipeline"""
    print("🚀 CIC-IDS-2017 COMPREHENSIVE TRAINING")
    print("=" * 60)
    
    success_baseline = train_cic_baseline()
    success_advanced = train_cic_advanced()
    
    print("\n" + "=" * 60)
    if success_baseline and success_advanced:
        print("✅ CIC training pipeline completed successfully!")
        print("📁 Baseline models: data/models/cic_baseline/")
        print("📁 Advanced models: data/models/cic_advanced/")
    elif success_baseline:
        print("⚠️ CIC training completed with warnings (only baseline models trained)")
    elif success_advanced:
        print("⚠️ CIC training completed with warnings (only advanced models trained)")
    else:
        print("❌ CIC training pipeline failed")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)