#!/usr/bin/env python3
# scripts/run_reverse_cross_dataset.py
"""
Reverse cross-dataset evaluation: Train on CIC-IDS-2017, test on NSL-KDD
This completes the bidirectional generalization analysis
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# Add src to path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root / 'src'))

def run_reverse_cross_dataset_evaluation():
    """
    Train on CIC-IDS-2017, test on NSL-KDD
    """
    print("🔄 REVERSE CROSS-DATASET EVALUATION")
    print("=" * 60)
    print("Training on CIC-IDS-2017, Testing on NSL-KDD")
    print("This completes the bidirectional generalization analysis!")
    print("=" * 60)
    
    # Import required modules
    from nsl_kdd_analyzer import NSLKDDAnalyzer
    from data.preprocessor import NSLKDDPreprocessor
    from data.cic_ids_preprocessor import CICIDSPreprocessor
    
    # Initialize preprocessors
    nsl_preprocessor = NSLKDDPreprocessor(balance_method='smote')
    cic_preprocessor = CICIDSPreprocessor()
    
    # Load CIC-IDS-2017 data for training (full dataset)
    print("\n📁 Loading CIC-IDS-2017 full dataset for training...")
    cic_data = cic_preprocessor.load_data(use_full_dataset=True)
    
    if cic_data is None:
        print("❌ Failed to load CIC-IDS-2017 dataset")
        return False
    
    # Load NSL-KDD data for testing
    print("\n📁 Loading NSL-KDD dataset for testing...")
    nsl_analyzer = NSLKDDAnalyzer()
    nsl_test = nsl_analyzer.load_data("KDDTest+.txt")
    
    if nsl_test is None:
        print("❌ Failed to load NSL-KDD test data")
        return False
    
    # Preprocess CIC-IDS-2017 for training
    print("\n🔄 Preprocessing CIC-IDS-2017 for training...")
    x_cic_train, y_cic_train = cic_preprocessor.fit_transform(cic_data)
    
    # Split CIC data for training and within-dataset validation
    from sklearn.model_selection import train_test_split
    x_cic_train_split, x_cic_val, y_cic_train_split, y_cic_val = train_test_split(
        x_cic_train, y_cic_train, test_size=0.2, random_state=42, stratify=y_cic_train
    )
    
    # Preprocess NSL-KDD for testing
    print("\n🔄 Preprocessing NSL-KDD for testing...")
    # We need to fit the NSL-KDD preprocessor first to handle categorical encoding
    nsl_train_sample = nsl_analyzer.load_data("KDDTrain+_20Percent.txt")
    if nsl_train_sample is not None:
        # Fit preprocessor on training data to get proper encoders
        _, _, _, _ = nsl_preprocessor.fit_transform(nsl_train_sample)
    # Now transform test data with fitted encoders
    x_nsl_test, y_nsl_test = nsl_preprocessor.transform(nsl_test)
    
    print(f"\n📊 Dataset Summary:")
    print(f"   CIC-IDS-2017 Training: {x_cic_train_split.shape} features: {x_cic_train_split.shape[1]}")
    print(f"   CIC-IDS-2017 Validation: {x_cic_val.shape}")
    print(f"   NSL-KDD Test: {x_nsl_test.shape} features: {x_nsl_test.shape[1]}")
    
    # Handle feature mismatch (CIC has 77 features, NSL-KDD has 41)
    print(f"\n🔧 Handling feature dimension mismatch...")
    print(f"   CIC features: {x_cic_train_split.shape[1]}")
    print(f"   NSL features: {x_nsl_test.shape[1]}")
    
    if x_cic_train_split.shape[1] != x_nsl_test.shape[1]:
        # Strategy: Train on CIC features, then align NSL-KDD to match
        if x_nsl_test.shape[1] < x_cic_train_split.shape[1]:
            # Pad NSL-KDD features to match CIC dimensions
            print(f"   📈 Padding NSL-KDD features to match CIC dimensions...")
            padding_cols = x_cic_train_split.shape[1] - x_nsl_test.shape[1]
            padding = np.zeros((x_nsl_test.shape[0], padding_cols))
            x_nsl_test_aligned = np.hstack([x_nsl_test, padding])
            print(f"   ✅ Padded NSL-KDD: {x_nsl_test_aligned.shape}")
        else:
            # Truncate NSL-KDD features (shouldn't happen in this case)
            x_nsl_test_aligned = x_nsl_test[:, :x_cic_train_split.shape[1]]
    else:
        x_nsl_test_aligned = x_nsl_test
    
    # Train models on CIC-IDS-2017
    models_to_train = {
        'Random Forest': RandomForestClassifier(
            n_estimators=100, 
            random_state=42, 
            n_jobs=-1,
            max_depth=10
        ),
        'XGBoost': None,  # Will try to import
        'LightGBM': None  # Will try to import
    }
    
    # Try to import advanced models
    try:
        import xgboost as xgb
        models_to_train['XGBoost'] = xgb.XGBClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            max_depth=6
        )
    except ImportError:
        print("⚠️ XGBoost not available")
        del models_to_train['XGBoost']
    
    try:
        import lightgbm as lgb
        models_to_train['LightGBM'] = lgb.LGBMClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            max_depth=6,
            verbosity=-1
        )
    except ImportError:
        print("⚠️ LightGBM not available")
        del models_to_train['LightGBM']
    
    results = []
    
    print(f"\n🚀 TRAINING MODELS ON CIC-IDS-2017")
    print("=" * 60)
    
    for model_name, model in models_to_train.items():
        if model is None:
            continue
            
        print(f"\n🤖 Training {model_name}...")
        
        try:
            # Train on CIC-IDS-2017
            print(f"   📚 Training on CIC-IDS-2017...")
            start_time = time.time()
            model.fit(x_cic_train_split, y_cic_train_split)
            training_time = time.time() - start_time
            print(f"   ⏱️ Training time: {training_time:.2f}s")
            
            # Test 1: Within-dataset validation (CIC → CIC)
            print(f"   📊 Within-dataset validation (CIC → CIC)...")
            y_pred_cic = model.predict(x_cic_val)
            
            cic_accuracy = accuracy_score(y_cic_val, y_pred_cic)
            cic_precision = precision_score(y_cic_val, y_pred_cic, average='binary')
            cic_recall = recall_score(y_cic_val, y_pred_cic, average='binary')
            cic_f1 = f1_score(y_cic_val, y_pred_cic, average='binary')
            
            print(f"      Accuracy: {cic_accuracy:.4f}")
            print(f"      F1-Score: {cic_f1:.4f}")
            
            # Test 2: Cross-dataset evaluation (CIC → NSL-KDD)
            print(f"   🔄 Cross-dataset evaluation (CIC → NSL-KDD)...")
            start_time = time.time()
            y_pred_nsl = model.predict(x_nsl_test_aligned)
            prediction_time = time.time() - start_time
            
            nsl_accuracy = accuracy_score(y_nsl_test, y_pred_nsl)
            nsl_precision = precision_score(y_nsl_test, y_pred_nsl, average='binary')
            nsl_recall = recall_score(y_nsl_test, y_pred_nsl, average='binary')
            nsl_f1 = f1_score(y_nsl_test, y_pred_nsl, average='binary')
            
            print(f"      Accuracy: {nsl_accuracy:.4f}")
            print(f"      F1-Score: {nsl_f1:.4f}")
            print(f"      Prediction time: {prediction_time:.2f}s")
            
            # Performance drop analysis
            accuracy_drop = cic_accuracy - nsl_accuracy
            f1_drop = cic_f1 - nsl_f1
            
            print(f"      📉 Performance Drop:")
            print(f"         Accuracy: {accuracy_drop:.4f} ({accuracy_drop/cic_accuracy*100:.1f}%)")
            print(f"         F1-Score: {f1_drop:.4f} ({f1_drop/cic_f1*100:.1f}%)")
            
            # Store results
            results.append({
                'Model': model_name,
                'CIC_Accuracy': cic_accuracy,
                'CIC_F1': cic_f1,
                'CIC_Precision': cic_precision,
                'CIC_Recall': cic_recall,
                'NSL_Accuracy': nsl_accuracy,
                'NSL_F1': nsl_f1,
                'NSL_Precision': nsl_precision,
                'NSL_Recall': nsl_recall,
                'Accuracy_Drop': accuracy_drop,
                'F1_Drop': f1_drop,
                'Generalization_Score': 1 - (accuracy_drop / cic_accuracy),
                'Training_Time': training_time,
                'Prediction_Time': prediction_time
            })
            
            # Save trained model
            model_save_path = Path(f"data/models/{model_name.lower().replace(' ', '_')}_cic_trained.joblib")
            model_save_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, model_save_path)
            print(f"   💾 Saved model to: {model_save_path}")
            
        except Exception as e:
            print(f"      ❌ Error with {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Create results summary
    if results:
        results_df = pd.DataFrame(results)
        
        print(f"\n📊 REVERSE CROSS-DATASET EVALUATION RESULTS")
        print("=" * 80)
        print("Training: CIC-IDS-2017 → Testing: NSL-KDD")
        print("=" * 80)
        print(results_df.round(4).to_string(index=False))
        
        # Save results
        output_path = Path("data/results/reverse_cross_dataset_evaluation.csv")
        results_df.to_csv(output_path, index=False)
        print(f"\n💾 Results saved to: {output_path}")
        
        # Key insights
        print(f"\n🔍 KEY INSIGHTS (CIC → NSL):")
        if len(results_df) > 0:
            best_generalizer = results_df.loc[results_df['Generalization_Score'].idxmax()]
            worst_drop = results_df.loc[results_df['Accuracy_Drop'].idxmax()]
            
            print(f"   🏆 Best Generalizing Model: {best_generalizer['Model']}")
            print(f"      Generalization Score: {best_generalizer['Generalization_Score']:.4f}")
            print(f"      Accuracy Drop: {best_generalizer['Accuracy_Drop']:.4f}")
            
            print(f"   📉 Largest Performance Drop: {worst_drop['Model']}")
            print(f"      Accuracy Drop: {worst_drop['Accuracy_Drop']:.4f}")
            
            avg_drop = results_df['Accuracy_Drop'].mean()
            print(f"   📊 Average Accuracy Drop: {avg_drop:.4f}")
            
            print(f"\n💡 RESEARCH IMPLICATIONS:")
            if avg_drop > 0.2:
                print(f"   • Significant bidirectional generalization challenge")
                print(f"   • Dataset-specific feature importance patterns")
                print(f"   • Need for domain-invariant feature learning")
            elif avg_drop > 0.1:
                print(f"   • Moderate reverse generalization gap")
                print(f"   • Different dataset characteristics impact transfer")
            else:
                print(f"   • Good reverse generalization capability")
                print(f"   • Models learned generalizable patterns")
        
        return True
    
    else:
        print("❌ No results generated")
        return False

def main():
    """Main function"""
    success = run_reverse_cross_dataset_evaluation()
    
    if success:
        print("\n🎯 REVERSE CROSS-DATASET EVALUATION COMPLETE!")
        print("📊 Check data/results/reverse_cross_dataset_evaluation.csv")
        print("🔄 Bidirectional generalization analysis is now complete!")
        print("📝 Ready for comprehensive paper writing!")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)