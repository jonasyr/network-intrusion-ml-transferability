#!/usr/bin/env python3
# scripts/run_cross_dataset_evaluation.py
"""
Cross-dataset evaluation: Train on one dataset, test on another
This is the KEY experiment for demonstrating model generalization
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import time

# Add src to path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root / 'src'))

def run_cross_dataset_evaluation():
    """
    Run comprehensive cross-dataset evaluation
    """
    print("🚀 CROSS-DATASET EVALUATION")
    print("=" * 60)
    print("This is the CRITICAL experiment for academic contribution!")
    print("Testing model generalization across different datasets")
    print("=" * 60)
    
    # Import required modules
    from nsl_kdd_analyzer import NSLKDDAnalyzer
    from data.preprocessor import NSLKDDPreprocessor
    from data.cic_ids_preprocessor import CICIDSPreprocessor
    
    # Initialize preprocessors
    nsl_preprocessor = NSLKDDPreprocessor(balance_method='smote')
    cic_preprocessor = CICIDSPreprocessor()
    
    # Load NSL-KDD data
    print("\n📁 Loading NSL-KDD dataset...")
    nsl_analyzer = NSLKDDAnalyzer()
    nsl_train = nsl_analyzer.load_data("KDDTrain+_20Percent.txt")
    nsl_test = nsl_analyzer.load_data("KDDTest+.txt")
    
    # Load CIC-IDS-2017 data
    print("\n📁 Loading CIC-IDS-2017 dataset...")
    cic_data = cic_preprocessor.load_data("data/raw/cic_ids_2017/cic_ids_sample.csv")
    
    if nsl_train is None or nsl_test is None or cic_data is None:
        print("❌ Failed to load datasets")
        return False
    
    # Preprocess NSL-KDD
    print("\n🔄 Preprocessing NSL-KDD...")
    X_nsl_train, X_nsl_val, y_nsl_train, y_nsl_val = nsl_preprocessor.fit_transform(nsl_train)
    X_nsl_test, y_nsl_test = nsl_preprocessor.transform(nsl_test)
    
    # Combine NSL-KDD train+val for cross-dataset training
    X_nsl_full = np.vstack([X_nsl_train, X_nsl_val])
    y_nsl_full = np.hstack([y_nsl_train, y_nsl_val])
    
    # Preprocess CIC-IDS-2017
    print("\n🔄 Preprocessing CIC-IDS-2017...")
    X_cic, y_cic = cic_preprocessor.fit_transform(cic_data)
    
    print(f"\n📊 Dataset Summary:")
    print(f"   NSL-KDD Training: {X_nsl_full.shape} features: {X_nsl_full.shape[1]}")
    print(f"   NSL-KDD Test:     {X_nsl_test.shape}")
    print(f"   CIC-IDS-2017:     {X_cic.shape} features: {X_cic.shape[1]}")
    
    # Load trained models (the best performers)
    model_paths = {
        'Random Forest': 'data/models/random_forest.joblib',
        'XGBoost': 'data/models/advanced/xgboost.joblib',
        'LightGBM': 'data/models/advanced/lightgbm.joblib'
    }
    
    results = []
    
    print(f"\n🧪 CROSS-DATASET EXPERIMENTS")
    print("=" * 60)
    
    for model_name, model_path in model_paths.items():
        print(f"\n🤖 Testing {model_name}...")
        
        try:
            # Load trained model
            model = joblib.load(model_path)
            
            # Experiment 1: NSL-KDD → NSL-KDD (within-dataset baseline)
            print(f"   📊 Within-dataset (NSL-KDD → NSL-KDD)...")
            start_time = time.time()
            y_pred_nsl = model.predict(X_nsl_test)
            nsl_time = time.time() - start_time
            
            nsl_accuracy = np.mean(y_pred_nsl == y_nsl_test)
            nsl_precision = precision_score_manual(y_nsl_test, y_pred_nsl)
            nsl_recall = recall_score_manual(y_nsl_test, y_pred_nsl)
            nsl_f1 = f1_score_manual(y_nsl_test, y_pred_nsl)
            
            print(f"      Accuracy: {nsl_accuracy:.4f}")
            print(f"      F1-Score: {nsl_f1:.4f}")
            
            # Experiment 2: NSL-KDD → CIC-IDS-2017 (cross-dataset)
            print(f"   🔄 Cross-dataset (NSL-KDD → CIC-IDS-2017)...")
            
            # Feature alignment challenge: NSL-KDD has 41 features, CIC has 77
            # We need to handle this mismatch
            if X_cic.shape[1] != X_nsl_full.shape[1]:
                print(f"      ⚠️ Feature mismatch: NSL-KDD={X_nsl_full.shape[1]}, CIC={X_cic.shape[1]}")
                print(f"      🔧 Using feature truncation/padding strategy...")
                
                # Simple strategy: truncate or pad to match NSL-KDD features
                if X_cic.shape[1] > X_nsl_full.shape[1]:
                    # Truncate CIC features to match NSL-KDD
                    X_cic_aligned = X_cic[:, :X_nsl_full.shape[1]]
                else:
                    # Pad CIC features to match NSL-KDD
                    padding = np.zeros((X_cic.shape[0], X_nsl_full.shape[1] - X_cic.shape[1]))
                    X_cic_aligned = np.hstack([X_cic, padding])
                
                print(f"      ✅ Aligned features: {X_cic_aligned.shape}")
            else:
                X_cic_aligned = X_cic
            
            start_time = time.time()
            y_pred_cic = model.predict(X_cic_aligned)
            cic_time = time.time() - start_time
            
            cic_accuracy = np.mean(y_pred_cic == y_cic)
            cic_precision = precision_score_manual(y_cic, y_pred_cic)
            cic_recall = recall_score_manual(y_cic, y_pred_cic)
            cic_f1 = f1_score_manual(y_cic, y_pred_cic)
            
            print(f"      Accuracy: {cic_accuracy:.4f}")
            print(f"      F1-Score: {cic_f1:.4f}")
            
            # Performance drop analysis
            accuracy_drop = nsl_accuracy - cic_accuracy
            f1_drop = nsl_f1 - cic_f1
            
            print(f"      📉 Performance Drop:")
            print(f"         Accuracy: {accuracy_drop:.4f} ({accuracy_drop/nsl_accuracy*100:.1f}%)")
            print(f"         F1-Score: {f1_drop:.4f} ({f1_drop/nsl_f1*100:.1f}%)")
            
            # Store results
            results.append({
                'Model': model_name,
                'NSL_Accuracy': nsl_accuracy,
                'NSL_F1': nsl_f1,
                'NSL_Precision': nsl_precision,
                'NSL_Recall': nsl_recall,
                'CIC_Accuracy': cic_accuracy,
                'CIC_F1': cic_f1,
                'CIC_Precision': cic_precision,
                'CIC_Recall': cic_recall,
                'Accuracy_Drop': accuracy_drop,
                'F1_Drop': f1_drop,
                'Generalization_Score': 1 - (accuracy_drop / nsl_accuracy)  # Higher is better
            })
            
        except Exception as e:
            print(f"      ❌ Error with {model_name}: {e}")
    
    # Create results summary
    if results:
        results_df = pd.DataFrame(results)
        
        print(f"\n📊 CROSS-DATASET EVALUATION RESULTS")
        print("=" * 60)
        print(results_df.round(4).to_string(index=False))
        
        # Save results
        output_path = Path("data/results/cross_dataset_evaluation.csv")
        results_df.to_csv(output_path, index=False)
        print(f"\n💾 Results saved to: {output_path}")
        
        # Key insights
        print(f"\n🔍 KEY INSIGHTS:")
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
            print(f"   • Significant generalization challenge observed")
            print(f"   • Models overfit to NSL-KDD characteristics")
            print(f"   • Need for domain adaptation techniques")
        elif avg_drop > 0.1:
            print(f"   • Moderate generalization gap")
            print(f"   • Some dataset-specific learning detected")
        else:
            print(f"   • Good generalization capability")
            print(f"   • Models learned general intrusion patterns")
        
        return True
    
    else:
        print("❌ No results generated")
        return False

def precision_score_manual(y_true, y_pred):
    """Manual precision calculation"""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tp / (tp + fp) if (tp + fp) > 0 else 0

def recall_score_manual(y_true, y_pred):
    """Manual recall calculation"""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp / (tp + fn) if (tp + fn) > 0 else 0

def f1_score_manual(y_true, y_pred):
    """Manual F1 calculation"""
    precision = precision_score_manual(y_true, y_pred)
    recall = recall_score_manual(y_true, y_pred)
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

def main():
    """Main function"""
    success = run_cross_dataset_evaluation()
    
    if success:
        print("\n🎯 CROSS-DATASET EVALUATION COMPLETE!")
        print("📊 Check data/results/cross_dataset_evaluation.csv for detailed results")
        print("🎓 These results provide crucial insights for your academic paper!")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)