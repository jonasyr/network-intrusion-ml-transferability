#!/usr/bin/env python3
"""
CIC-IDS-2017 Baseline Models Analysis
Generate comprehensive baseline performance analysis for CIC-IDS-2017 dataset
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

def run_cic_baseline_analysis():
    """
    Run comprehensive baseline analysis on CIC-IDS-2017 dataset
    """
    print("🚀 CIC-IDS-2017 BASELINE ANALYSIS")
    print("=" * 60)
    print("Comprehensive evaluation of baseline models on CIC-IDS-2017")
    print("=" * 60)
    
    try:
        # Import modules
        from src.preprocessing import CICIDSPreprocessor
        from src.models import BaselineModels
        
        # Load and preprocess CIC-IDS-2017 data
        print("\n📁 Loading CIC-IDS-2017 dataset...")
        cic_preprocessor = CICIDSPreprocessor()
        cic_data = cic_preprocessor.load_data(use_full_dataset=True)
        
        if cic_data is None:
            print("❌ Error: Could not load CIC-IDS-2017 data!")
            return
        
        print("🔄 Preprocessing CIC-IDS-2017...")
        X_cic, y_cic = cic_preprocessor.fit_transform(cic_data)
        
        # Sample the dataset for baseline analysis (full dataset is too large)
        print("🎲 Sampling dataset for baseline analysis...")
        from sklearn.model_selection import train_test_split
        
        # First, create a manageable sample (100k samples)
        sample_size = min(100000, len(X_cic))
        if len(X_cic) > sample_size:
            print(f"   📉 Sampling {sample_size:,} from {len(X_cic):,} total samples")
            X_sample, _, y_sample, _ = train_test_split(
                X_cic, y_cic, train_size=sample_size, random_state=42, stratify=y_cic
            )
        else:
            X_sample, y_sample = X_cic, y_cic
        
        # Now split the sample into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X_sample, y_sample, test_size=0.2, random_state=42, stratify=y_sample
        )
        
        print(f"\n📊 CIC-IDS-2017 Dataset Summary:")
        print(f"   Training: {X_train.shape}")
        print(f"   Test:     {X_test.shape}")
        print(f"   Features: {X_train.shape[1]}")
        print(f"   Classes:  {len(np.unique(y_cic))} (Binary: Normal/Attack)")
        
        # Initialize baseline models
        print("\n🤖 BASELINE MODELS EVALUATION")
        print("=" * 60)
        
        baseline_models = BaselineModels()
        
        # Models to evaluate
        models_to_test = [
            'logistic_regression',
            'decision_tree', 
            'random_forest',
            'naive_bayes',
            'knn'
        ]
        
        results = []
        
        for model_name in models_to_test:
            print(f"\n🔧 Training {model_name.replace('_', ' ').title()}...")
            
            # Train model using BaselineModels interface
            start_time = time.time()
            result = baseline_models.train_model(model_name, X_train, y_train)
            training_time = time.time() - start_time
            
            # Get trained model
            model = baseline_models.trained_models[model_name]
            
            # Evaluate model
            start_time = time.time()
            y_pred = model.predict(X_test)
            inference_time = time.time() - start_time
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
            recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
            
            print(f"   ✅ Accuracy:  {accuracy:.4f}")
            print(f"   ✅ Precision: {precision:.4f}")
            print(f"   ✅ Recall:    {recall:.4f}")
            print(f"   ✅ F1-Score:  {f1:.4f}")
            print(f"   ⏱️ Training:  {training_time:.2f}s")
            print(f"   ⏱️ Inference: {inference_time:.4f}s")
            
            # Store results
            results.append({
                'Model': model_name.replace('_', ' ').title(),
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1_Score': f1,
                'Training_Time': training_time,
                'Inference_Time': inference_time,
                'Dataset': 'CIC-IDS-2017',
                'Features': X_train.shape[1],
                'Train_Samples': X_train.shape[0],
                'Test_Samples': X_test.shape[0]
            })
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Display summary
        print(f"\n📊 CIC-IDS-2017 BASELINE RESULTS SUMMARY")
        print("=" * 60)
        print(results_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1_Score']].to_string(index=False))
        
        # Find best models
        best_accuracy = results_df.loc[results_df['Accuracy'].idxmax()]
        best_f1 = results_df.loc[results_df['F1_Score'].idxmax()]
        
        print(f"\n🏆 BEST PERFORMING MODELS:")
        print(f"   🎯 Best Accuracy: {best_accuracy['Model']} ({best_accuracy['Accuracy']:.4f})")
        print(f"   🎯 Best F1-Score: {best_f1['Model']} ({best_f1['F1_Score']:.4f})")
        
        # Save results
        output_path = "data/results/cic_baseline_results.csv"
        results_df.to_csv(output_path, index=False)
        print(f"\n💾 Results saved to: {output_path}")
        
        # Performance analysis
        print(f"\n🔍 PERFORMANCE ANALYSIS:")
        avg_accuracy = results_df['Accuracy'].mean()
        avg_f1 = results_df['F1_Score'].mean()
        print(f"   📈 Average Accuracy: {avg_accuracy:.4f}")
        print(f"   📈 Average F1-Score: {avg_f1:.4f}")
        
        # Compare with NSL-KDD results if available
        try:
            nsl_results_path = "data/results/baseline_results.csv"
            nsl_results = pd.read_csv(nsl_results_path)
            
            print("\n🔄 COMPARISON WITH NSL-KDD:")
            print(f"   NSL-KDD Avg Accuracy: {nsl_results['accuracy'].mean():.4f}")
            print(f"   CIC-IDS Avg Accuracy: {avg_accuracy:.4f}")
            print(f"   NSL-KDD Avg F1-Score: {nsl_results['f1_score'].mean():.4f}")
            print(f"   CIC-IDS Avg F1-Score: {avg_f1:.4f}")
            
        except FileNotFoundError:
            print("\n⚠️ NSL-KDD baseline results not found for comparison")
        
        print(f"\n🎯 CIC-IDS-2017 BASELINE ANALYSIS COMPLETE!")
        print(f"📊 Results provide foundation for CIC-specific analysis")
        print(f"🎓 Use these baselines for cross-dataset comparison")
        
        return results_df
        
    except Exception as e:
        print(f"❌ Error in CIC baseline analysis: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    run_cic_baseline_analysis()