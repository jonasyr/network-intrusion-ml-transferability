#!/usr/bin/env python3
"""
Check CIC Cross-Validation Progress
Shows current progress and which models are completed/remaining
"""

import sys
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

def check_progress():
    """Check current cross-validation progress"""
    print("🔍 CIC CROSS-VALIDATION PROGRESS CHECK")
    print("=" * 50)
    
    # Check for results file
    results_file = Path("data/results/cross_validation/cic/cv_detailed_results_cic.csv")
    
    if not results_file.exists():
        print("📄 No results file found - cross-validation not started yet")
        return
    
    try:
        # Load existing results
        df = pd.read_csv(results_file)
        completed_models = df['model_name'].tolist()
        
        print(f"✅ Found {len(completed_models)} completed models:")
        for i, model in enumerate(completed_models, 1):
            accuracy = df[df['model_name'] == model]['accuracy_mean'].iloc[0]
            accuracy_std = df[df['model_name'] == model]['accuracy_std'].iloc[0]
            print(f"  {i:2d}. {model:<25} Accuracy: {accuracy:.4f} ± {accuracy_std:.4f}")
        
        # Check which models are available but not done
        all_possible_models = []
        
        # Check baseline models
        for model in ['random_forest', 'logistic_regression', 'decision_tree', 'naive_bayes', 'knn', 'svm_linear']:
            model_path = Path(f'data/models/cic_baseline/{model}_cic.joblib')
            if model_path.exists():
                all_possible_models.append(f"{model}_cic")
        
        # Check advanced models
        for model in ['xgboost', 'lightgbm', 'gradient_boosting', 'extra_trees', 'mlp', 'voting_classifier']:
            model_path = Path(f'data/models/cic_advanced/{model}_cic.joblib')
            if model_path.exists():
                all_possible_models.append(f"{model}_cic")
        
        remaining_models = [model for model in all_possible_models if model not in completed_models]
        
        if remaining_models:
            print(f"\n⏳ {len(remaining_models)} models remaining:")
            for i, model in enumerate(remaining_models, 1):
                print(f"  {i:2d}. {model}")
            
            estimated_minutes = len(remaining_models) * 20  # rough estimate
            print(f"\n⏱️  Estimated time remaining: {estimated_minutes} minutes ({estimated_minutes/60:.1f} hours)")
        else:
            print("\n🎉 All models completed!")
            
            # Show top performers
            print(f"\n🏆 TOP 5 PERFORMERS:")
            top_models = df.nlargest(5, 'accuracy_mean')
            for i, (_, row) in enumerate(top_models.iterrows(), 1):
                print(f"  {i}. {row['model_name']:<25} {row['accuracy_mean']:.4f} ± {row['accuracy_std']:.4f}")
                
    except Exception as e:
        print(f"❌ Error reading results: {e}")

if __name__ == "__main__":
    check_progress()