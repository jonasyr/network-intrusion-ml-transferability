#!/usr/bin/env python3
# scripts/run_baseline.py
"""
Fixed baseline training script
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

def main():
    print("🚀 Quick Baseline Training")
    print("=" * 50)
    
    try:
        # Import modules
        from nsl_kdd_analyzer import NSLKDDAnalyzer
        from data.preprocessor import NSLKDDPreprocessor  
        from models.baseline import BaselineModels  # Fixed: Use BaselineModels, not QuickBaseline
        
        # Load data
        print("📁 Loading data...")
        analyzer = NSLKDDAnalyzer()
        train_data = analyzer.load_data("KDDTrain+_20Percent.txt")
        test_data = analyzer.load_data("KDDTest+.txt")
        
        if train_data is None or test_data is None:
            print("❌ Data not found!")
            return
        
        # Preprocess
        print("🔄 Preprocessing data...")
        preprocessor = NSLKDDPreprocessor(balance_method='undersample')  # Faster than SMOTE
        X_train, X_val, y_train, y_val = preprocessor.fit_transform(train_data)
        X_test, y_test = preprocessor.transform(test_data)
        
        print(f"✅ Data shapes:")
        print(f"   Training: {X_train.shape}")
        print(f"   Validation: {X_val.shape}")
        print(f"   Test: {X_test.shape}")
        
        # Train models
        print("\n🤖 Training baseline models...")
        baseline = BaselineModels()
        
        # Exclude slow models for quick test
        exclude_models = ['svm_linear'] if len(X_train) > 5000 else []
        training_results = baseline.train_all(X_train, y_train, exclude_models=exclude_models)
        
        # Evaluate on validation set
        print("\n📊 Evaluating on validation set...")
        val_results = baseline.evaluate_all(X_val, y_val)
        print("\n🏆 Validation Results:")
        print(val_results[['model_name', 'accuracy', 'f1_score', 'precision', 'recall']].round(3))
        
        # Test best model
        if len(val_results) > 0:
            best_model_name = val_results.iloc[0]['model_name']
            print(f"\n🎯 Testing best model ({best_model_name}) on test set...")
            
            test_result = baseline.evaluate_model(best_model_name, X_test, y_test)
            print(f"📈 Test Results for {best_model_name}:")
            print(f"   Accuracy:  {test_result['accuracy']:.3f}")
            print(f"   F1 Score:  {test_result['f1_score']:.3f}")
            print(f"   Precision: {test_result['precision']:.3f}")
            print(f"   Recall:    {test_result['recall']:.3f}")
            
            # Save models
            print(f"\n💾 Saving models...")
            baseline.save_models("data/models", "data/results")
            preprocessor.save("data/models/preprocessor.pkl")
            
            print(f"\n✅ Training complete!")
            print(f"🎯 Best model: {best_model_name}")
            print(f"📁 Models saved to: data/models/")
            print(f"📊 Results saved to: data/results/")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure all files are created from the artifacts")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()