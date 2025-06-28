# scripts/run_baseline.py
"""
Quick baseline training script
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from nsl_kdd_analyzer import NSLKDDAnalyzer
from data.preprocessor import NSLKDDPreprocessor  
from models.baseline import QuickBaseline

def main():
    print("🚀 Quick Baseline Training")
    print("=" * 50)
    
    # Load data
    analyzer = NSLKDDAnalyzer()
    train_data = analyzer.load_data("KDDTrain+_20Percent.txt")
    test_data = analyzer.load_data("KDDTest+.txt")
    
    if train_data is None or test_data is None:
        print("❌ Data not found!")
        return
    
    # Preprocess
    preprocessor = NSLKDDPreprocessor()
    X_train, X_val, y_train, y_val = preprocessor.fit_transform(train_data)
    X_test, y_test = preprocessor.transform(test_data)
    
    print(f"Training: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")
    
    # Train models
    baseline = QuickBaseline()
    baseline.train_all(X_train, y_train)
    
    # Evaluate
    results = baseline.evaluate_all(X_val, y_val)
    print("\n🏆 Results:")
    print(results)
    
    print("\n✅ Done! Check notebooks/ for detailed analysis.")

if __name__ == "__main__":
    main()
