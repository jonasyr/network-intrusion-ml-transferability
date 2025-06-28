# instant_test.py - Quick smoke test for core functionality
"""
Fast validation script (~10 seconds) to verify:
- Data loading works
- Basic preprocessing works  
- Simple models can train
- Core pipeline is functional

Use this for quick debugging and validation during development.
"""
import sys
sys.path.append('src')

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import time

def main():
    """Run quick smoke test"""
    print("⚡ INSTANT SMOKE TEST")
    print("=" * 30)
    start_time = time.time()
    
    try:
        # Load data
        from nsl_kdd_analyzer import NSLKDDAnalyzer
        analyzer = NSLKDDAnalyzer()
        data = analyzer.load_data("KDDTrain+_20Percent.txt")
        
        if data is None:
            print("❌ Data loading failed!")
            return False
        
        print(f"✅ Data loaded: {data.shape}")
        
        # Simple preprocessing
        print("🔄 Quick preprocessing...")
        data['is_attack'] = (data['attack_type'] != 'normal').astype(int)
        
        # Encode categoricals
        for col in ['protocol_type', 'service', 'flag']:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
        
        # Prepare features
        exclude = ['attack_type', 'difficulty_level', 'is_attack', 'attack_category']
        features = [col for col in data.columns if col not in exclude]
        
        X = data[features].values
        y = data['is_attack'].values
        
        # Scale and split
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print(f"✅ Features: {X_train.shape[1]}")
        print(f"✅ Train/Test: {X_train.shape[0]}/{X_test.shape[0]}")
        print(f"✅ Classes: Normal={np.sum(y_train==0)}, Attack={np.sum(y_train==1)}")
        
        # Quick models
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=10, random_state=42, max_depth=10),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=100)
        }
        
        print("\n🤖 Training...")
        results = {}
        for name, model in models.items():
            model_start = time.time()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            model_time = time.time() - model_start
            
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            results[name] = {'accuracy': acc, 'f1': f1, 'time': model_time}
            print(f"✅ {name}: Acc={acc:.3f}, F1={f1:.3f} ({model_time:.1f}s)")
        
        total_time = time.time() - start_time
        print(f"\n🎯 Best Model: {max(results.keys(), key=lambda k: results[k]['f1'])}")
        print(f"⏱️  Total time: {total_time:.1f}s")
        print("\n🎉 SMOKE TEST PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ SMOKE TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n💡 Try running: python test_setup.py")
        sys.exit(1)
