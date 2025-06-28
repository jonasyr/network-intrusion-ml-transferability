# quick_setup_and_run.py
"""
One-click setup and baseline training for NSL-KDD anomaly detection
This script will:
1. Create proper project structure
2. Fix the notebook issue
3. Run baseline preprocessing and training
4. Generate initial results

Run this after fixing your project structure!
"""

import os
import sys
import json
from pathlib import Path
import shutil

def create_proper_notebook():
    """Create a proper Jupyter notebook file"""
    
    notebook_content = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# NSL-KDD Dataset Exploration\\n",
                    "\\n",
                    "Comprehensive analysis of the NSL-KDD intrusion detection dataset."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Import libraries\\n",
                    "import sys\\n",
                    "sys.path.append('../src')\\n",
                    "\\n",
                    "import pandas as pd\\n",
                    "import numpy as np\\n",
                    "import matplotlib.pyplot as plt\\n",
                    "import seaborn as sns\\n",
                    "from nsl_kdd_analyzer import NSLKDDAnalyzer\\n",
                    "\\n",
                    "plt.style.use('default')\\n",
                    "sns.set_palette(\\\"husl\\\")\\n",
                    "%matplotlib inline\\n",
                    "\\n",
                    "import warnings\\n",
                    "warnings.filterwarnings('ignore')\\n",
                    "\\n",
                    "print(\\\"📊 Environment Ready!\\\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Quick analysis\\n",
                    "analyzer = NSLKDDAnalyzer()\\n",
                    "data = analyzer.load_data('KDDTrain+_20Percent.txt')\\n",
                    "\\n",
                    "if data is not None:\\n",
                    "    print(f\\\"Dataset shape: {data.shape}\\\")\\n",
                    "    print(f\\\"Attack distribution:\\\")\\n",
                    "    print(data['attack_category'].value_counts())"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.9.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    # Save the proper notebook
    notebook_path = Path("notebooks/01_data_exploration_FIXED.ipynb")
    notebook_path.parent.mkdir(exist_ok=True)
    
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook_content, f, indent=2)
    
    print(f"✅ Created proper notebook: {notebook_path}")
    return notebook_path

def create_project_structure():
    """Create missing directories and files"""
    
    directories = [
        "src/data",
        "src/models", 
        "src/evaluation",
        "src/utils",
        "data/models",
        "data/features",
        "experiments/configs",
        "experiments/runs",
        "reports",
        "scripts"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        
        # Create __init__.py files for Python packages
        if directory.startswith("src/"):
            init_file = Path(directory) / "__init__.py"
            if not init_file.exists():
                init_file.touch()
    
    print("✅ Created project structure")

def create_init_files():
    """Create __init__.py files to make src a proper package"""
    
    init_files = [
        "src/__init__.py",
        "src/data/__init__.py", 
        "src/models/__init__.py",
        "src/evaluation/__init__.py",
        "src/utils/__init__.py"
    ]
    
    for init_file in init_files:
        Path(init_file).touch()
    
    print("✅ Created __init__.py files")

def create_preprocessor_file():
    """Create the preprocessor.py file in src/data/"""
    
    preprocessor_code = '''# src/data/preprocessor.py
"""
Baseline preprocessing pipeline for NSL-KDD dataset
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Tuple
import pickle

class NSLKDDPreprocessor:
    """Simple preprocessor for quick baseline"""
    
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.label_encoders = {}
        self.scaler = None
        
        # Attack categories mapping
        self.attack_categories = {
            'normal': 'Normal',
            'back': 'DoS', 'land': 'DoS', 'neptune': 'DoS', 'pod': 'DoS', 
            'smurf': 'DoS', 'teardrop': 'DoS',
            'ipsweep': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe', 'satan': 'Probe',
            'ftp_write': 'R2L', 'guess_passwd': 'R2L', 'imap': 'R2L', 'multihop': 'R2L',
            'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'perl': 'U2R', 'rootkit': 'U2R'
        }
        
        self.categorical_features = ['protocol_type', 'service', 'flag']
    
    def preprocess_data(self, data):
        """Simple preprocessing"""
        data = data.copy()
        
        # Add attack categories
        data['attack_category'] = data['attack_type'].map(self.attack_categories)
        data['attack_category'] = data['attack_category'].fillna('Unknown')
        
        # Binary labels
        data['is_attack'] = (data['attack_type'] != 'normal').astype(int)
        
        return data
    
    def prepare_features(self, data, fit=True):
        """Prepare features for ML"""
        data = self.preprocess_data(data)
        
        # Encode categorical
        for feature in self.categorical_features:
            if feature in data.columns:
                if fit:
                    self.label_encoders[feature] = LabelEncoder()
                    data[feature] = self.label_encoders[feature].fit_transform(data[feature])
                else:
                    if feature in self.label_encoders:
                        # Handle unseen values
                        le = self.label_encoders[feature]
                        data[feature] = data[feature].apply(
                            lambda x: le.transform([x])[0] if x in le.classes_ else -1
                        )
        
        # Get feature columns
        exclude_cols = ['attack_type', 'attack_category', 'difficulty_level', 'is_attack']
        feature_cols = [col for col in data.columns if col not in exclude_cols]
        
        X = data[feature_cols]
        
        # Scale features
        if fit:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)
        
        y = data['is_attack'].values
        
        return X, y
    
    def fit_transform(self, train_data):
        """Fit and transform training data"""
        X, y = self.prepare_features(train_data, fit=True)
        
        # Split into train/val
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.test_size, 
            random_state=self.random_state, 
            stratify=y
        )
        
        return X_train, X_val, y_train, y_val
    
    def transform(self, test_data):
        """Transform test data"""
        X, y = self.prepare_features(test_data, fit=False)
        return X, y
'''
    
    preprocessor_path = Path("src/data/preprocessor.py")
    with open(preprocessor_path, 'w') as f:
        f.write(preprocessor_code)
    
    print("✅ Created preprocessor.py")

def create_baseline_models_file():
    """Create baseline_models.py file"""
    
    models_code = '''# src/models/baseline.py
"""
Simple baseline models for quick testing
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, classification_report
import pandas as pd
import time

class QuickBaseline:
    """Quick baseline models for rapid prototyping"""
    
    def __init__(self):
        self.models = {
            'random_forest': RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1),
            'logistic_regression': LogisticRegression(random_state=42, max_iter=500),
            'decision_tree': DecisionTreeClassifier(max_depth=15, random_state=42),
            'naive_bayes': GaussianNB()
        }
        self.trained_models = {}
        self.results = []
    
    def train_all(self, X_train, y_train):
        """Train all models"""
        print("🤖 Training baseline models...")
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            start_time = time.time()
            
            try:
                model.fit(X_train, y_train)
                training_time = time.time() - start_time
                self.trained_models[name] = model
                print(f"✅ {name}: {training_time:.2f}s")
            except Exception as e:
                print(f"❌ {name}: {str(e)}")
    
    def evaluate_all(self, X_test, y_test):
        """Evaluate all trained models"""
        print("📊 Evaluating models...")
        
        for name, model in self.trained_models.items():
            y_pred = model.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            self.results.append({
                'model': name,
                'accuracy': accuracy,
                'f1_score': f1
            })
            
            print(f"{name}: Acc={accuracy:.3f}, F1={f1:.3f}")
        
        # Return results sorted by F1 score
        results_df = pd.DataFrame(self.results)
        return results_df.sort_values('f1_score', ascending=False)
'''
    
    models_path = Path("src/models/baseline.py")
    with open(models_path, 'w') as f:
        f.write(models_code)
    
    print("✅ Created baseline.py")

def create_run_script():
    """Create a simple run script"""
    
    run_code = '''# scripts/run_baseline.py
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
    print("\\n🏆 Results:")
    print(results)
    
    print("\\n✅ Done! Check notebooks/ for detailed analysis.")

if __name__ == "__main__":
    main()
'''
    
    script_path = Path("scripts/run_baseline.py")
    script_path.parent.mkdir(exist_ok=True)
    with open(script_path, 'w') as f:
        f.write(run_code)
    
    print("✅ Created run_baseline.py")

def update_requirements():
    """Update requirements.txt with essential packages only"""
    
    requirements = '''# Core packages for ML anomaly detection
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.2.0
matplotlib>=3.6.0
seaborn>=0.11.0
jupyter>=1.0.0
imbalanced-learn>=0.8.0
'''
    
    with open("requirements.txt", 'w') as f:
        f.write(requirements)
    
    print("✅ Updated requirements.txt")

def main():
    """Run complete setup"""
    print("🔧 QUICK SETUP & RESTRUCTURE")
    print("=" * 60)
    
    print("\\n1. Creating project structure...")
    create_project_structure()
    create_init_files()
    
    print("\\n2. Fixing notebook...")
    notebook_path = create_proper_notebook()
    
    print("\\n3. Creating core modules...")
    create_preprocessor_file()
    create_baseline_models_file()
    create_run_script()
    
    print("\\n4. Updating requirements...")
    update_requirements()
    
    print("\\n" + "=" * 60)
    print("✅ SETUP COMPLETE!")
    print("=" * 60)
    
    print("\\n🎯 NEXT STEPS:")
    print("1. Install/update packages:")
    print("   pip install -r requirements.txt")
    
    print("\\n2. Test the fixed notebook:")
    print(f"   jupyter lab {notebook_path}")
    
    print("\\n3. Run quick baseline training:")
    print("   python scripts/run_baseline.py")
    
    print("\\n4. Or continue with detailed analysis:")
    print("   python run_analysis.py")
    
    print("\\n📁 Key files created:")
    print("   • notebooks/01_data_exploration_FIXED.ipynb")
    print("   • src/data/preprocessor.py")
    print("   • src/models/baseline.py") 
    print("   • scripts/run_baseline.py")
    
    print("\\n🎉 You're ready to iterate!")

if __name__ == "__main__":
    main()
'''

# Also create a simple test script
def create_test_script():
    """Test that everything works"""
    print("\\n🧪 Testing setup...")
    
    try:
        # Test imports
        sys.path.append('src')
        from nsl_kdd_analyzer import NSLKDDAnalyzer
        print("✅ NSLKDDAnalyzer import works")
        
        # Test data loading
        analyzer = NSLKDDAnalyzer()
        data_files = list(analyzer.data_dir.glob("*.txt"))
        if data_files:
            print(f"✅ Found {len(data_files)} data files")
        else:
            print("⚠️ No data files found - make sure NSL-KDD files are in data/raw/")
        
        print("✅ Basic setup test passed!")
        
    except Exception as e:
        print(f"❌ Setup test failed: {e}")
        print("💡 Try running: pip install -r requirements.txt")

if __name__ == "__main__":
    main()
    create_test_script()
'''