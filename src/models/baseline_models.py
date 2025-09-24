# src/models/baseline.py
"""
Baseline machine learning models for NSL-KDD anomaly detection
Quick implementation for rapid prototyping and comparison
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
import time
from typing import Dict, Any, List, Tuple
import joblib
from pathlib import Path

class BaselineModels:
    """
    Collection of baseline models for intrusion detection
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize baseline models with default hyperparameters
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = self._initialize_models()
        self.trained_models: Dict[str, Any] = {}
        self.results: List[Dict] = []
        
    def _initialize_models(self) -> Dict[str, Any]:
        """Initialize all baseline models with memory-efficient settings"""
        return {
            'random_forest': RandomForestClassifier(
                n_estimators=200,  # Increased for better performance
                max_depth=25,      # Slightly increased depth
                min_samples_split=10,  # Prevent overfitting on large datasets
                min_samples_leaf=5,    # Prevent overfitting on large datasets
                random_state=self.random_state,
                n_jobs=-1
            ),
            'logistic_regression': LogisticRegression(
                random_state=self.random_state,
                max_iter=2000,     # Increased for convergence on large datasets
                solver='saga',     # Better for large datasets than liblinear
                penalty='l2',      # L2 regularization for better generalization
                C=1.0
            ),
            'decision_tree': DecisionTreeClassifier(
                max_depth=25,      # Slightly increased depth
                min_samples_split=10,  # Prevent overfitting
                min_samples_leaf=5,    # Prevent overfitting
                random_state=self.random_state
            ),
            'naive_bayes': GaussianNB(),
            'knn': KNeighborsClassifier(
                n_neighbors=3,     # Reduced for speed
                n_jobs=-1,         # Use all cores
                algorithm='kd_tree',  # Fast for high-dimensional data
                leaf_size=50,      # Larger leaf size for speed
                metric='euclidean'  # Fast metric
            ),
            'svm_linear': SVC(
                kernel='linear',
                random_state=self.random_state,
                probability=True,
                max_iter=2000,     # Increased for large datasets
                C=1.0
            )
        }
    
    def train_model(self, model_name: str, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """
        Train a single model and return training info
        
        Args:
            model_name: Name of the model to train
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Dictionary with training information
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available: {list(self.models.keys())}")
        
        print(f"🤖 Training {model_name}...")
        
        model = self.models[model_name]
        start_time = time.time()
        
        try:
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            self.trained_models[model_name] = model
            
            return {
                'model_name': model_name,
                'training_time': training_time,
                'status': 'success',
                'error': None
            }
            
        except Exception as e:
            training_time = time.time() - start_time
            print(f"❌ Error training {model_name}: {str(e)}")
            
            return {
                'model_name': model_name,
                'training_time': training_time,
                'status': 'failed',
                'error': str(e)
            }
    
    def train_all(self, X_train: np.ndarray, y_train: np.ndarray, 
                  exclude_models: List[str] = None) -> Dict[str, Any]:
        """
        Train all baseline models
        
        Args:
            X_train: Training features
            y_train: Training labels
            exclude_models: List of model names to skip
            
        Returns:
            Dictionary with all training results
        """
        if exclude_models is None:
            exclude_models = []
        
        print(f"🚀 Training {len(self.models) - len(exclude_models)} baseline models...")
        print(f"Training data shape: {X_train.shape}")
        print(f"Class distribution: {np.bincount(y_train)}")
        
        training_results = {}
        
        for model_name in self.models:
            if model_name in exclude_models:
                print(f"⏭️ Skipping {model_name}")
                continue
                
            result = self.train_model(model_name, X_train, y_train)
            training_results[model_name] = result
            
            if result['status'] == 'success':
                print(f"✅ {model_name}: {result['training_time']:.2f}s")
            else:
                print(f"❌ {model_name}: Failed")
        
        successful_models = [name for name, result in training_results.items() 
                           if result['status'] == 'success']
        
        print(f"\n📊 Training Summary:")
        print(f"✅ Successful: {len(successful_models)}/{len(training_results)}")
        print(f"🤖 Models ready: {', '.join(successful_models)}")
        
        return training_results
    
    def evaluate_model(self, model_name: str, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate a single trained model
        
        Args:
            model_name: Name of the model to evaluate
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained yet")
        
        model = self.trained_models[model_name]
        
        # Make predictions
        start_time = time.time()
        y_pred = model.predict(X_test)
        prediction_time = time.time() - start_time
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # ROC AUC (for binary classification)
        try:
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)
                if y_proba.shape[1] == 2:  # Binary classification
                    roc_auc = roc_auc_score(y_test, y_proba[:, 1])
                else:  # Multiclass
                    roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
            else:
                roc_auc = None
        except Exception:
            roc_auc = None
        
        return {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'prediction_time': prediction_time,
            'predictions': y_pred
        }
    
    def evaluate_all(self, X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
        """
        Evaluate all trained models
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            DataFrame with all model results
        """
        print(f"📊 Evaluating {len(self.trained_models)} models...")
        print(f"Test data shape: {X_test.shape}")
        
        self.results = []
        
        for model_name in self.trained_models:
            print(f"🔍 Evaluating {model_name}...")
            
            try:
                result = self.evaluate_model(model_name, X_test, y_test)
                self.results.append(result)
                
                print(f"✅ {model_name}: F1={result['f1_score']:.3f}, Acc={result['accuracy']:.3f}")
                
            except Exception as e:
                print(f"❌ Error evaluating {model_name}: {str(e)}")
        
        # Create results DataFrame
        results_df = pd.DataFrame(self.results)
        
        if len(results_df) > 0:
            # Sort by F1 score
            results_df = results_df.sort_values('f1_score', ascending=False)
            
            print(f"\n🏆 Model Ranking (by F1 Score):")
            for i, (_, row) in enumerate(results_df.iterrows(), 1):
                print(f"{i}. {row['model_name']:<20} F1: {row['f1_score']:.3f} | "
                      f"Acc: {row['accuracy']:.3f} | Prec: {row['precision']:.3f} | "
                      f"Rec: {row['recall']:.3f}")
        
        return results_df
    
    def get_detailed_report(self, model_name: str, X_test: np.ndarray, y_test: np.ndarray) -> str:
        """
        Get detailed classification report for a specific model
        
        Args:
            model_name: Name of the model
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Detailed classification report as string
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained yet")
        
        model = self.trained_models[model_name]
        y_pred = model.predict(X_test)
        
        report = f"Detailed Report for {model_name}\n"
        report += "=" * 50 + "\n\n"
        report += classification_report(y_test, y_pred)
        report += "\n\nConfusion Matrix:\n"
        report += str(confusion_matrix(y_test, y_pred))
        
        return report
    
    def save_models(self, output_dir: str, results_dir: str = None, dataset_suffix: str = ""):
        """
        Save all trained models to disk
        
        Args:
            output_dir: Directory to save models
            results_dir: Directory to save results (defaults to output_dir)
            dataset_suffix: Optional suffix to add to model names (e.g., "_cic_trained")
        """
        models_path = Path(output_dir)
        models_path.mkdir(parents=True, exist_ok=True)
        
        # Use separate results directory if provided, otherwise same as models
        results_path = Path(results_dir) if results_dir else models_path
        results_path.mkdir(parents=True, exist_ok=True)
        
        for model_name, model in self.trained_models.items():
            filename = f"{model_name}{dataset_suffix}.joblib"
            model_path = models_path / filename
            joblib.dump(model, model_path)
            print(f"💾 Saved {model_name} to {model_path}")
        
        # Save results DataFrame with dataset-specific naming
        if self.results:
            results_df = pd.DataFrame(self.results)
            
            # Determine dataset from suffix or results_path
            dataset_name = "nsl"  # default
            if "cic" in dataset_suffix.lower():
                dataset_name = "cic"
            elif "cic" in str(results_path).lower():
                dataset_name = "cic"
            
            # Always save to data/results with dataset-specific name
            results_output_dir = Path("data/results")
            results_output_dir.mkdir(parents=True, exist_ok=True)
            
            dataset_results_file = results_output_dir / f"{dataset_name}_baseline_results.csv"
            results_df.to_csv(dataset_results_file, index=False)
            print(f"💾 Saved {dataset_name.upper()} baseline results to {dataset_results_file}")
            
            # NO MORE DUPLICATE SAVES - only dataset-specific naming from now on!
    
    def load_models(self, input_dir: str):
        """
        Load trained models from disk
        
        Args:
            input_dir: Directory containing saved models
        """
        input_path = Path(input_dir)
        
        for model_name in self.models.keys():
            model_path = input_path / f"{model_name}.joblib"
            if model_path.exists():
                self.trained_models[model_name] = joblib.load(model_path)
                print(f"📁 Loaded {model_name} from {model_path}")
    
    def get_best_model(self, metric: str = 'f1_score') -> Tuple[str, Any]:
        """
        Get the best performing model
        
        Args:
            metric: Metric to use for comparison
            
        Returns:
            Tuple of (model_name, model_object)
        """
        if not self.results:
            raise ValueError("No evaluation results available. Run evaluate_all() first.")
        
        results_df = pd.DataFrame(self.results)
        best_model_name = results_df.loc[results_df[metric].idxmax(), 'model_name']
        best_model = self.trained_models[best_model_name]
        
        print(f"🏆 Best model by {metric}: {best_model_name}")
        return best_model_name, best_model


# Quick training script
def quick_baseline_training():
    """
    Quick baseline training script for immediate results
    """
    print("🚀 Quick Baseline Training Script")
    print("=" * 60)
    
    # Import required modules
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))

    from src.preprocessing import NSLKDDAnalyzer, NSLKDDPreprocessor
    
    # Load data
    print("📁 Loading data...")
    analyzer = NSLKDDAnalyzer()
    train_data = analyzer.load_data("KDDTrain+.txt")
    test_data = analyzer.load_data("KDDTest+.txt")
    
    if train_data is None or test_data is None:
        print("❌ Failed to load data")
        return
    
    # Preprocess data
    print("🔄 Preprocessing data...")
    preprocessor = NSLKDDPreprocessor(balance_method='undersample')  # Faster than SMOTE
    X_train, X_val, y_train, y_val = preprocessor.fit_transform(train_data, target_type='binary')
    X_test, y_test = preprocessor.transform(test_data, target_type='binary')
    
    # Initialize and train models
    print("🤖 Training baseline models...")
    baseline = BaselineModels()
    
    # Exclude SVM for speed (can be slow on large datasets)
    exclude_models = ['svm_linear'] if len(X_train) > 10000 else []
    
    training_results = baseline.train_all(X_train, y_train, exclude_models=exclude_models)
    
    # Evaluate models
    print("📊 Evaluating models...")
    results_df = baseline.evaluate_all(X_val, y_val)  # Use validation set
    
    # Test best model on test set
    if len(results_df) > 0:
        best_model_name = results_df.iloc[0]['model_name']
        print(f"\n🏆 Testing best model ({best_model_name}) on test set...")
        
        test_result = baseline.evaluate_model(best_model_name, X_test, y_test)
        print(f"📈 Test Results:")
        print(f"   Accuracy: {test_result['accuracy']:.3f}")
        print(f"   F1 Score: {test_result['f1_score']:.3f}")
        print(f"   Precision: {test_result['precision']:.3f}")
        print(f"   Recall: {test_result['recall']:.3f}")
        
        # Detailed report
        print(f"\n📋 Detailed Report for {best_model_name}:")
        print(baseline.get_detailed_report(best_model_name, X_test, y_test))
    
    # Save everything
    print("💾 Saving models and results...")
    baseline.save_models("data/models/baseline")
    preprocessor.save("data/models/preprocessor.pkl")
    
    print("\n✅ Baseline training complete!")
    print("🎯 Next steps:")
    print("   • Analyze results in detail")
    print("   • Try different preprocessing options")
    print("   • Experiment with hyperparameter tuning")
    print("   • Implement advanced models")


if __name__ == "__main__":
    quick_baseline_training()