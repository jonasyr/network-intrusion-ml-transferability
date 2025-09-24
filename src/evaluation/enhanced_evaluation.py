"""
Enhanced evaluation module with comprehensive scientific analysis
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve
)
from sklearn.model_selection import learning_curve as sklearn_learning_curve, validation_curve
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class EnhancedEvaluator:
    """
    Enhanced evaluation framework with comprehensive scientific analysis
    """
    
    def __init__(self, output_dir: str = "data/results"):
        """
        Initialize enhanced evaluator
        
        Args:
            output_dir: Central directory for all results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for organized storage
        self.subdirs = {
            'confusion_matrices': self.output_dir / 'confusion_matrices',
            'roc_curves': self.output_dir / 'roc_curves',
            'precision_recall_curves': self.output_dir / 'precision_recall_curves',
            'feature_importance': self.output_dir / 'feature_importance',
            'learning_curves': self.output_dir / 'learning_curves',
            'model_analysis': self.output_dir / 'model_analysis',
            'timing_analysis': self.output_dir / 'timing_analysis',
            'tables': self.output_dir / 'tables'
        }
        
        for subdir in self.subdirs.values():
            subdir.mkdir(parents=True, exist_ok=True)
            
        print(f"✅ Enhanced evaluator initialized with output: {self.output_dir}")
    
    def generate_confusion_matrix(self, 
                                y_true: np.ndarray, 
                                y_pred: np.ndarray,
                                model_name: str,
                                dataset_name: str = "test",
                                class_names: List[str] = None) -> str:
        """
        Generate and save confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels  
            model_name: Name of the model
            dataset_name: Name of the dataset
            class_names: Names of classes
            
        Returns:
            Path to saved confusion matrix
        """
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        if class_names is None:
            class_names = ['Normal', 'Attack']
        
        # Create figure
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        
        plt.title(f'Confusion Matrix: {model_name}\nDataset: {dataset_name}', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        
        # Calculate additional metrics
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Add metrics text
        metrics_text = f'Accuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1-Score: {f1:.4f}'
        plt.text(1.05, 0.5, metrics_text, transform=plt.gca().transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"),
                verticalalignment='center', fontsize=10)
        
        plt.tight_layout()
        
        # Save figure
        filename = f"{model_name.lower().replace(' ', '_')}_{dataset_name}_confusion_matrix.png"
        save_path = self.subdirs['confusion_matrices'] / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Confusion matrix saved: {save_path}")
        return str(save_path)
    
    def generate_roc_curve(self,
                          y_true: np.ndarray,
                          y_proba: np.ndarray,
                          model_name: str,
                          dataset_name: str = "test") -> str:
        """
        Generate and save ROC curve
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities for positive class
            model_name: Name of the model
            dataset_name: Name of the dataset
            
        Returns:
            Path to saved ROC curve
        """
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        
        # Create figure
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'ROC Curve: {model_name}\nDataset: {dataset_name}', 
                 fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        filename = f"{model_name.lower().replace(' ', '_')}_{dataset_name}_roc_curve.png"
        save_path = self.subdirs['roc_curves'] / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 ROC curve saved: {save_path}")
        return str(save_path)
    
    def generate_precision_recall_curve(self,
                                      y_true: np.ndarray,
                                      y_proba: np.ndarray,
                                      model_name: str,
                                      dataset_name: str = "test") -> str:
        """
        Generate and save Precision-Recall curve
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities for positive class
            model_name: Name of the model
            dataset_name: Name of the dataset
            
        Returns:
            Path to saved Precision-Recall curve
        """
        # Calculate Precision-Recall curve
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        pr_auc = auc(recall, precision)
        
        # Calculate baseline (random classifier performance)
        baseline = np.sum(y_true) / len(y_true)
        
        # Create figure
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='darkorange', lw=2, 
                label=f'PR curve (AUC = {pr_auc:.4f})')
        plt.axhline(y=baseline, color='navy', lw=2, linestyle='--', 
                   label=f'Random classifier (AP = {baseline:.4f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title(f'Precision-Recall Curve: {model_name}\nDataset: {dataset_name}', 
                 fontsize=14, fontweight='bold')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        
        # Add analysis text for imbalanced datasets
        if baseline < 0.5:
            analysis = f"Imbalanced dataset (pos: {baseline:.1%})\nPR-AUC more informative than ROC"
        else:
            analysis = f"Balanced dataset (pos: {baseline:.1%})\nBoth PR and ROC curves informative"
        
        plt.text(0.02, 0.02, analysis,
                transform=plt.gca().transAxes, verticalalignment='bottom',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"),
                fontsize=10)
        
        plt.tight_layout()
        
        # Save figure
        filename = f"{model_name.lower().replace(' ', '_')}_{dataset_name}_precision_recall_curve.png"
        # Create precision_recall_curves subdirectory if not exists
        pr_dir = self.output_dir / 'precision_recall_curves'
        pr_dir.mkdir(parents=True, exist_ok=True)
        save_path = pr_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Precision-Recall curve saved: {save_path}")
        return str(save_path)
    
    def generate_feature_importance(self,
                                  model: Any,
                                  feature_names: List[str],
                                  model_name: str,
                                  top_n: int = 20) -> Optional[str]:
        """
        Generate feature importance plot for tree-based models
        
        Args:
            model: Trained model
            feature_names: Names of features
            model_name: Name of the model
            top_n: Number of top features to show
            
        Returns:
            Path to saved feature importance plot or None
        """
        # Check if model has feature importance
        if not hasattr(model, 'feature_importances_'):
            print(f"⚠️ {model_name} does not have feature importance")
            return None
        
        # Get feature importance
        importance = model.feature_importances_
        indices = np.argsort(importance)[::-1][:top_n]
        
        # Create figure
        plt.figure(figsize=(10, 8))
        plt.title(f'Feature Importance: {model_name}', 
                 fontsize=14, fontweight='bold')
        
        # Create horizontal bar plot
        y_pos = np.arange(len(indices))
        plt.barh(y_pos, importance[indices], align='center', alpha=0.8)
        plt.yticks(y_pos, [feature_names[i] for i in indices])
        plt.xlabel('Importance Score', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        
        # Add value labels
        for i, v in enumerate(importance[indices]):
            plt.text(v + 0.001, i, f'{v:.4f}', va='center', fontsize=9)
        
        plt.tight_layout()
        
        # Save figure
        filename = f"{model_name.lower().replace(' ', '_')}_feature_importance.png"
        save_path = self.subdirs['feature_importance'] / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Also save as CSV
        importance_df = pd.DataFrame({
            'feature': [feature_names[i] for i in indices],
            'importance': importance[indices]
        })
        csv_path = self.subdirs['feature_importance'] / f"{model_name.lower().replace(' ', '_')}_feature_importance.csv"
        importance_df.to_csv(csv_path, index=False)
        
        print(f"🎯 Feature importance saved: {save_path}")
        return str(save_path)
    
    def generate_learning_curve(self,
                              model: Any,
                              X: np.ndarray,
                              y: np.ndarray,
                              model_name: str,
                              cv: int = 5,
                              train_sizes: np.ndarray = None) -> str:
        """
        Generate learning curve to analyze overfitting/underfitting
        
        Args:
            model: Trained model (will be cloned)
            X: Training features
            y: Training labels
            model_name: Name of the model
            cv: Number of CV folds
            train_sizes: Training set sizes to evaluate
            
        Returns:
            Path to saved learning curve
        """
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)
        
        # Generate learning curve with limited parallelization to prevent memory issues
        n_jobs = 1 if 'xgboost' in model_name.lower() else 2  # XGBoost has threading issues
        
        try:
            # For XGBoost, use even smaller training sizes to prevent segfault
            if 'xgboost' in model_name.lower():
                train_sizes = np.linspace(0.3, 1.0, 5)  # Fewer, larger sizes for XGBoost
                cv = 2  # Reduce CV folds for XGBoost
                # Force single threading for XGBoost to prevent segfaults
                import os
                old_nthreads = os.environ.get('OMP_NUM_THREADS')
                os.environ['OMP_NUM_THREADS'] = '1'
                
            train_sizes_abs, train_scores, val_scores = sklearn_learning_curve(
                model, X, y, cv=cv, train_sizes=train_sizes,
                scoring='f1', n_jobs=n_jobs, random_state=42
            )
            
            # Restore threading if we changed it
            if 'xgboost' in model_name.lower():
                if old_nthreads is not None:
                    os.environ['OMP_NUM_THREADS'] = old_nthreads
                else:
                    os.environ.pop('OMP_NUM_THREADS', None)
        except Exception as e:
            print(f"⚠️ Learning curve generation failed for {model_name}: {e}")
            # Return a placeholder path to prevent downstream errors
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, f'Learning curve unavailable\nfor {model_name}\n\nError: {str(e)}', 
                    ha='center', va='center', transform=plt.gca().transAxes,
                    fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            plt.title(f'{model_name} Learning Curve (Error)')
            plt.xlabel('Training Set Size')
            plt.ylabel('F1 Score')
            
            # Save error plot
            output_path = self.output_dir / "learning_curves" / f"{model_name}_learning_curve_error.png"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            return str(output_path)
        
        # Calculate mean and std
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Plot training scores
        plt.plot(train_sizes_abs, train_mean, 'o-', color='blue', 
                label='Training F1-Score')
        plt.fill_between(train_sizes_abs, train_mean - train_std, 
                        train_mean + train_std, alpha=0.1, color='blue')
        
        # Plot validation scores
        plt.plot(train_sizes_abs, val_mean, 'o-', color='red',
                label='Validation F1-Score')
        plt.fill_between(train_sizes_abs, val_mean - val_std,
                        val_mean + val_std, alpha=0.1, color='red')
        
        plt.xlabel('Training Set Size', fontsize=12)
        plt.ylabel('F1-Score', fontsize=12)
        plt.title(f'Learning Curve: {model_name}', fontsize=14, fontweight='bold')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        
        # Add analysis text
        final_gap = train_mean[-1] - val_mean[-1]
        if final_gap > 0.1:
            analysis = "Potential overfitting detected"
        elif val_mean[-1] < 0.7:
            analysis = "Potential underfitting detected"
        else:
            analysis = "Good fit achieved"
        
        plt.text(0.02, 0.98, f'Analysis: {analysis}\nTrain-Val Gap: {final_gap:.4f}',
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"),
                fontsize=10)
        
        plt.tight_layout()
        
        # Save figure
        filename = f"{model_name.lower().replace(' ', '_')}_learning_curve.png"
        save_path = self.subdirs['learning_curves'] / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📚 Learning curve saved: {save_path}")
        return str(save_path)
    
    def comprehensive_model_analysis(self,
                                   model: Any,
                                   X_test: np.ndarray,
                                   y_test: np.ndarray,
                                   X_train: np.ndarray = None,
                                   y_train: np.ndarray = None,
                                   model_name: str = "Model",
                                   dataset_name: str = "test",
                                   feature_names: List[str] = None) -> Dict[str, str]:
        """
        Perform comprehensive analysis of a model with crash prevention
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            X_train: Training features (for learning curve)
            y_train: Training labels (for learning curve)
            model_name: Name of the model
            dataset_name: Name of the dataset
            feature_names: Names of features
            
        Returns:
            Dictionary with paths to generated plots
        """
        
        # Setup safe execution environment
        try:
            from src.utils.system_limits import system_manager
            system_manager.setup_safe_environment()
        except ImportError:
            print("⚠️ System limits module not available, proceeding without safety limits")
        print(f"\n🔍 Comprehensive analysis for {model_name}")
        print("=" * 50)
        
        results = {}
        
        # Get predictions
        y_pred = model.predict(X_test)
        
        # Get probabilities if available
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, 'decision_function'):
            y_proba = model.decision_function(X_test)
        else:
            y_proba = None
        
        # 1. Confusion Matrix
        cm_path = self.generate_confusion_matrix(y_test, y_pred, model_name, dataset_name)
        results['confusion_matrix'] = cm_path
        
        # 2. ROC Curve (if probabilities available)
        if y_proba is not None:
            roc_path = self.generate_roc_curve(y_test, y_proba, model_name, dataset_name)
            results['roc_curve'] = roc_path
            
            # 2b. Precision-Recall Curve (if probabilities available)
            pr_path = self.generate_precision_recall_curve(y_test, y_proba, model_name, dataset_name)
            results['precision_recall_curve'] = pr_path
        
        # 3. Feature Importance (if available)
        if feature_names is not None:
            fi_path = self.generate_feature_importance(model, feature_names, model_name)
            if fi_path:
                results['feature_importance'] = fi_path
        
        # 4. Learning Curve (if training data available)
        if X_train is not None and y_train is not None:
            try:
                lc_path = self.generate_learning_curve(model, X_train, y_train, model_name)
                results['learning_curve'] = lc_path
            except Exception as e:
                print(f"⚠️ Learning curve generation failed for {model_name}: {e}")
                results['learning_curve'] = None
        
        return results
    
    def generate_top_models_comparison(self,
                                     models_results: List[Dict],
                                     top_n: int = 3,
                                     metric: str = 'f1_score') -> str:
        """
        Generate comparison of top N models with ROC curves
        
        Args:
            models_results: List of model results with y_true, y_proba, model_name
            top_n: Number of top models to compare
            metric: Metric to rank models by
            
        Returns:
            Path to saved comparison plot
        """
        # Sort models by metric
        sorted_models = sorted(models_results, 
                             key=lambda x: x.get(metric, 0), 
                             reverse=True)[:top_n]
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        colors = ['darkorange', 'darkblue', 'darkgreen', 'darkred', 'purple']
        
        for i, model_result in enumerate(sorted_models):
            if 'y_proba' in model_result and model_result['y_proba'] is not None:
                fpr, tpr, _ = roc_curve(model_result['y_true'], model_result['y_proba'])
                roc_auc = auc(fpr, tpr)
                
                plt.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
                        label=f"{model_result['model_name']} (AUC = {roc_auc:.4f})")
        
        # Plot random classifier
        plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--',
                label='Random classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'ROC Curves Comparison - Top {top_n} Models', 
                 fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        filename = f"top_{top_n}_models_roc_comparison.png"
        save_path = self.subdirs['model_analysis'] / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"🏆 Top models comparison saved: {save_path}")
        return str(save_path)
    
    def generate_computational_time_analysis(self,
                                           model_results: List[Dict],
                                           save_prefix: str = "computational_analysis") -> str:
        """
        Generate comprehensive computational time analysis
        
        Args:
            model_results: List of model results containing timing information
            save_prefix: Prefix for saved files
            
        Returns:
            Path to saved analysis plot
        """
        # Extract timing data
        timing_data = []
        for result in model_results:
            model_name = result.get('model_name', 'Unknown')
            training_time = result.get('training_time', 0)
            prediction_time = result.get('prediction_time', 0)
            
            # Calculate per-sample prediction time if available
            n_samples = result.get('n_test_samples', 1)
            per_sample_time = (prediction_time * 1000) / n_samples  # Convert to milliseconds
            
            timing_data.append({
                'Model': model_name,
                'Training_Time_s': training_time,
                'Prediction_Time_s': prediction_time,
                'Per_Sample_Time_ms': per_sample_time,
                'Total_Time_s': training_time + prediction_time
            })
        
        timing_df = pd.DataFrame(timing_data)
        
        # Create comprehensive timing visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Computational Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. Training Time Comparison
        axes[0, 0].barh(timing_df['Model'], timing_df['Training_Time_s'], 
                       color='skyblue', alpha=0.8)
        axes[0, 0].set_xlabel('Training Time (seconds)', fontsize=12)
        axes[0, 0].set_title('Training Time Comparison', fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(timing_df['Training_Time_s']):
            axes[0, 0].text(v + max(timing_df['Training_Time_s']) * 0.01, i, 
                           f'{v:.2f}s', va='center', fontsize=9)
        
        # 2. Prediction Time Comparison
        axes[0, 1].barh(timing_df['Model'], timing_df['Prediction_Time_s'], 
                       color='lightcoral', alpha=0.8)
        axes[0, 1].set_xlabel('Prediction Time (seconds)', fontsize=12)
        axes[0, 1].set_title('Prediction Time Comparison', fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(timing_df['Prediction_Time_s']):
            axes[0, 1].text(v + max(timing_df['Prediction_Time_s']) * 0.01, i, 
                           f'{v:.4f}s', va='center', fontsize=9)
        
        # 3. Per-Sample Prediction Time
        axes[1, 0].barh(timing_df['Model'], timing_df['Per_Sample_Time_ms'], 
                       color='lightgreen', alpha=0.8)
        axes[1, 0].set_xlabel('Per-Sample Prediction Time (milliseconds)', fontsize=12)
        axes[1, 0].set_title('Per-Sample Prediction Time', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(timing_df['Per_Sample_Time_ms']):
            axes[1, 0].text(v + max(timing_df['Per_Sample_Time_ms']) * 0.01, i, 
                           f'{v:.4f}ms', va='center', fontsize=9)
        
        # 4. Total Time vs Accuracy Trade-off (if accuracy available)
        if any('accuracy' in result for result in model_results):
            accuracy_data = [result.get('accuracy', 0) for result in model_results]
            scatter = axes[1, 1].scatter(timing_df['Total_Time_s'], accuracy_data,
                                       c=range(len(timing_df)), cmap='viridis', 
                                       s=100, alpha=0.7)
            
            # Add model name labels
            for i, (time_val, acc_val, model_name) in enumerate(
                zip(timing_df['Total_Time_s'], accuracy_data, timing_df['Model'])):
                axes[1, 1].annotate(model_name, (time_val, acc_val),
                                  xytext=(5, 5), textcoords='offset points',
                                  fontsize=8, alpha=0.8)
            
            axes[1, 1].set_xlabel('Total Time (seconds)', fontsize=12)
            axes[1, 1].set_ylabel('Accuracy', fontsize=12)
            axes[1, 1].set_title('Time vs Accuracy Trade-off', fontweight='bold')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            # Alternative: Training vs Prediction Time scatter
            axes[1, 1].scatter(timing_df['Training_Time_s'], timing_df['Prediction_Time_s'],
                             c=range(len(timing_df)), cmap='viridis', s=100, alpha=0.7)
            
            for i, (train_time, pred_time, model_name) in enumerate(
                zip(timing_df['Training_Time_s'], timing_df['Prediction_Time_s'], timing_df['Model'])):
                axes[1, 1].annotate(model_name, (train_time, pred_time),
                                  xytext=(5, 5), textcoords='offset points',
                                  fontsize=8, alpha=0.8)
            
            axes[1, 1].set_xlabel('Training Time (seconds)', fontsize=12)
            axes[1, 1].set_ylabel('Prediction Time (seconds)', fontsize=12)
            axes[1, 1].set_title('Training vs Prediction Time', fontweight='bold')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        # Create timing_analysis subdirectory if not exists
        timing_dir = self.output_dir / 'timing_analysis'
        timing_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"{save_prefix}_timing_analysis.png"
        save_path = timing_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save timing data as CSV
        csv_path = timing_dir / f"{save_prefix}_timing_data.csv"
        timing_df.to_csv(csv_path, index=False)
        
        # Generate timing summary statistics
        summary_stats = {
            'fastest_training': timing_df.loc[timing_df['Training_Time_s'].idxmin(), 'Model'],
            'fastest_prediction': timing_df.loc[timing_df['Prediction_Time_s'].idxmin(), 'Model'],
            'slowest_training': timing_df.loc[timing_df['Training_Time_s'].idxmax(), 'Model'],
            'slowest_prediction': timing_df.loc[timing_df['Prediction_Time_s'].idxmax(), 'Model'],
            'avg_training_time': timing_df['Training_Time_s'].mean(),
            'avg_prediction_time': timing_df['Prediction_Time_s'].mean(),
            'total_training_time': timing_df['Training_Time_s'].sum(),
            'total_prediction_time': timing_df['Prediction_Time_s'].sum()
        }
        
        # Save summary as JSON
        import json
        summary_path = timing_dir / f"{save_prefix}_timing_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        print(f"⏱️ Computational time analysis saved: {save_path}")
        print(f"📊 Timing data saved: {csv_path}")
        print(f"📋 Timing summary saved: {summary_path}")
        
        # Print key insights
        print(f"\n🎯 Key Timing Insights:")
        print(f"   Fastest Training: {summary_stats['fastest_training']}")
        print(f"   Fastest Prediction: {summary_stats['fastest_prediction']}")
        print(f"   Average Training Time: {summary_stats['avg_training_time']:.3f}s")
        print(f"   Average Prediction Time: {summary_stats['avg_prediction_time']:.6f}s")
        
        return str(save_path)
    
    def consolidate_results(self, 
                          source_dirs: List[str],
                          target_subdir: str = "consolidated") -> None:
        """
        Consolidate results from multiple directories to central location
        
        Args:
            source_dirs: List of source directories to consolidate
            target_subdir: Subdirectory in output_dir for consolidated results
        """
        target_dir = self.output_dir / target_subdir
        target_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n📁 Consolidating results to {target_dir}")
        
        for source_dir in source_dirs:
            source_path = Path(source_dir)
            if source_path.exists():
                print(f"  Consolidating from: {source_path}")
                
                # Copy CSV files
                for csv_file in source_path.glob("*.csv"):
                    target_file = target_dir / csv_file.name
                    if not target_file.exists():
                        import shutil
                        shutil.copy2(csv_file, target_file)
                        print(f"    ✅ Copied: {csv_file.name}")
                    else:
                        print(f"    ⚠️ Exists: {csv_file.name}")
        
        print(f"✅ Consolidation complete!")


def create_results_summary_table(evaluator: EnhancedEvaluator) -> str:
    """
    Create comprehensive results summary table
    
    Args:
        evaluator: Enhanced evaluator instance
        
    Returns:
        Path to saved summary table
    """
    # Look for all CSV result files
    result_files = {
        'baseline': evaluator.output_dir / 'baseline_results.csv',
        'advanced': evaluator.output_dir / 'advanced_results.csv',
        'cross_validation': evaluator.output_dir / 'cross_validation' / 'cv_summary_table.csv',
        'cross_dataset': evaluator.output_dir / 'cross_dataset_evaluation_fixed.csv'
    }
    
    summary_data = []
    
    for result_type, file_path in result_files.items():
        if file_path.exists():
            df = pd.read_csv(file_path)
            
            # Extract key metrics
            if 'model_name' in df.columns:
                for _, row in df.iterrows():
                    summary_data.append({
                        'Result_Type': result_type.title(),
                        'Model': row.get('model_name', row.get('Model', 'Unknown')),
                        'Accuracy': row.get('accuracy', row.get('Accuracy', 'N/A')),
                        'F1_Score': row.get('f1_score', row.get('F1', 'N/A')),
                        'Precision': row.get('precision', row.get('Precision', 'N/A')),
                        'Recall': row.get('recall', row.get('Recall', 'N/A')),
                        'ROC_AUC': row.get('roc_auc', row.get('ROC_AUC', 'N/A'))
                    })
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    # Save to CSV
    summary_path = evaluator.subdirs['tables'] / 'comprehensive_results_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    
    # Create LaTeX table
    latex_table = summary_df.to_latex(index=False, float_format="%.4f")
    latex_path = evaluator.subdirs['tables'] / 'comprehensive_results_summary.tex'
    with open(latex_path, 'w') as f:
        f.write(latex_table)
    
    print(f"📋 Comprehensive summary saved: {summary_path}")
    return str(summary_path)