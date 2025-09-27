# src/metrics/cross_validation.py
"""
Cross-validation and statistical validation framework with memory adaptation
For rigorous model comparison and academic publication
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from scipy import stats
from typing import Dict, List, Tuple, Any
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

class CrossValidationFramework:
    """
    Comprehensive cross-validation and statistical testing framework
    """
    
    def __init__(self, n_folds: int = 5, random_state: int = 42):
        """
        Initialize cross-validation framework
        
        Args:
            n_folds: Number of folds for cross-validation
            random_state: Random seed for reproducibility
        """
        self.n_folds = n_folds
        self.random_state = random_state
        self.cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        self.results = {}
        
    def evaluate_model_cv(self, model, X: np.ndarray, y: np.ndarray, model_name: str) -> Dict[str, Any]:
        """
        Perform cross-validation on a single model
        
        Args:
            model: Trained model object
            X: Feature matrix
            y: Target vector
            model_name: Name of the model
            
        Returns:
            Dictionary with cross-validation results
        """
        print(f"🔄 Cross-validating {model_name}... (this may take 15-30 minutes)")
        
        # Define scoring metrics
        scoring = {
            'accuracy': 'accuracy',
            'f1': 'f1',
            'precision': 'precision',
            'recall': 'recall',
            'roc_auc': 'roc_auc'
        }
        
        # Perform cross-validation with progress indication
        import time
        from sklearn.base import clone
        start_time = time.time()
        print(f"   ⏳ Starting {self.n_folds}-fold cross-validation...")
        
        # Initialize result containers
        cv_results = {
            'test_accuracy': [],
            'test_f1': [],
            'test_precision': [],
            'test_recall': [],
            'test_roc_auc': [],
            'train_accuracy': [],
            'train_f1': [],
            'train_precision': [],
            'train_recall': [],
            'train_roc_auc': []
        }
        
        # Manual cross-validation with per-fold progress
        for fold_idx, (train_idx, test_idx) in enumerate(self.cv.split(X, y), 1):
            fold_start_time = time.time()
            print(f"   📁 Processing fold {fold_idx}/{self.n_folds}...", end=" ", flush=True)
            
            # Split data
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Clone and train model
            fold_model = clone(model)
            fold_model.fit(X_train, y_train)
            
            # Predict
            y_train_pred = fold_model.predict(X_train)
            y_test_pred = fold_model.predict(X_test)
            y_train_proba = fold_model.predict_proba(X_train)[:, 1] if hasattr(fold_model, 'predict_proba') else y_train_pred
            y_test_proba = fold_model.predict_proba(X_test)[:, 1] if hasattr(fold_model, 'predict_proba') else y_test_pred
            
            # Calculate metrics for test set
            cv_results['test_accuracy'].append(accuracy_score(y_test, y_test_pred))
            cv_results['test_f1'].append(f1_score(y_test, y_test_pred))
            cv_results['test_precision'].append(precision_score(y_test, y_test_pred))
            cv_results['test_recall'].append(recall_score(y_test, y_test_pred))
            cv_results['test_roc_auc'].append(roc_auc_score(y_test, y_test_proba))
            
            # Calculate metrics for train set
            cv_results['train_accuracy'].append(accuracy_score(y_train, y_train_pred))
            cv_results['train_f1'].append(f1_score(y_train, y_train_pred))
            cv_results['train_precision'].append(precision_score(y_train, y_train_pred))
            cv_results['train_recall'].append(recall_score(y_train, y_train_pred))
            cv_results['train_roc_auc'].append(roc_auc_score(y_train, y_train_proba))
            
            fold_elapsed = time.time() - fold_start_time
            print(f"✅ {fold_elapsed:.1f}s (Acc: {cv_results['test_accuracy'][-1]:.3f})")
        
        # Convert lists to numpy arrays for consistency with cross_validate output
        for key in cv_results:
            cv_results[key] = np.array(cv_results[key])
        
        elapsed = time.time() - start_time
        print(f"   ✅ All folds completed in {elapsed/60:.1f} minutes")
        
        # Calculate statistics
        results = {
            'model_name': model_name,
            'n_folds': self.n_folds
        }
        
        for metric in scoring.keys():
            test_scores = cv_results[f'test_{metric}']
            train_scores = cv_results[f'train_{metric}']
            
            results.update({
                f'{metric}_mean': np.mean(test_scores),
                f'{metric}_std': np.std(test_scores),
                f'{metric}_min': np.min(test_scores),
                f'{metric}_max': np.max(test_scores),
                f'{metric}_train_mean': np.mean(train_scores),
                f'{metric}_train_std': np.std(train_scores),
                f'{metric}_scores': test_scores.tolist()
            })
            
            # Calculate 95% confidence interval
            confidence_interval = stats.t.interval(
                0.95, len(test_scores)-1,
                loc=np.mean(test_scores),
                scale=stats.sem(test_scores)
            )
            results.update({
                f'{metric}_ci_lower': confidence_interval[0],
                f'{metric}_ci_upper': confidence_interval[1]
            })
        
        self.results[model_name] = results
        
        print(f"✅ {model_name}: Accuracy {results['accuracy_mean']:.4f} ± {results['accuracy_std']:.4f}")
        
        return results
    
    def compare_models_statistical(self, model_results: List[Dict]) -> pd.DataFrame:
        """
        Perform statistical comparison between models
        
        Args:
            model_results: List of cross-validation results
            
        Returns:
            DataFrame with statistical comparison results
        """
        print("📊 Performing statistical model comparison...")
        
        comparison_results = []
        
        # Pairwise t-tests
        for i, model1 in enumerate(model_results):
            for j, model2 in enumerate(model_results):
                if i < j:  # Avoid duplicate comparisons
                    for metric in ['accuracy', 'f1', 'precision', 'recall', 'roc_auc']:
                        scores1 = model1[f'{metric}_scores']
                        scores2 = model2[f'{metric}_scores']
                        
                        # Paired t-test
                        t_stat, p_value = stats.ttest_rel(scores1, scores2)
                        
                        # Effect size (Cohen's d)
                        pooled_std = np.sqrt(((np.std(scores1)**2) + (np.std(scores2)**2)) / 2)
                        cohens_d = (np.mean(scores1) - np.mean(scores2)) / pooled_std
                        
                        comparison_results.append({
                            'model1': model1['model_name'],
                            'model2': model2['model_name'],
                            'metric': metric,
                            'mean_diff': np.mean(scores1) - np.mean(scores2),
                            't_statistic': t_stat,
                            'p_value': p_value,
                            'significant': p_value < 0.05,
                            'cohens_d': cohens_d,
                            'effect_size': self._interpret_effect_size(abs(cohens_d))
                        })
        
        return pd.DataFrame(comparison_results)
    
    def _interpret_effect_size(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size"""
        if cohens_d < 0.2:
            return 'negligible'
        elif cohens_d < 0.5:
            return 'small'
        elif cohens_d < 0.8:
            return 'medium'
        else:
            return 'large'
    
    def create_cv_summary_table(self, model_results: List[Dict]) -> pd.DataFrame:
        """
        Create summary table of cross-validation results
        
        Args:
            model_results: List of cross-validation results
            
        Returns:
            DataFrame with summary statistics
        """
        summary_data = []
        
        for result in model_results:
            summary_data.append({
                'Model': result['model_name'],
                'Accuracy': f"{result['accuracy_mean']:.4f} ± {result['accuracy_std']:.4f}",
                'F1-Score': f"{result['f1_mean']:.4f} ± {result['f1_std']:.4f}",
                'Precision': f"{result['precision_mean']:.4f} ± {result['precision_std']:.4f}",
                'Recall': f"{result['recall_mean']:.4f} ± {result['recall_std']:.4f}",
                'ROC-AUC': f"{result['roc_auc_mean']:.4f} ± {result['roc_auc_std']:.4f}",
                'Accuracy_CI': f"[{result['accuracy_ci_lower']:.4f}, {result['accuracy_ci_upper']:.4f}]"
            })
        
        df = pd.DataFrame(summary_data)
        
        # Sort by accuracy mean if we have results
        if not df.empty and 'Accuracy' in df.columns:
            df['_accuracy_mean'] = [float(acc.split(' ± ')[0]) for acc in df['Accuracy']]
            df = df.sort_values('_accuracy_mean', ascending=False).drop('_accuracy_mean', axis=1)
        
        return df
    
    def plot_cv_results(self, model_results: List[Dict], save_path: str = None) -> plt.Figure:
        """
        Create box plots of cross-validation results
        
        Args:
            model_results: List of cross-validation results
            save_path: Path to save the plot
            
        Returns:
            matplotlib Figure object
        """
        metrics = ['accuracy', 'f1', 'precision', 'recall', 'roc_auc']
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            # Prepare data for box plot
            data = []
            labels = []
            
            for result in model_results:
                data.append(result[f'{metric}_scores'])
                labels.append(result['model_name'])
            
            # Create box plot
            box_plot = ax.boxplot(data, labels=labels, patch_artist=True)
            
            # Customize colors
            colors = plt.cm.Set3(np.linspace(0, 1, len(data)))
            for patch, color in zip(box_plot['boxes'], colors):
                patch.set_facecolor(color)
            
            ax.set_title(f'{metric.replace("_", " ").title()} Distribution')
            ax.set_ylabel(metric.replace("_", " ").title())
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
        
        # Remove empty subplot
        fig.delaxes(axes[5])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 CV results plot saved to {save_path}")
        
        return fig
    
    def generate_latex_table(self, summary_df: pd.DataFrame) -> str:
        """
        Generate LaTeX table code for the summary results
        
        Args:
            summary_df: Summary DataFrame
            
        Returns:
            LaTeX table code as string
        """
        latex_code = """
\\begin{table}[htbp]
\\centering
\\caption{Cross-Validation Results Summary (5-Fold)}
\\label{tab:cv_results}
\\begin{tabular}{|l|c|c|c|c|c|}
\\hline
\\textbf{Model} & \\textbf{Accuracy} & \\textbf{F1-Score} & \\textbf{Precision} & \\textbf{Recall} & \\textbf{ROC-AUC} \\\\
\\hline
"""
        
        for _, row in summary_df.iterrows():
            latex_code += f"{row['Model']} & {row['Accuracy']} & {row['F1-Score']} & {row['Precision']} & {row['Recall']} & {row['ROC-AUC']} \\\\\n"
        
        latex_code += """\\hline
\\end{tabular}
\\end{table}
"""
        
        return latex_code
    
    def save_results(self, output_dir: str):
        """
        Save all cross-validation results
        
        Args:
            output_dir: Directory to save results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Determine dataset suffix from output path or context
        dataset_suffix = ""
        if "nsl" in str(output_path).lower():
            dataset_suffix = "_nsl"
        elif "cic" in str(output_path).lower():
            dataset_suffix = "_cic"
        
        # Save detailed results
        detailed_results = pd.DataFrame([result for result in self.results.values()])
        detailed_results.to_csv(output_path / f'cv_detailed_results{dataset_suffix}.csv', index=False)
        
        # Save summary table
        summary_df = self.create_cv_summary_table(list(self.results.values()))
        summary_df.to_csv(output_path / f'cv_summary_table{dataset_suffix}.csv', index=False)
        
        # Save LaTeX table
        latex_table = self.generate_latex_table(summary_df)
        with open(output_path / f'cv_summary_table{dataset_suffix}.tex', 'w') as f:
            f.write(latex_table)
        
        # Save statistical comparison
        if len(self.results) > 1:
            comparison_df = self.compare_models_statistical(list(self.results.values()))
            comparison_df.to_csv(output_path / f'statistical_comparison{dataset_suffix}.csv', index=False)
        
        print(f"📊 Cross-validation results saved to {output_path}")
    
    def save_incremental_result(self, model_result: Dict[str, Any], output_dir: str):
        """
        Save a single model's cross-validation result incrementally
        
        Args:
            model_result: Single model CV result dictionary
            output_dir: Directory to save results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Determine dataset suffix
        dataset_suffix = ""
        if "nsl" in str(output_path).lower():
            dataset_suffix = "_nsl"
        elif "cic" in str(output_path).lower():
            dataset_suffix = "_cic"
        
        # Incremental detailed results file
        detailed_file = output_path / f'cv_detailed_results{dataset_suffix}.csv'
        
        # Convert single result to DataFrame
        result_df = pd.DataFrame([model_result])
        
        # Append to existing file or create new one
        if detailed_file.exists():
            result_df.to_csv(detailed_file, mode='a', header=False, index=False)
            print(f"   📄 Appended {model_result['model_name']} to {detailed_file}")
        else:
            result_df.to_csv(detailed_file, mode='w', header=True, index=False)
            print(f"   📄 Created {detailed_file} with {model_result['model_name']}")
    
    def load_existing_results(self, output_dir: str) -> List[str]:
        """
        Load existing cross-validation results and return list of completed models
        
        Args:
            output_dir: Directory containing existing results
            
        Returns:
            List of model names that have already been processed
        """
        output_path = Path(output_dir)
        
        # Determine dataset suffix
        dataset_suffix = ""
        if "nsl" in str(output_path).lower():
            dataset_suffix = "_nsl"
        elif "cic" in str(output_path).lower():
            dataset_suffix = "_cic"
        
        detailed_file = output_path / f'cv_detailed_results{dataset_suffix}.csv'
        
        completed_models = []
        if detailed_file.exists():
            try:
                existing_df = pd.read_csv(detailed_file)
                completed_models = existing_df['model_name'].tolist()
                print(f"📋 Found {len(completed_models)} existing results: {completed_models}")
                
                # Load results back into framework
                for _, row in existing_df.iterrows():
                    model_name = row['model_name']
                    # Convert row back to result dictionary
                    result_dict = row.to_dict()
                    # Parse list columns back from string representation
                    for col in result_dict.keys():
                        if col.endswith('_scores') and isinstance(result_dict[col], str):
                            try:
                                # Convert string representation back to list
                                result_dict[col] = eval(result_dict[col])
                            except:
                                pass
                    self.results[model_name] = result_dict
            except Exception as e:
                print(f"⚠️ Error loading existing results: {e}")
        
        return completed_models


def run_cic_cross_validation():
    """
    Run cross-validation specifically for CIC-IDS-2017 models
    """
    print("🚀 CIC-IDS-2017 Cross-Validation Analysis")
    print("=" * 60)
    
    from src.utils import get_memory_adaptive_config
    config = get_memory_adaptive_config()
    
    # Load CIC data (same preprocessing as training)
    from src.preprocessing import CICIDSPreprocessor
    from pathlib import Path
    
    print("🔍 Loading CIC-IDS-2017 dataset...")
    preprocessor = CICIDSPreprocessor()
    
    # Load the same dataset that was used for training (based on memory config)
    try:
        use_full_dataset = config["use_full_dataset"]
        dataset_type = "full dataset" if use_full_dataset else "sample dataset"
        print(f"🔍 Loading CIC-IDS-2017 {dataset_type} (same as training)...")
        print(f"📊 Memory config: use_full_dataset={use_full_dataset}")
        
        cic_data = preprocessor.load_data(use_full_dataset=use_full_dataset)
        if cic_data is not None:
            X, y = preprocessor.fit_transform(cic_data)
            print(f"✅ Loaded CIC {dataset_type}: {X.shape}")
            print(f"📊 Class distribution: Normal={np.sum(y == 0)}, Attack={np.sum(y == 1)}")
        else:
            print(f"❌ CIC {dataset_type} loading failed, skipping CIC cross-validation")
            return None, []
    except Exception as e:
        print(f"❌ Error loading CIC data: {e}")
        print("🔄 Falling back to sample dataset...")
        try:
            cic_data = preprocessor.load_data(use_full_dataset=False)  # Fallback to sample
            if cic_data is not None:
                X, y = preprocessor.fit_transform(cic_data)
                print(f"⚠️ Using CIC sample dataset: {X.shape}")
                print("⚠️ WARNING: This may not match training data distribution!")
            else:
                print("❌ Both full and sample CIC data loading failed")
                return None, []
        except Exception as e2:
            print(f"❌ Error loading CIC sample data: {e2}")
            return None, []
    
    # Initialize cross-validation framework
    cv_framework = CrossValidationFramework(n_folds=3)  # Use 3 folds for efficiency
    
    # Setup output directory
    output_dir = 'data/results/cross_validation/cic'
    
    # Check for existing results and resume if possible
    print("🔍 Checking for existing cross-validation results...")
    completed_models = cv_framework.load_existing_results(output_dir)
    
    # Load CIC models
    model_paths = {}
    
    # Baseline CIC models
    for model in ['random_forest', 'logistic_regression', 'decision_tree', 'naive_bayes', 'knn', 'svm_linear']:
        model_path = Path(f'data/models/cic_baseline/{model}_cic.joblib')
        if model_path.exists():
            model_paths[f"{model}_cic"] = str(model_path)
    
    # Advanced CIC models  
    for model in ['xgboost', 'lightgbm', 'gradient_boosting', 'extra_trees', 'mlp', 'voting_classifier']:
        model_path = Path(f'data/models/cic_advanced/{model}_cic.joblib')
        if model_path.exists():
            model_paths[f"{model}_cic"] = str(model_path)
    
    print(f"📊 Found {len(model_paths)} CIC models: {list(model_paths.keys())}")
    
    if len(model_paths) == 0:
        print("❌ No CIC models found for cross-validation!")
        return None, []
    
    # Filter out already completed models
    remaining_models = {name: path for name, path in model_paths.items() 
                       if name not in completed_models}
    
    if completed_models:
        print(f"✅ Resuming: {len(completed_models)} models already done, {len(remaining_models)} remaining")
    else:
        print(f"🚀 Starting fresh: {len(remaining_models)} models to process")
    
    # Estimate time
    if remaining_models:
        print(f"⏱️  Estimated time: {len(remaining_models) * 15-30} minutes")
        print("💾 Results will be saved after each model to prevent data loss")
    
    # Perform cross-validation with incremental saving
    cv_results = []
    total_models = len(model_paths)
    completed_count = len(completed_models)
    
    for model_name, model_path in remaining_models.items():
        completed_count += 1
        print(f"\n📊 [{completed_count}/{total_models}] Processing {model_name}...")
        
        try:
            import joblib
            
            model = joblib.load(model_path)
            result = cv_framework.evaluate_model_cv(model, X, y, model_name)
            
            # Save immediately after each model
            cv_framework.save_incremental_result(result, output_dir)
            cv_results.append(result)
            
            print(f"   ✅ {model_name} complete!")
            
        except Exception as e:
            print(f"   ❌ Error with {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Collect all results (existing + new)
    all_results = list(cv_framework.results.values())
    
    # Save final comprehensive results
    if all_results:
        cv_framework.save_results(output_dir)
    
    return cv_framework, all_results


def run_full_cross_validation():
    """
    Run complete cross-validation pipeline on all models (NSL-KDD only)
    """
    print("🚀 Running Complete Cross-Validation Pipeline")
    print("=" * 60)
    
    # Suppress LightGBM verbose output
    import os
    os.environ['LIGHTGBM_VERBOSITY'] = '-1'
    
    # Import required modules
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))

    from src.preprocessing import NSLKDDAnalyzer, NSLKDDPreprocessor
    from src.utils import get_memory_adaptive_config, MemoryMonitor
    
    # Get memory configuration
    config = get_memory_adaptive_config()
    
    # Load data with memory monitoring
    with MemoryMonitor("Data Loading"):
        print("📁 Loading NSL-KDD data...")
        analyzer = NSLKDDAnalyzer()
        train_data = analyzer.load_data("KDDTrain+.txt")
        
        if train_data is None:
            print("❌ Failed to load training data")
            return
    
    # Preprocess data with memory monitoring
    with MemoryMonitor("Data Preprocessing"):
        print("🔄 Preprocessing data...")
        balance_method = 'smote' if config["use_full_dataset"] else 'undersample'
        print(f"   Using balance method: {balance_method}")
        
        preprocessor = NSLKDDPreprocessor(balance_method=balance_method)
        X_train, X_val, y_train, y_val = preprocessor.fit_transform(train_data)
        
        # Combine training and validation for cross-validation
        X = np.vstack([X_train, X_val])
        y = np.hstack([y_train, y_val])
        
        print(f"✅ Data prepared: {X.shape}")
        print(f"📊 Memory mode: {'Full dataset' if config['use_full_dataset'] else 'Optimized'}")
    
    # Initialize cross-validation framework
    # Reduce folds if memory constrained
    n_folds = 5 if config["use_full_dataset"] else 3
    print(f"🔄 Using {n_folds}-fold cross-validation")
    cv_framework = CrossValidationFramework(n_folds=n_folds)
    
    # Load all trained models
    print("📂 Loading trained models...")
    
    # Check for existing model files and build paths dynamically
    from pathlib import Path
    model_paths = {}
    
    # Baseline models
    baseline_models = ['random_forest', 'logistic_regression', 'decision_tree', 'naive_bayes', 'knn', 'svm_linear']
    for model in baseline_models:
        model_path = Path(f'data/models/baseline/{model}_nsl.joblib')
        if model_path.exists():
            model_paths[model] = str(model_path)
    
    # Advanced models  
    advanced_models = ['xgboost', 'lightgbm', 'gradient_boosting', 'extra_trees', 'mlp', 'voting_classifier']
    for model in advanced_models:
        model_path = Path(f'data/models/advanced/{model}_nsl.joblib')
        if model_path.exists():
            model_paths[model] = str(model_path)
    
    print(f"📊 Found {len(model_paths)} trained models: {list(model_paths.keys())}")
    
    # Perform cross-validation for each model
    cv_results = []
    total_models = len(model_paths)
    
    print(f"\n🔄 Starting cross-validation for {total_models} models...")
    print("⏳ Estimated time: 10-15 minutes total")
    
    for idx, (model_name, model_path) in enumerate(model_paths.items(), 1):
        print(f"\n📊 [{idx}/{total_models}] Processing {model_name}...")
        try:
            model = joblib.load(model_path)
            result = cv_framework.evaluate_model_cv(model, X, y, model_name)
            cv_results.append(result)
            print(f"   ✅ {model_name} complete!")
        except FileNotFoundError:
            print(f"   ⚠️ Model file not found: {model_path}")
        except Exception as e:
            print(f"   ❌ Error loading {model_name}: {e}")
            import traceback
            print(f"   📝 Details: {traceback.format_exc()}")
    
    print(f"\n✅ Cross-validation completed for {len(cv_results)} models!")
    
    # Check if we have any results before proceeding
    if len(cv_results) == 0:
        print("\n❌ No models found for cross-validation!")
        print("🔧 Please ensure models are trained first by running:")
        print("   python experiments/02_baseline_training.py")  
        print("   python experiments/03_advanced_training.py")
        return None, []
    
    # Statistical comparison
    if len(cv_results) > 1:
        print("\n📊 Statistical comparison...")
        comparison_df = cv_framework.compare_models_statistical(cv_results)
        print(f"✅ Found {len(comparison_df)} pairwise comparisons")
    
    # Create summary table
    try:
        summary_table = cv_framework.create_cv_summary_table(cv_results)
        print("\n📋 Cross-Validation Summary:")
        print(summary_table.to_string(index=False))
    except Exception as e:
        print(f"\n❌ Error creating summary table: {e}")
        # Create a basic summary as fallback
        print("\n📋 Basic Cross-Validation Results:")
        for result in cv_results:
            print(f"   {result['model_name']}: Accuracy={result['accuracy_mean']:.4f}±{result['accuracy_std']:.4f}")
        return cv_framework, cv_results
    
    # Create visualizations
    print("\n📊 Creating visualizations...")
    fig = cv_framework.plot_cv_results(cv_results, 'data/results/cv_results_boxplot.png')
    
    # Save all results
    cv_framework.save_results('data/results/cross_validation')
    
    print("\n✅ Cross-validation pipeline complete!")
    return cv_framework, cv_results


if __name__ == "__main__":
    run_full_cross_validation()