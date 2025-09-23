"""
Script to fix redundant results storage and enhance scientific value
"""
import os
import sys
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))

from evaluation.enhanced_evaluation import EnhancedEvaluator, create_results_summary_table
from preprocessing.nsl_kdd_preprocessor import NSLKDDPreprocessor


def consolidate_redundant_results():
    """
    Fix redundant results storage by consolidating everything to data/results/
    """
    print("🔧 FIXING REDUNDANT RESULTS STORAGE")
    print("=" * 60)
    
    # Central results directory
    results_dir = Path("data/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Source directories with redundant results
    source_dirs = [
        Path("data/models"),
        Path("data/models/baseline"),
        Path("data/models/advanced"),
        Path("data/models/cic_baseline"),
        Path("data/models/cic_advanced")
    ]
    
    consolidated_files = []
    
    print("\n📁 Consolidating results from model directories...")
    
    for source_dir in source_dirs:
        if source_dir.exists():
            print(f"\n  📂 Processing: {source_dir}")
            
            # Find CSV files
            csv_files = list(source_dir.glob("*.csv"))
            for csv_file in csv_files:
                target_file = results_dir / csv_file.name
                
                # Check if file already exists in results
                if target_file.exists():
                    print(f"    ⚠️ Already exists: {csv_file.name}")
                    # Compare and keep the more recent one
                    if csv_file.stat().st_mtime > target_file.stat().st_mtime:
                        shutil.copy2(csv_file, target_file)
                        print(f"    ✅ Updated with newer version")
                else:
                    shutil.copy2(csv_file, target_file)
                    print(f"    ✅ Copied: {csv_file.name}")
                
                consolidated_files.append(csv_file.name)
                
                # Remove original to avoid confusion
                try:
                    csv_file.unlink()
                    print(f"    🗑️ Removed original: {csv_file}")
                except Exception as e:
                    print(f"    ⚠️ Could not remove {csv_file}: {e}")
    
    # Create a mapping file to track what was moved
    mapping_file = results_dir / "consolidation_mapping.txt"
    with open(mapping_file, 'w') as f:
        f.write("RESULTS CONSOLIDATION MAPPING\n")
        f.write("=" * 40 + "\n\n")
        f.write("All results have been centralized to data/results/\n\n")
        f.write("Consolidated files:\n")
        for file in sorted(set(consolidated_files)):
            f.write(f"  - {file}\n")
    
    print(f"\n✅ Consolidation complete! Mapping saved to: {mapping_file}")
    print(f"📊 Total files consolidated: {len(set(consolidated_files))}")
    
    return results_dir


def enhance_scientific_value():
    """
    Add comprehensive scientific value enhancements
    """
    print("\n🔬 ENHANCING SCIENTIFIC VALUE")
    print("=" * 60)
    
    # Initialize enhanced evaluator
    evaluator = EnhancedEvaluator()
    
    # Load test data for analysis
    print("\n📊 Loading test data...")
    try:
        from preprocessing.data_analyzer import NSLKDDAnalyzer
        
        # Load NSL-KDD data
        analyzer = NSLKDDAnalyzer()
        train_data = analyzer.load_data("KDDTrain+.txt")
        test_data = analyzer.load_data("KDDTest+.txt")
        
        if train_data is None or test_data is None:
            print("❌ Could not load NSL-KDD data files")
            return None
        
        # Initialize preprocessor
        preprocessor = NSLKDDPreprocessor()
        
        # Fit and transform training data
        X_train, X_val, y_train, y_val = preprocessor.fit_transform(train_data)
        
        # Transform test data  
        X_test, y_test = preprocessor.transform(test_data)
        
        feature_names = preprocessor.feature_names
        
        print(f"✅ Loaded test data: {X_test.shape[0]} samples, {X_test.shape[1]} features")
        print(f"✅ Training data: {X_train.shape[0]} samples")
        
    except Exception as e:
        print(f"❌ Could not load test data: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Load trained models for analysis
    model_dirs = [
        Path("data/models"),
        Path("data/models/baseline"), 
        Path("data/models/advanced")
    ]
    
    trained_models = {}
    
    print("\n🤖 Loading trained models...")
    for model_dir in model_dirs:
        if model_dir.exists():
            for model_file in model_dir.glob("*.joblib"):
                if "preprocessor" not in model_file.name:
                    try:
                        model_name = model_file.stem.replace('_', ' ').title()
                        model = joblib.load(model_file)
                        trained_models[model_name] = model
                        print(f"  ✅ Loaded: {model_name}")
                    except Exception as e:
                        print(f"  ❌ Failed to load {model_file}: {e}")
    
    if not trained_models:
        print("⚠️ No trained models found. Train models first using experiments 01-02.")
        return None
    
    print(f"\n📈 Analyzing {len(trained_models)} models...")
    
    # Comprehensive analysis for each model
    all_model_results = []
    
    for model_name, model in trained_models.items():
        print(f"\n🔍 Analyzing: {model_name}")
        
        try:
            # Comprehensive analysis
            analysis_results = evaluator.comprehensive_model_analysis(
                model=model,
                X_test=X_test,
                y_test=y_test,
                X_train=X_train,
                y_train=y_train,
                model_name=model_name,
                dataset_name="NSL-KDD",
                feature_names=feature_names
            )
            
            # Get predictions and probabilities for comparison
            y_pred = model.predict(X_test)
            y_proba = None
            
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, 'decision_function'):
                y_proba = model.decision_function(X_test)
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
            
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            
            roc_auc = None
            if y_proba is not None:
                roc_auc = roc_auc_score(y_test, y_proba)
            
            all_model_results.append({
                'model_name': model_name,
                'y_true': y_test,
                'y_pred': y_pred,
                'y_proba': y_proba,
                'accuracy': accuracy,
                'f1_score': f1,
                'precision': precision,
                'recall': recall,
                'roc_auc': roc_auc,
                'analysis_paths': analysis_results
            })
            
            print(f"  ✅ Analysis complete for {model_name}")
            
        except Exception as e:
            print(f"  ❌ Error analyzing {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Generate top models comparison
    if len(all_model_results) >= 3:
        print(f"\n🏆 Generating top 3 models comparison...")
        try:
            comparison_path = evaluator.generate_top_models_comparison(
                all_model_results, top_n=3, metric='f1_score'
            )
            print(f"✅ Top models comparison saved: {comparison_path}")
        except Exception as e:
            print(f"❌ Error generating comparison: {e}")
    
    # Create comprehensive results summary
    print(f"\n📋 Creating comprehensive results summary...")
    try:
        summary_path = create_results_summary_table(evaluator)
        print(f"✅ Summary table created: {summary_path}")
    except Exception as e:
        print(f"❌ Error creating summary: {e}")
    
    # Create enhanced results report
    print(f"\n📄 Creating enhanced results report...")
    report_path = create_enhanced_results_report(evaluator, all_model_results)
    
    return evaluator, all_model_results


def create_enhanced_results_report(evaluator: EnhancedEvaluator, 
                                 model_results: List[Dict]) -> str:
    """
    Create comprehensive results report
    """
    report_path = evaluator.output_dir / "enhanced_results_report.md"
    
    with open(report_path, 'w') as f:
        f.write("# Enhanced Scientific Analysis Report\n\n")
        f.write("Generated automatically by enhanced evaluation framework.\n\n")
        
        f.write("## Overview\n\n")
        f.write(f"- **Total models analyzed**: {len(model_results)}\n")
        f.write(f"- **Analysis timestamp**: {pd.Timestamp.now()}\n")
        f.write(f"- **Output directory**: {evaluator.output_dir}\n\n")
        
        f.write("## Model Performance Summary\n\n")
        f.write("| Model | Accuracy | F1-Score | Precision | Recall | ROC-AUC |\n")
        f.write("|-------|----------|----------|-----------|--------|---------|\n")
        
        # Sort by F1-score
        sorted_results = sorted(model_results, key=lambda x: x['f1_score'], reverse=True)
        
        for result in sorted_results:
            roc_auc_str = f"{result['roc_auc']:.4f}" if result['roc_auc'] else "N/A"
            f.write(f"| {result['model_name']} | {result['accuracy']:.4f} | "
                   f"{result['f1_score']:.4f} | {result['precision']:.4f} | "
                   f"{result['recall']:.4f} | {roc_auc_str} |\n")
        
        f.write("\n## Generated Visualizations\n\n")
        
        # List all generated files
        viz_dirs = [
            ('Confusion Matrices', evaluator.subdirs['confusion_matrices']),
            ('ROC Curves', evaluator.subdirs['roc_curves']),
            ('Feature Importance', evaluator.subdirs['feature_importance']),
            ('Learning Curves', evaluator.subdirs['learning_curves']),
            ('Model Analysis', evaluator.subdirs['model_analysis'])
        ]
        
        for viz_name, viz_dir in viz_dirs:
            f.write(f"### {viz_name}\n\n")
            if viz_dir.exists():
                files = list(viz_dir.glob("*.png"))
                if files:
                    for file in sorted(files):
                        f.write(f"- `{file.name}`\n")
                else:
                    f.write("- No files generated\n")
            f.write("\n")
        
        f.write("## Key Insights\n\n")
        
        if sorted_results:
            best_model = sorted_results[0]
            worst_model = sorted_results[-1]
            
            f.write(f"- **Best performing model**: {best_model['model_name']} "
                   f"(F1: {best_model['f1_score']:.4f})\n")
            f.write(f"- **Worst performing model**: {worst_model['model_name']} "
                   f"(F1: {worst_model['f1_score']:.4f})\n")
            
            # Performance gap
            gap = best_model['f1_score'] - worst_model['f1_score']
            f.write(f"- **Performance gap**: {gap:.4f} F1-score difference\n")
            
            # Models with feature importance
            fi_models = [r['model_name'] for r in model_results if 'feature_importance' in r.get('analysis_paths', {})]
            f.write(f"- **Models with feature importance**: {len(fi_models)} "
                   f"({', '.join(fi_models)})\n")
        
        f.write("\n## File Organization\n\n")
        f.write("All results are now centrally organized in `data/results/` with the following structure:\n\n")
        f.write("```\n")
        f.write("data/results/\n")
        f.write("├── confusion_matrices/     # Confusion matrix plots\n")
        f.write("├── roc_curves/            # ROC curve plots\n") 
        f.write("├── feature_importance/    # Feature importance plots and data\n")
        f.write("├── learning_curves/       # Learning curve analysis\n")
        f.write("├── model_analysis/        # Comparative analysis\n")
        f.write("├── tables/                # Summary tables (CSV and LaTeX)\n")
        f.write("└── *.csv                  # Individual result files\n")
        f.write("```\n\n")
        
        f.write("## Next Steps\n\n")
        f.write("1. Review confusion matrices for misclassification patterns\n")
        f.write("2. Analyze ROC curves for model discrimination ability\n")
        f.write("3. Examine feature importance for interpretability\n")
        f.write("4. Check learning curves for overfitting/underfitting\n")
        f.write("5. Use generated LaTeX tables in your paper\n")
    
    print(f"📄 Enhanced results report saved: {report_path}")
    return str(report_path)


def main():
    """
    Main function to fix redundant storage and enhance scientific value
    """
    print("🚀 REPOSITORY ENHANCEMENT PIPELINE")
    print("=" * 60)
    print("Fixing redundant results storage and adding scientific value...")
    
    try:
        # Step 1: Fix redundant results storage
        results_dir = consolidate_redundant_results()
        
        # Step 2: Enhance scientific value
        evaluator, model_results = enhance_scientific_value()
        
        if evaluator and model_results:
            print(f"\n🎉 ENHANCEMENT COMPLETE!")
            print("=" * 60)
            print(f"✅ Fixed redundant results storage")
            print(f"✅ Generated {len(model_results)} model analyses")
            print(f"✅ Created comprehensive visualizations")
            print(f"✅ Centralized all results in: {results_dir}")
            
            print(f"\n📊 GENERATED OUTPUTS:")
            print(f"  🎯 Confusion matrices: {len(list(evaluator.subdirs['confusion_matrices'].glob('*.png')))}")
            print(f"  📈 ROC curves: {len(list(evaluator.subdirs['roc_curves'].glob('*.png')))}")
            print(f"  🔍 Feature importance: {len(list(evaluator.subdirs['feature_importance'].glob('*.png')))}")
            print(f"  📚 Learning curves: {len(list(evaluator.subdirs['learning_curves'].glob('*.png')))}")
            print(f"  📋 Summary tables: {len(list(evaluator.subdirs['tables'].glob('*')))}")
            
            print(f"\n🎓 READY FOR PAPER SUBMISSION!")
            print(f"Check the enhanced_results_report.md for detailed insights.")
            
        else:
            print(f"\n⚠️ Enhancement partially completed")
            print(f"Results consolidation: ✅")
            print(f"Scientific enhancements: ❌")
            
    except Exception as e:
        print(f"\n❌ Error during enhancement: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)