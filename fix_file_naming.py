#!/usr/bin/env python3
"""
COMPREHENSIVE FILE NAMING STANDARDIZATION
Fixes ALL file saving operations to use dataset-specific names
"""

import os
from pathlib import Path
import re

def find_and_fix_all_files():
    """Find and fix ALL file naming issues across the entire codebase"""
    
    fixes_applied = []
    
    # 1. CRITICAL: Fix paper_figures.py - save_path generation with dataset prefixes
    paper_figures_fixes = [
        # Performance comparison figure should include dataset info
        {
            'file': 'src/visualization/paper_figures.py',
            'find': 'save_path = self.output_dir / "model_performance_comparison.png"',
            'replace': 'dataset_prefix = "nsl_cic" if self._has_both_datasets() else ("nsl" if self._has_nsl_data() else "cic")\n            save_path = self.output_dir / f"{dataset_prefix}_model_performance_comparison.png"'
        },
        # Attack distribution analysis
        {
            'file': 'src/visualization/paper_figures.py', 
            'find': 'save_path = self.output_dir / "attack_distribution_analysis.png"',
            'replace': 'save_path = self.output_dir / "nsl_attack_distribution_analysis.png"'
        },
        # CIC attack distribution (already has good naming)
        {
            'file': 'src/visualization/paper_figures.py',
            'find': 'save_path = self.output_dir / "cic_attack_distribution_analysis.png"', 
            'replace': 'save_path = self.output_dir / "cic_attack_distribution_analysis.png"'  # Already correct
        },
        # Performance summary table with dataset prefix
        {
            'file': 'src/visualization/paper_figures.py',
            'find': 'save_path = self.output_dir / "performance_summary_table"',
            'replace': 'dataset_prefix = "nsl_cic" if self._has_both_datasets() else ("nsl" if self._has_nsl_data() else "cic")\n            save_path = self.output_dir / f"{dataset_prefix}_performance_summary_table"'
        }
    ]
    
    # 2. Fix enhanced_evaluation.py - ALL figure saving operations
    evaluation_fixes = [
        # Confusion matrix with dataset prefix
        {
            'file': 'src/evaluation/enhanced_evaluation.py',
            'find': 'filename = f"{model_name.lower().replace(\' \', \'_\')}_{dataset_name}_confusion_matrix.png"',
            'replace': 'filename = f"{model_name.lower().replace(\' \', \'_\')}_{dataset_name.lower()}_confusion_matrix.png"'
        },
        # ROC curve with dataset prefix
        {
            'file': 'src/evaluation/enhanced_evaluation.py',
            'find': 'filename = f"{model_name.lower().replace(\' \', \'_\')}_{dataset_name}_roc_curve.png"',
            'replace': 'filename = f"{model_name.lower().replace(\' \', \'_\')}_{dataset_name.lower()}_roc_curve.png"'
        },
        # Precision-Recall curve with dataset prefix
        {
            'file': 'src/evaluation/enhanced_evaluation.py',
            'find': 'filename = f"{model_name.lower().replace(\' \', \'_\')}_{dataset_name}_precision_recall_curve.png"',
            'replace': 'filename = f"{model_name.lower().replace(\' \', \'_\')}_{dataset_name.lower()}_precision_recall_curve.png"'
        },
        # Feature importance with dataset prefix
        {
            'file': 'src/evaluation/enhanced_evaluation.py',
            'find': 'filename = f"{model_name.lower().replace(\' \', \'_\')}_feature_importance.png"',
            'replace': 'dataset_suffix = f"_{dataset_name.lower()}" if hasattr(self, "current_dataset") and self.current_dataset else ""\n        filename = f"{model_name.lower().replace(\' \', \'_\')}{dataset_suffix}_feature_importance.png"'
        },
        # Learning curves with dataset prefix
        {
            'file': 'src/evaluation/enhanced_evaluation.py',
            'find': 'filename = f"{model_name.lower().replace(\' \', \'_\')}_learning_curve.png"',
            'replace': 'dataset_suffix = f"_{dataset_name.lower()}" if hasattr(self, "current_dataset") and self.current_dataset else ""\n        filename = f"{model_name.lower().replace(\' \', \'_\')}{dataset_suffix}_learning_curve.png"'
        },
        # Timing analysis with dataset prefix
        {
            'file': 'src/evaluation/enhanced_evaluation.py',
            'find': 'filename = f"{save_prefix}_timing_analysis.png"',
            'replace': 'dataset_suffix = f"_{self.current_dataset.lower()}" if hasattr(self, "current_dataset") and self.current_dataset else ""\n        filename = f"{save_prefix}{dataset_suffix}_timing_analysis.png"'
        }
    ]
    
    # 3. Fix cross_validation.py - ALL CSV and TEX outputs
    cv_fixes = [
        # CV detailed results with dataset prefix
        {
            'file': 'src/metrics/cross_validation.py',
            'find': "detailed_results.to_csv(output_path / 'cv_detailed_results.csv', index=False)",
            'replace': "dataset_suffix = f\"_{self.dataset_name.lower()}\" if hasattr(self, 'dataset_name') and self.dataset_name else \"\"\n        detailed_results.to_csv(output_path / f'cv_detailed_results{dataset_suffix}.csv', index=False)"
        },
        # CV summary table with dataset prefix
        {
            'file': 'src/metrics/cross_validation.py',
            'find': "summary_df.to_csv(output_path / 'cv_summary_table.csv', index=False)",
            'replace': "dataset_suffix = f\"_{self.dataset_name.lower()}\" if hasattr(self, 'dataset_name') and self.dataset_name else \"\"\n        summary_df.to_csv(output_path / f'cv_summary_table{dataset_suffix}.csv', index=False)"
        },
        # CV TEX table with dataset prefix  
        {
            'file': 'src/metrics/cross_validation.py',
            'find': "with open(output_path / 'cv_summary_table.tex', 'w') as f:",
            'replace': "dataset_suffix = f\"_{self.dataset_name.lower()}\" if hasattr(self, 'dataset_name') and self.dataset_name else \"\"\n        with open(output_path / f'cv_summary_table{dataset_suffix}.tex', 'w') as f:"
        },
        # Statistical comparison with dataset prefix
        {
            'file': 'src/metrics/cross_validation.py',
            'find': "comparison_df.to_csv(output_path / 'statistical_comparison.csv', index=False)",
            'replace': "dataset_suffix = f\"_{self.dataset_name.lower()}\" if hasattr(self, 'dataset_name') and self.dataset_name else \"\"\n            comparison_df.to_csv(output_path / f'statistical_comparison{dataset_suffix}.csv', index=False)"
        }
    ]
    
    # 4. Fix experiment scripts that save to generic names
    experiment_fixes = [
        # 02_baseline_training.py - fix legacy save
        {
            'file': 'experiments/02_baseline_training.py',
            'find': 'val_results.to_csv(main_results_dir / "baseline_results.csv", index=False)',
            'replace': '# Legacy save removed - using dataset-specific naming from save_models() instead'
        }
    ]
    
    print("🔧 COMPREHENSIVE FILE NAMING FIXES")
    print("="*60)
    
    all_fixes = paper_figures_fixes + evaluation_fixes + cv_fixes + experiment_fixes
    
    print(f"📋 Total fixes to apply: {len(all_fixes)}")
    
    for i, fix in enumerate(all_fixes, 1):
        print(f"{i:2d}. {fix['file']}")
        print(f"    Find: {fix['find'][:50]}...")
        print(f"    Replace: {fix['replace'][:50]}...")
    
    return all_fixes

if __name__ == "__main__":
    fixes = find_and_fix_all_files()
    
    print(f"\n🎯 Ready to apply {len(fixes)} fixes for dataset-specific file naming")
    print("These fixes will ensure NO conflicts between NSL-KDD and CIC-IDS-2017 files")