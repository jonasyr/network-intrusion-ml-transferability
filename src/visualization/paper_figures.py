# src/visualization/paper_figures.py
"""
Publication-ready figures and tables for the research paper
High-quality visualizations for academic submission
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
from typing import Dict, List, Tuple, Optional
import warnings
from scipy import stats
from scipy.stats import pearsonr
import json

from src.preprocessing import CICIDSPreprocessor

warnings.filterwarnings('ignore')

# Set publication-ready style following scientific standards
plt.style.use('default')
sns.set_style("whitegrid", {'grid.linestyle': '-', 'grid.linewidth': 0.5, 'grid.alpha': 0.5})

# Publication-quality matplotlib settings (IEEE/ACM standards)
plt.rcParams.update({
    # Typography - Arial/Sans-Serif fonts as required
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica', 'sans-serif'],
    'mathtext.fontset': 'dejavusans',  # Consistent math fonts
    
    # Font sizes (scaled for 2-column journal format)
    'font.size': 9,           # Base font size
    'axes.titlesize': 10,     # Subplot titles
    'axes.labelsize': 9,      # Axis labels
    'xtick.labelsize': 8,     # X-axis tick labels
    'ytick.labelsize': 8,     # Y-axis tick labels
    'legend.fontsize': 8,     # Legend
    'figure.titlesize': 11,   # Main title
    
    # Layout and quality
    'figure.dpi': 300,        # High resolution
    'savefig.dpi': 300,       # Save quality
    'savefig.format': 'pdf',  # Vector format for journals
    'savefig.bbox': 'tight',  # Tight bounding box
    'figure.constrained_layout.use': True,  # Better layout
    
    # Lines and markers
    'lines.linewidth': 1.0,
    'lines.markersize': 4,
    'patch.linewidth': 0.5,
    
    # Grid and spines
    'axes.grid': True,
    'axes.grid.axis': 'both',
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.8
})

class PaperFigureGenerator:
    """
    Generate high-quality figures for academic paper
    """
    
    def __init__(self, output_dir: str = "data/results/paper_figures"):
        """
        Initialize figure generator
        
        Args:
            output_dir: Directory to save figures
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Scientific color palette (colorblind-friendly, grayscale-safe)
        # Based on Paul Tol's qualitative color schemes for scientific visualization
        self.colors = {
            'primary': '#1f77b4',      # Blue - primary data
            'secondary': '#ff7f0e',    # Orange - secondary data
            'accent': '#2ca02c',       # Green - positive results
            'warning': '#d62728',      # Red - negative/warning
            'neutral': '#7f7f7f',      # Gray - neutral
            'purple': '#9467bd',       # Purple - alternative
            'brown': '#8c564b',        # Brown - additional
            'pink': '#e377c2'          # Pink - supplementary
        }
        
        # Qualitative palette for categorical data (up to 8 categories)
        self.qual_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
                           '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        
        # Sequential palette for continuous data
        self.seq_colors = sns.color_palette('viridis', n_colors=256)
        
        # Model categories for grouping
        self.model_categories = {
            'Traditional ML': ['random_forest', 'decision_tree', 'logistic_regression', 'naive_bayes', 'knn'],
            'Ensemble Methods': ['xgboost', 'lightgbm', 'gradient_boosting', 'extra_trees', 'voting_classifier'],
            'Neural Networks': ['mlp']
        }
    
    def _has_nsl_data(self) -> bool:
        """Check if NSL-KDD data/results are available"""
        nsl_paths = [
            "data/results/nsl_baseline_results.csv",
            "data/results/nsl_advanced_results.csv",
            "data/results/nsl/baseline_results.csv",
            "data/results/baseline_results.csv"  # Legacy that might be NSL
        ]
        return any(Path(path).exists() for path in nsl_paths)
    
    def _has_cic_data(self) -> bool:
        """Check if CIC-IDS-2017 data/results are available"""
        cic_paths = [
            "data/results/cic_baseline_results.csv", 
            "data/results/cic_advanced_results.csv",
            "data/results/cic/baseline_results.csv"
        ]
        return any(Path(path).exists() for path in cic_paths)
    
    def _has_both_datasets(self) -> bool:
        """Check if both datasets are available"""
        return self._has_nsl_data() and self._has_cic_data()
    
    def load_dataset_results(self, dataset: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load baseline and advanced model results for a specific dataset
        
        Args:
            dataset: 'nsl' or 'cic'
            
        Returns:
            Tuple of (baseline_results, advanced_results) DataFrames
        """
        dataset_lower = dataset.lower()
        
        # Dataset-specific paths
        baseline_paths = [
            f"data/results/{dataset_lower}_baseline_results.csv",
            f"data/results/{dataset_lower}/baseline_results.csv"
        ]
        advanced_paths = [
            f"data/results/{dataset_lower}_advanced_results.csv",
            f"data/results/{dataset_lower}/advanced_results.csv"
        ]
        
        # Load baseline results
        baseline_results = None
        for path in baseline_paths:
            try:
                baseline_results = pd.read_csv(path)
                print(f"📊 Loaded {dataset.upper()} baseline results from: {path}")
                break
            except FileNotFoundError:
                continue
                
        if baseline_results is None:
            print(f"⚠️ No {dataset.upper()} baseline results found")
            baseline_results = pd.DataFrame()
        
        # Load advanced results
        advanced_results = None
        for path in advanced_paths:
            try:
                advanced_results = pd.read_csv(path)
                print(f"📊 Loaded {dataset.upper()} advanced results from: {path}")
                break
            except FileNotFoundError:
                continue
                
        if advanced_results is None:
            print(f"⚠️ No {dataset.upper()} advanced results found")
            advanced_results = pd.DataFrame()
        
        # Add model category and dataset info
        def categorize_model(model_name):
            for category, models in self.model_categories.items():
                if model_name in models:
                    return category
            return 'Other'
        
        if not baseline_results.empty:
            baseline_results['category'] = baseline_results['model_name'].apply(categorize_model)
            baseline_results['dataset'] = dataset.upper()
            
        if not advanced_results.empty:
            advanced_results['category'] = advanced_results['model_name'].apply(categorize_model)
            advanced_results['dataset'] = dataset.upper()
        
        return baseline_results, advanced_results
    
    def load_all_results(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load all baseline and advanced model results from both datasets
        
        Returns:
            Tuple of (combined_baseline_results, combined_advanced_results) DataFrames
        """
        all_baseline = []
        all_advanced = []
        
        # Load NSL-KDD results
        if self._has_nsl_data():
            nsl_baseline, nsl_advanced = self.load_dataset_results('nsl')
            if not nsl_baseline.empty:
                all_baseline.append(nsl_baseline)
            if not nsl_advanced.empty:
                all_advanced.append(nsl_advanced)
        
        # Load CIC-IDS-2017 results
        if self._has_cic_data():
            cic_baseline, cic_advanced = self.load_dataset_results('cic')
            if not cic_baseline.empty:
                all_baseline.append(cic_baseline)
            if not cic_advanced.empty:
                all_advanced.append(cic_advanced)
        
        # Combine results
        combined_baseline = pd.concat(all_baseline, ignore_index=True) if all_baseline else pd.DataFrame()
        combined_advanced = pd.concat(all_advanced, ignore_index=True) if all_advanced else pd.DataFrame()
        
        return combined_baseline, combined_advanced
    
    def load_cross_validation_results(self) -> Dict[str, pd.DataFrame]:
        """
        Load cross-validation results for both datasets
        
        Returns:
            Dictionary with dataset names as keys and CV results as values
        """
        cv_results = {}
        
        # NSL-KDD cross-validation
        nsl_cv_path = "data/results/cross_validation/cv_summary_table.csv"
        if Path(nsl_cv_path).exists():
            try:
                cv_results['NSL-KDD'] = pd.read_csv(nsl_cv_path)
                print(f"📊 Loaded NSL-KDD CV results from: {nsl_cv_path}")
            except Exception as e:
                print(f"⚠️ Failed to load NSL-KDD CV results: {e}")
        
        # CIC-IDS-2017 cross-validation
        cic_cv_path = "data/results/cross_validation/cic/cv_summary_table.csv"
        if Path(cic_cv_path).exists():
            try:
                cv_results['CIC-IDS-2017'] = pd.read_csv(cic_cv_path)
                print(f"📊 Loaded CIC-IDS-2017 CV results from: {cic_cv_path}")
            except Exception as e:
                print(f"⚠️ Failed to load CIC-IDS-2017 CV results: {e}")
        
        return cv_results
    
    def load_cross_dataset_results(self) -> Dict[str, pd.DataFrame]:
        """
        Load cross-dataset transfer learning results
        
        Returns:
            Dictionary with transfer direction as keys
        """
        cross_dataset_results = {}
        
        # NSL → CIC transfer
        nsl_to_cic_path = "data/results/nsl_trained_tested_on_cic.csv"
        if Path(nsl_to_cic_path).exists():
            try:
                cross_dataset_results['NSL→CIC'] = pd.read_csv(nsl_to_cic_path)
                print(f"📊 Loaded NSL→CIC transfer results")
            except Exception as e:
                print(f"⚠️ Failed to load NSL→CIC results: {e}")
        
        # CIC → NSL transfer
        cic_to_nsl_path = "data/results/cic_trained_tested_on_nsl.csv"
        if Path(cic_to_nsl_path).exists():
            try:
                cross_dataset_results['CIC→NSL'] = pd.read_csv(cic_to_nsl_path)
                print(f"📊 Loaded CIC→NSL transfer results")
            except Exception as e:
                print(f"⚠️ Failed to load CIC→NSL results: {e}")
        
        # Bidirectional analysis
        bidirectional_path = "data/results/bidirectional_cross_dataset_analysis.csv"
        if Path(bidirectional_path).exists():
            try:
                cross_dataset_results['Bidirectional'] = pd.read_csv(bidirectional_path)
                print(f"📊 Loaded bidirectional analysis results")
            except Exception as e:
                print(f"⚠️ Failed to load bidirectional results: {e}")
        
        return cross_dataset_results
    
    def load_harmonized_results(self) -> Optional[Dict]:
        """
        Load harmonized evaluation results
        
        Returns:
            Dictionary with harmonized results or None if not available
        """
        harmonized_path = "data/results/harmonized_cross_validation.json"
        if Path(harmonized_path).exists():
            try:
                import json
                with open(harmonized_path, 'r') as f:
                    harmonized_data = json.load(f)
                print(f"📊 Loaded harmonized evaluation results")
                return harmonized_data
            except Exception as e:
                print(f"⚠️ Failed to load harmonized results: {e}")
        return None
    
    def create_model_performance_comparison(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Create comprehensive model performance comparison chart
        
        Args:
            save_path: Optional path to save figure
            
        Returns:
            matplotlib Figure object
        """
        baseline_results, advanced_results = self.load_all_results()
        
        if baseline_results.empty or advanced_results.empty:
            print("❌ No results data available")
            return None
        
        # Combine results
        all_results = pd.concat([baseline_results, advanced_results], ignore_index=True)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle('Machine Learning Models Performance Comparison\nNSL-KDD Dataset', 
                     fontsize=14, fontweight='bold')
        
        # Sort by accuracy for consistent ordering (descending for better readability)
        all_results = all_results.sort_values('accuracy', ascending=False)
        
        # Clean model names for better display
        all_results['display_name'] = all_results['model_name'].str.replace('_', ' ').str.title()
        
        # Color mapping by category
        category_colors = {
            'Traditional ML': self.colors['primary'],
            'Ensemble Methods': self.colors['secondary'],
            'Neural Networks': self.colors['accent'],
            'Other': self.colors['neutral']
        }
        colors = [category_colors.get(cat, self.colors['neutral']) for cat in all_results['category']]
        
        # 1. Accuracy comparison (horizontal bars, top performers first)
        y_pos = np.arange(len(all_results))
        axes[0, 0].barh(y_pos, all_results['accuracy'], color=colors, alpha=0.8, edgecolor='white', linewidth=0.5)
        axes[0, 0].set_yticks(y_pos)
        axes[0, 0].set_yticklabels(all_results['display_name'], fontsize=8)
        axes[0, 0].set_title('Model Accuracy Comparison', fontweight='bold', fontsize=10)
        axes[0, 0].set_xlabel('Accuracy', fontsize=9)
        
        # Set x-axis limits based on data range
        min_acc = all_results['accuracy'].min()
        if min_acc > 0.8:
            axes[0, 0].set_xlim(max(0.0, min_acc - 0.1), 1.02)
        else:
            axes[0, 0].set_xlim(0.0, 1.02)
        
        # Add value labels with better positioning
        for i, v in enumerate(all_results['accuracy']):
            axes[0, 0].text(v + 0.01, i, f'{v:.3f}', va='center', ha='left', fontsize=7, fontweight='bold')
        
        axes[0, 0].grid(True, alpha=0.3, axis='x')
        
        # 2. F1-Score comparison
        axes[0, 1].barh(y_pos, all_results['f1_score'], color=colors, alpha=0.8, edgecolor='white', linewidth=0.5)
        axes[0, 1].set_yticks(y_pos)
        axes[0, 1].set_yticklabels(all_results['display_name'], fontsize=8)
        axes[0, 1].set_title('F1-Score Comparison', fontweight='bold', fontsize=10)
        axes[0, 1].set_xlabel('F1-Score', fontsize=9)
        
        min_f1 = all_results['f1_score'].min()
        if min_f1 > 0.8:
            axes[0, 1].set_xlim(max(0.0, min_f1 - 0.1), 1.02)
        else:
            axes[0, 1].set_xlim(0.0, 1.02)
        
        for i, v in enumerate(all_results['f1_score']):
            axes[0, 1].text(v + 0.01, i, f'{v:.3f}', va='center', ha='left', fontsize=7, fontweight='bold')
        
        axes[0, 1].grid(True, alpha=0.3, axis='x')
        
        # 3. Precision vs Recall scatter (improved)
        scatter_colors = [category_colors.get(cat, self.colors['neutral']) for cat in all_results['category']]
        scatter = axes[1, 0].scatter(all_results['recall'], all_results['precision'], 
                                    c=scatter_colors, s=120, alpha=0.7, edgecolors='white', linewidth=1)
        
        # Add model name labels with better positioning to avoid overlaps
        texts = []
        for i, row in all_results.iterrows():
            text = axes[1, 0].annotate(row['display_name'], 
                                      (row['recall'], row['precision']),
                                      xytext=(3, 3), textcoords='offset points',
                                      fontsize=6, alpha=0.8, ha='left')
            texts.append(text)
        
        axes[1, 0].set_title('Precision vs Recall', fontweight='bold', fontsize=10)
        axes[1, 0].set_xlabel('Recall', fontsize=9)
        axes[1, 0].set_ylabel('Precision', fontsize=9)
        
        # Set limits based on data range
        min_recall, max_recall = all_results['recall'].min(), all_results['recall'].max()
        min_precision, max_precision = all_results['precision'].min(), all_results['precision'].max()
        
        x_margin = (max_recall - min_recall) * 0.1 if max_recall > min_recall else 0.1
        y_margin = (max_precision - min_precision) * 0.1 if max_precision > min_precision else 0.1
        
        axes[1, 0].set_xlim(max(0, min_recall - x_margin), min(1.05, max_recall + x_margin))
        axes[1, 0].set_ylim(max(0, min_precision - y_margin), min(1.05, max_precision + y_margin))
        
        # Add diagonal line (perfect balance)
        lim_min = max(axes[1, 0].get_xlim()[0], axes[1, 0].get_ylim()[0])
        lim_max = min(axes[1, 0].get_xlim()[1], axes[1, 0].get_ylim()[1])
        axes[1, 0].plot([lim_min, lim_max], [lim_min, lim_max], 'k--', alpha=0.3, linewidth=1)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. ROC-AUC comparison (if available) or model distribution
        if 'roc_auc' in all_results.columns:
            axes[1, 1].barh(y_pos, all_results['roc_auc'], color=colors, alpha=0.8, edgecolor='white', linewidth=0.5)
            axes[1, 1].set_yticks(y_pos)
            axes[1, 1].set_yticklabels(all_results['display_name'], fontsize=8)
            axes[1, 1].set_title('ROC-AUC Comparison', fontweight='bold', fontsize=10)
            axes[1, 1].set_xlabel('ROC-AUC', fontsize=9)
            
            min_roc = all_results['roc_auc'].min()
            if min_roc > 0.9:
                axes[1, 1].set_xlim(max(0.0, min_roc - 0.05), 1.02)
            else:
                axes[1, 1].set_xlim(0.0, 1.02)
            
            for i, v in enumerate(all_results['roc_auc']):
                axes[1, 1].text(v + 0.005, i, f'{v:.3f}', va='center', ha='left', fontsize=7, fontweight='bold')
            
            axes[1, 1].grid(True, alpha=0.3, axis='x')
        else:
            # Alternative: Model category distribution
            category_counts = all_results['category'].value_counts()
            wedges, texts_pie, autotexts = axes[1, 1].pie(category_counts.values, 
                                                          labels=category_counts.index, 
                                                          colors=[category_colors.get(cat, self.colors['neutral']) for cat in category_counts.index],
                                                          autopct='%1.1f%%',
                                                          startangle=90)
            axes[1, 1].set_title('Model Categories Distribution', fontweight='bold', fontsize=10)
            
            # Improve pie chart text
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(8)
        
        # Create improved legend
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, edgecolor='white', label=category) 
                          for category, color in category_colors.items() if category in all_results['category'].values]
        fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=8)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Leave space for suptitle
        
        # Save figure
        if save_path is None:
            dataset_prefix = "nsl_cic" if self._has_both_datasets() else ("nsl" if self._has_nsl_data() else "cic")
            save_path = self.output_dir / f"{dataset_prefix}_model_performance_comparison.png"
            
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"📊 Model comparison figure saved to {save_path}")
        
        return fig
    
    def create_attack_distribution_analysis(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Create attack distribution analysis figure
        
        Args:
            save_path: Optional path to save figure
            
        Returns:
            matplotlib Figure object
        """
        # Load NSL-KDD data for analysis
        try:
            from src.preprocessing import NSLKDDAnalyzer

            analyzer = NSLKDDAnalyzer()
            train_data = analyzer.load_data("KDDTrain+.txt")
            test_data = analyzer.load_data("KDDTest+.txt")
            
            if train_data is None or test_data is None:
                print("❌ Could not load dataset for analysis")
                return None
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('NSL-KDD Dataset Analysis\nAttack Distribution and Characteristics', 
                     fontsize=16, fontweight='bold')
        
        # 1. Training set attack distribution
        train_attack_dist = train_data['attack_category'].value_counts()
        colors_pie = plt.cm.Set3(np.linspace(0, 1, len(train_attack_dist)))
        
        wedges, texts, autotexts = axes[0, 0].pie(train_attack_dist.values, 
                                                 labels=train_attack_dist.index,
                                                 autopct='%1.1f%%',
                                                 colors=colors_pie,
                                                 startangle=90)
        axes[0, 0].set_title('Training Set\nAttack Category Distribution', fontweight='bold')
        
        # 2. Test set attack distribution
        test_attack_dist = test_data['attack_category'].value_counts()
        
        wedges2, texts2, autotexts2 = axes[0, 1].pie(test_attack_dist.values,
                                                     labels=test_attack_dist.index,
                                                     autopct='%1.1f%%',
                                                     colors=colors_pie,
                                                     startangle=90)
        axes[0, 1].set_title('Test Set\nAttack Category Distribution', fontweight='bold')
        
        # 3. Detailed attack types (top 10)
        train_attack_types = train_data['attack_type'].value_counts().head(10)
        
        axes[1, 0].barh(range(len(train_attack_types)), train_attack_types.values, 
                       color=self.colors['primary'])
        axes[1, 0].set_yticks(range(len(train_attack_types)))
        axes[1, 0].set_yticklabels(train_attack_types.index)
        axes[1, 0].set_title('Top 10 Attack Types\n(Training Set)', fontweight='bold')
        axes[1, 0].set_xlabel('Number of Instances')
        
        # Add value labels
        for i, v in enumerate(train_attack_types.values):
            axes[1, 0].text(v + max(train_attack_types.values) * 0.01, i, 
                           f'{v:,}', va='center', fontsize=9)
        
        # 4. Dataset comparison table
        comparison_data = {
            'Metric': ['Total Records', 'Features', 'Attack Categories', 'Normal Traffic %', 'Attack Traffic %'],
            'Training Set': [
                f'{len(train_data):,}',
                f'{train_data.shape[1] - 2}',
                f'{train_data["attack_category"].nunique()}',
                f'{(train_data["attack_category"] == "Normal").mean()*100:.1f}%',
                f'{(train_data["attack_category"] != "Normal").mean()*100:.1f}%'
            ],
            'Test Set': [
                f'{len(test_data):,}',
                f'{test_data.shape[1] - 2}',
                f'{test_data["attack_category"].nunique()}',
                f'{(test_data["attack_category"] == "Normal").mean()*100:.1f}%',
                f'{(test_data["attack_category"] != "Normal").mean()*100:.1f}%'
            ]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Create table
        axes[1, 1].axis('tight')
        axes[1, 1].axis('off')
        table = axes[1, 1].table(cellText=comparison_df.values,
                                colLabels=comparison_df.columns,
                                cellLoc='center',
                                loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Style table
        for i in range(len(comparison_df.columns)):
            table[(0, i)].set_facecolor(self.colors['primary'])
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        axes[1, 1].set_title('Dataset Statistics Comparison', fontweight='bold')
        
        plt.tight_layout()
        
        # Save figure
        if save_path is None:
            save_path = self.output_dir / "nsl_attack_distribution_analysis.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"📊 Attack distribution analysis saved to {save_path}")
        
        return fig
    
    def create_cic_attack_distribution_analysis(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Create CIC-IDS-2017 attack distribution analysis figure
        
        Args:
            save_path: Optional path to save figure
            
        Returns:
            matplotlib Figure object
        """
        # Load CIC-IDS-2017 data for analysis
        try:
            preprocessor = CICIDSPreprocessor()
            cic_data = preprocessor.load_data(use_full_dataset=True)
            
            if cic_data is None:
                print("❌ Could not load CIC-IDS-2017 dataset for analysis")
                return None
            
        except Exception as e:
            print(f"❌ Error loading CIC-IDS-2017 data: {e}")
            return None
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('CIC-IDS-2017 Dataset Analysis\nAttack Distribution and Characteristics', 
                     fontsize=16, fontweight='bold')
        
        # 1. Overall attack type distribution (top 10)
        attack_dist = cic_data['Label'].value_counts().head(10)
        colors_pie = plt.cm.Set3(np.linspace(0, 1, len(attack_dist)))
        
        wedges, texts, autotexts = axes[0, 0].pie(attack_dist.values, 
                                                 labels=attack_dist.index,
                                                 autopct='%1.1f%%',
                                                 colors=colors_pie,
                                                 startangle=90)
        axes[0, 0].set_title('Top 10 Attack Types\nDistribution', fontweight='bold')
        
        # 2. Binary classification distribution (Normal vs Attack)
        binary_dist = cic_data['Label'].apply(lambda x: 'Normal' if x == 'BENIGN' else 'Attack').value_counts()
        
        wedges2, texts2, autotexts2 = axes[0, 1].pie(binary_dist.values,
                                                     labels=binary_dist.index,
                                                     autopct='%1.1f%%',
                                                     colors=[self.colors['success'], self.colors['primary']],
                                                     startangle=90)
        axes[0, 1].set_title('Binary Classification\nNormal vs Attack', fontweight='bold')
        
        # 3. Daily attack distribution (by file source)
        # Map files to days
        day_mapping = {
            'Monday-WorkingHours.pcap_ISCX.csv': 'Monday',
            'Tuesday-WorkingHours.pcap_ISCX.csv': 'Tuesday', 
            'Wednesday-workingHours.pcap_ISCX.csv': 'Wednesday',
            'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv': 'Thu-Morning',
            'Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv': 'Thu-Afternoon',
            'Friday-WorkingHours-Morning.pcap_ISCX.csv': 'Fri-Morning',
            'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv': 'Fri-DDoS',
            'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv': 'Fri-PortScan'
        }
        
        # Sample data for visualization (full dataset is too large to process efficiently)
        sample_size = min(50000, len(cic_data))
        if len(cic_data) > sample_size:
            cic_sample = cic_data.sample(n=sample_size, random_state=42)
        else:
            cic_sample = cic_data
        
        # Create attack category groupings
        attack_categories = {
            'Normal': ['BENIGN'],
            'DoS/DDoS': ['DoS Hulk', 'DoS GoldenEye', 'DoS slowloris', 'DoS Slowhttptest', 'DDoS'],
            'Brute Force': ['FTP-Patator', 'SSH-Patator'],
            'Web Attacks': ['Web Attack – Brute Force', 'Web Attack – XSS', 'Web Attack – Sql Injection'],
            'Botnet': ['Bot'],
            'Infiltration': ['Infiltration'],
            'Port Scan': ['PortScan'],
            'Heartbleed': ['Heartbleed']
        }
        
        # Map attacks to categories
        def categorize_attack(label):
            for category, attacks in attack_categories.items():
                if label in attacks:
                    return category
            return 'Other'
        
        cic_sample['attack_category'] = cic_sample['Label'].apply(categorize_attack)
        category_dist = cic_sample['attack_category'].value_counts()
        
        axes[1, 0].barh(range(len(category_dist)), category_dist.values, 
                       color=self.colors['secondary'])
        axes[1, 0].set_yticks(range(len(category_dist)))
        axes[1, 0].set_yticklabels(category_dist.index)
        axes[1, 0].set_title('Attack Categories\nDistribution', fontweight='bold')
        axes[1, 0].set_xlabel('Number of Instances (Sample)')
        
        # Add value labels
        for i, v in enumerate(category_dist.values):
            axes[1, 0].text(v + max(category_dist.values) * 0.01, i, 
                           f'{v:,}', va='center', fontsize=9)
        
        # 4. Dataset statistics table
        normal_count = (cic_data['Label'] == 'BENIGN').sum()
        attack_count = (cic_data['Label'] != 'BENIGN').sum()
        
        comparison_data = {
            'Metric': ['Total Records', 'Features', 'Attack Types', 'Normal Traffic %', 'Attack Traffic %', 'Days Covered'],
            'CIC-IDS-2017': [
                f'{len(cic_data):,}',
                f'{cic_data.shape[1] - 1}',  # Exclude Label column
                f'{cic_data["Label"].nunique()}',
                f'{(normal_count / len(cic_data))*100:.1f}%',
                f'{(attack_count / len(cic_data))*100:.1f}%',
                '5 (Mon-Fri)'
            ]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Create table
        axes[1, 1].axis('tight')
        axes[1, 1].axis('off')
        table = axes[1, 1].table(cellText=comparison_df.values,
                                colLabels=comparison_df.columns,
                                cellLoc='center',
                                loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Style table
        for i in range(len(comparison_df.columns)):
            table[(0, i)].set_facecolor(self.colors['secondary'])
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        axes[1, 1].set_title('CIC-IDS-2017 Statistics', fontweight='bold')
        
        plt.tight_layout()
        
        # Save figure
        if save_path is None:
            save_path = self.output_dir / "cic_attack_distribution_analysis.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"📊 CIC-IDS-2017 attack distribution analysis saved to {save_path}")
        
        return fig
    
    def create_cross_validation_comparison(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Create cross-validation performance comparison between datasets
        
        Args:
            save_path: Optional path to save figure
            
        Returns:
            matplotlib Figure object
        """
        cv_results = self.load_cross_validation_results()
        
        if not cv_results:
            print("❌ No cross-validation results available")
            return None
        
        # Use single plot if only one dataset, otherwise subplot layout
        if len(cv_results) == 1:
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            axes = [ax]
        else:
            fig, axes = plt.subplots(1, len(cv_results), figsize=(6*len(cv_results), 8))
            if not isinstance(axes, (list, np.ndarray)):
                axes = [axes]
        
        fig.suptitle('Cross-Validation Performance Comparison', fontsize=14, fontweight='bold')
        
        for i, (dataset, df) in enumerate(cv_results.items()):
            if df.empty:
                axes[i].text(0.5, 0.5, f'No data available for {dataset}', 
                           ha='center', va='center', transform=axes[i].transAxes,
                           fontsize=12, style='italic')
                axes[i].set_title(f'{dataset} Cross-Validation', fontweight='bold')
                continue
                
            # Parse F1-Score (remove ± part)
            f1_scores = df['F1-Score'].str.split(' ±').str[0].astype(float)
            
            # Sort by F1-Score in descending order (best first)
            sorted_indices = f1_scores.argsort()[::-1]
            sorted_models = df.iloc[sorted_indices]['Model'].str.replace('_', ' ').str.title()
            sorted_f1 = f1_scores.iloc[sorted_indices]
            
            # Create horizontal bar plot with better colors
            y_pos = np.arange(len(sorted_models))
            bars = axes[i].barh(y_pos, sorted_f1, 
                               color=self.colors['primary'], alpha=0.8, edgecolor='white', linewidth=0.5)
            
            # Customize plot
            axes[i].set_yticks(y_pos)
            axes[i].set_yticklabels(sorted_models, fontsize=8)
            axes[i].set_xlabel('F1-Score', fontsize=9)
            axes[i].set_title(f'{dataset} Cross-Validation', fontweight='bold', fontsize=10)
            
            # Set appropriate x-axis limits based on data range
            min_score = sorted_f1.min()
            if min_score > 0.9:
                axes[i].set_xlim(0.85, 1.0)
            else:
                axes[i].set_xlim(0, 1.0)
            
            # Add value labels with better positioning
            for j, v in enumerate(sorted_f1):
                axes[i].text(v + 0.005, j, f'{v:.3f}', 
                           va='center', ha='left', fontsize=8, fontweight='bold')
            
            # Improve grid
            axes[i].grid(True, alpha=0.3, axis='x')
            axes[i].set_axisbelow(True)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Leave space for suptitle
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        else:
            plt.savefig(self.output_dir / "cross_validation_comparison.png", dpi=300, bbox_inches='tight', facecolor='white')
        
        return fig
    
    def create_cross_dataset_transfer_analysis(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Create cross-dataset transfer learning analysis
        
        Args:
            save_path: Optional path to save figure
            
        Returns:
            matplotlib Figure object
        """
        cross_dataset_results = self.load_cross_dataset_results()
        
        if not cross_dataset_results:
            print("❌ No cross-dataset results available")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Cross-Dataset Transfer Learning Analysis', fontsize=14, fontweight='bold')
        
        # 1. Transfer Ratios Comparison using bidirectional data
        if 'Bidirectional' in cross_dataset_results:
            bidirectional = cross_dataset_results['Bidirectional']
            
            models = bidirectional['Model'].values
            nsl_cic_ratios = bidirectional['NSL_to_CIC_Transfer_Ratio'].values
            cic_nsl_ratios = bidirectional['CIC_to_NSL_Transfer_Ratio'].values
            
            x = np.arange(len(models))
            width = 0.35
            
            bars1 = axes[0, 0].bar(x - width/2, nsl_cic_ratios, width, label='NSL→CIC', 
                          color=self.colors['primary'], alpha=0.8)
            bars2 = axes[0, 0].bar(x + width/2, cic_nsl_ratios, width, label='CIC→NSL', 
                          color=self.colors['secondary'], alpha=0.8)
            
            # Add value labels
            for i, (v1, v2) in enumerate(zip(nsl_cic_ratios, cic_nsl_ratios)):
                axes[0, 0].text(x[i] - width/2, v1 + 0.02, f'{v1:.2f}', ha='center', fontsize=8)
                axes[0, 0].text(x[i] + width/2, v2 + 0.02, f'{v2:.2f}', ha='center', fontsize=8)
            
            axes[0, 0].set_title('Transfer Ratios by Direction', fontweight='bold', fontsize=10)
            axes[0, 0].set_xlabel('Models', fontsize=9)
            axes[0, 0].set_ylabel('Transfer Ratio', fontsize=9)
            axes[0, 0].set_xticks(x)
            axes[0, 0].set_xticklabels(models, rotation=45, ha='right', fontsize=8)
            axes[0, 0].legend(fontsize=8)
            axes[0, 0].grid(True, alpha=0.3, axis='y')
            axes[0, 0].set_ylim(0, 1.1)
        
        # 2. Performance Drop Analysis
        if 'NSL→CIC' in cross_dataset_results:
            nsl_to_cic = cross_dataset_results['NSL→CIC']
            source_acc = nsl_to_cic['Source_Accuracy'].values
            target_acc = nsl_to_cic['Target_Accuracy'].values
            
            scatter = axes[0, 1].scatter(source_acc, target_acc,
                              c=self.colors['accent'], s=120, alpha=0.7, edgecolors='white')
            
            # Add diagonal line (no performance drop)
            min_acc = min(source_acc.min(), target_acc.min())
            max_acc = max(source_acc.max(), target_acc.max())
            axes[0, 1].plot([min_acc, max_acc], [min_acc, max_acc], 'k--', alpha=0.5, label='Perfect Transfer')
            
            # Add model labels with better positioning
            for i, model in enumerate(nsl_to_cic['Model']):
                axes[0, 1].annotate(model, (source_acc[i], target_acc[i]),
                                   xytext=(3, 3), textcoords='offset points', fontsize=7, alpha=0.8)
            
            axes[0, 1].set_title('NSL→CIC Transfer Performance', fontweight='bold', fontsize=10)
            axes[0, 1].set_xlabel('Source Accuracy (NSL-KDD)', fontsize=9)
            axes[0, 1].set_ylabel('Target Accuracy (CIC-IDS-2017)', fontsize=9)
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].legend(fontsize=8)
            axes[0, 1].set_xlim(0, 1)
            axes[0, 1].set_ylim(0, 1)
        
        # 3. Relative Performance Drop
        if 'Bidirectional' in cross_dataset_results:
            bidirectional = cross_dataset_results['Bidirectional']
            
            # Sort by average transfer ratio
            sorted_idx = bidirectional['Avg_Transfer_Ratio'].argsort()[::-1]
            sorted_models = bidirectional['Model'].iloc[sorted_idx]
            sorted_ratios = bidirectional['Avg_Transfer_Ratio'].iloc[sorted_idx]
            
            y_pos = np.arange(len(sorted_models))
            bars = axes[1, 0].barh(y_pos, sorted_ratios,
                           color=self.colors['accent'], alpha=0.8)
            axes[1, 0].set_yticks(y_pos)
            axes[1, 0].set_yticklabels(sorted_models, fontsize=8)
            axes[1, 0].set_title('Average Transfer Performance', fontweight='bold', fontsize=10)
            axes[1, 0].set_xlabel('Average Transfer Ratio', fontsize=9)
            axes[1, 0].grid(True, alpha=0.3, axis='x')
            
            # Add value labels
            for i, v in enumerate(sorted_ratios):
                axes[1, 0].text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=8, fontweight='bold')
        
        # 4. Transfer Asymmetry
        if 'Bidirectional' in cross_dataset_results:
            bidirectional = cross_dataset_results['Bidirectional']
            
            nsl_cic_ratios = bidirectional['NSL_to_CIC_Transfer_Ratio'].values
            cic_nsl_ratios = bidirectional['CIC_to_NSL_Transfer_Ratio'].values
            
            scatter = axes[1, 1].scatter(nsl_cic_ratios, cic_nsl_ratios,
                              c=self.colors['warning'], s=120, alpha=0.7, edgecolors='white')
            
            # Add diagonal line (symmetric transfer)
            axes[1, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Symmetry')
            
            # Add model labels with better positioning
            for i, model in enumerate(bidirectional['Model']):
                axes[1, 1].annotate(model, (nsl_cic_ratios[i], cic_nsl_ratios[i]),
                                   xytext=(3, 3), textcoords='offset points', fontsize=7, alpha=0.8)
            
            axes[1, 1].set_title('Transfer Asymmetry Analysis', fontweight='bold', fontsize=10)
            axes[1, 1].set_xlabel('NSL→CIC Transfer Ratio', fontsize=9)
            axes[1, 1].set_ylabel('CIC→NSL Transfer Ratio', fontsize=9)
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].legend(fontsize=8)
            axes[1, 1].set_xlim(0, 1.1)
            axes[1, 1].set_ylim(0, 1.1)
        else:
            # Show message if no data available
            axes[1, 0].text(0.5, 0.5, 'Cross-dataset analysis data not available', 
                           ha='center', va='center', transform=axes[1, 0].transAxes,
                           fontsize=12, style='italic')
            axes[1, 1].text(0.5, 0.5, 'Transfer asymmetry data not available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes,
                           fontsize=12, style='italic')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        else:
            plt.savefig(self.output_dir / "cross_dataset_transfer_analysis.png", dpi=300, bbox_inches='tight', facecolor='white')
        
        return fig
    
    def create_harmonized_evaluation_summary(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Create harmonized evaluation summary visualization
        
        Args:
            save_path: Optional path to save figure
            
        Returns:
            matplotlib Figure object
        """
        harmonized_data = self.load_harmonized_results()
        
        if not harmonized_data:
            print("❌ No harmonized evaluation results available")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Harmonized Cross-Dataset Evaluation Summary', fontsize=14, fontweight='bold')
        
        # 1. Dataset statistics comparison
        if 'nsl_summary' in harmonized_data and 'cic_summary' in harmonized_data:
            nsl_stats = harmonized_data['nsl_summary']
            cic_stats = harmonized_data['cic_summary']
            
            # Compare dataset sizes and characteristics
            datasets = ['NSL-KDD', 'CIC-IDS-2017']
            rows = [nsl_stats.get('rows', 0), cic_stats.get('rows', 0)]
            
            bars = axes[0, 0].bar(datasets, rows, color=[self.colors['primary'], self.colors['secondary']], alpha=0.8)
            axes[0, 0].set_title('Dataset Size Comparison', fontweight='bold', fontsize=10)
            axes[0, 0].set_ylabel('Number of Records', fontsize=9)
            axes[0, 0].grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar, val in zip(bars, rows):
                height = bar.get_height()
                axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                               f'{val:,}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        else:
            axes[0, 0].text(0.5, 0.5, 'Dataset statistics not available', 
                           ha='center', va='center', transform=axes[0, 0].transAxes,
                           fontsize=12, style='italic')
            axes[0, 0].set_title('Dataset Statistics', fontweight='bold', fontsize=10)
        
        
        # 2. Configuration summary from harmonized data
        if harmonized_data:
            config_text = f"Harmonized Evaluation Configuration:\n\n"
            config_text += f"Schema Version: {harmonized_data.get('schema_version', 'N/A')}\n"
            config_text += f"Max Samples: {harmonized_data.get('max_samples', 'N/A'):,}\n"
            config_text += f"Batch Size: {harmonized_data.get('batch_size', 'N/A'):,}\n"
            config_text += f"Memory Mode: {harmonized_data.get('memory_mode', 'N/A')}\n"
            config_text += f"Training Method: {harmonized_data.get('training_method', 'N/A')}"
            
            axes[0, 1].text(0.05, 0.95, config_text, transform=axes[0, 1].transAxes,
                           fontsize=10, verticalalignment='top', fontfamily='monospace',
                           bbox=dict(boxstyle='round', facecolor=self.colors['neutral'], alpha=0.1))
            axes[0, 1].set_title('Harmonized Configuration', fontweight='bold', fontsize=10)
            axes[0, 1].axis('off')
        
        # 3. Show dataset feature comparison if available
        if 'nsl_summary' in harmonized_data and 'cic_summary' in harmonized_data:
            nsl_features = harmonized_data['nsl_summary'].get('feature_count', 0)
            cic_features = harmonized_data['cic_summary'].get('feature_count', 0)
            
            datasets = ['NSL-KDD', 'CIC-IDS-2017']
            features = [nsl_features, cic_features]
            
            if features[0] > 0 or features[1] > 0:
                axes[1, 0].bar(datasets, features, color=[self.colors['accent'], self.colors['warning']], alpha=0.8)
                axes[1, 0].set_title('Feature Count Comparison', fontweight='bold', fontsize=10)
                axes[1, 0].set_ylabel('Number of Features', fontsize=9)
                axes[1, 0].grid(True, alpha=0.3, axis='y')
                
                # Add value labels
                for i, val in enumerate(features):
                    if val > 0:
                        axes[1, 0].text(i, val + val*0.01, f'{val}', ha='center', va='bottom', fontsize=9, fontweight='bold')
            else:
                axes[1, 0].text(0.5, 0.5, 'Feature information not available', 
                               ha='center', va='center', transform=axes[1, 0].transAxes,
                               fontsize=12, style='italic')
                axes[1, 0].set_title('Feature Comparison', fontweight='bold', fontsize=10)
        
        # 4. Display key harmonization metrics if available
        metrics_text = "Harmonization Process Metrics:\n\n"
        if 'evaluation_summary' in harmonized_data:
            eval_summary = harmonized_data['evaluation_summary']
            metrics_text += f"Total Experiments: {eval_summary.get('total_experiments', 'N/A')}\n"
            metrics_text += f"Successful Runs: {eval_summary.get('successful_runs', 'N/A')}\n"
            metrics_text += f"Average Runtime: {eval_summary.get('avg_runtime', 'N/A')}\n"
        else:
            metrics_text += "Harmonization enables consistent\nevaluation across different datasets\nby standardizing feature spaces\nand training procedures.\n\n"
            metrics_text += "This ensures fair comparison\nof model performance across\nheterogeneous network datasets."
        
        axes[1, 1].text(0.05, 0.95, metrics_text, transform=axes[1, 1].transAxes,
                       fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor=self.colors['accent'], alpha=0.1))
        axes[1, 1].set_title('Harmonization Benefits', fontweight='bold', fontsize=10)
        axes[1, 1].axis('off')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        else:
            plt.savefig(self.output_dir / "harmonized_evaluation_summary.png", dpi=300, bbox_inches='tight', facecolor='white')
        
        print(f"📊 Harmonized evaluation summary saved successfully")
        return fig
    
    def create_dataset_comparison_overview(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Create comprehensive dataset comparison overview
        
        Args:
            save_path: Optional path to save figure
            
        Returns:
            matplotlib Figure object
        """
        baseline_results, advanced_results = self.load_all_results()
        
        if baseline_results.empty and advanced_results.empty:
            print("❌ No results available for dataset comparison")
            return None
        
        # Combine all results
        all_results = pd.concat([baseline_results, advanced_results], ignore_index=True)
        
        if 'dataset' not in all_results.columns:
            print("⚠️ Dataset information not available, cannot create comparison")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Dataset Performance Comparison Overview', fontsize=14, fontweight='bold')
        
        # Get datasets and create better color scheme
        datasets = sorted(all_results['dataset'].unique())
        dataset_colors = {datasets[i]: self.qual_colors[i] for i in range(len(datasets))}
        
        # 1. Accuracy comparison by dataset (improved scatter plot)
        for i, dataset in enumerate(datasets):
            dataset_data = all_results[all_results['dataset'] == dataset].reset_index(drop=True)
            axes[0, 0].scatter(range(len(dataset_data)), dataset_data['accuracy'], 
                              label=dataset, color=dataset_colors[dataset], s=80, alpha=0.7, edgecolors='white')
        
        axes[0, 0].set_title('Accuracy by Dataset', fontweight='bold', fontsize=10)
        axes[0, 0].set_xlabel('Model Index', fontsize=9)
        axes[0, 0].set_ylabel('Accuracy', fontsize=9)
        axes[0, 0].legend(fontsize=8)
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim(0.0, 1.05)
        
        # 2. F1-Score comparison by dataset (improved scatter plot)
        for i, dataset in enumerate(datasets):
            dataset_data = all_results[all_results['dataset'] == dataset].reset_index(drop=True)
            axes[0, 1].scatter(range(len(dataset_data)), dataset_data['f1_score'], 
                              label=dataset, color=dataset_colors[dataset], s=80, alpha=0.7, edgecolors='white')
        
        axes[0, 1].set_title('F1-Score by Dataset', fontweight='bold', fontsize=10)
        axes[0, 1].set_xlabel('Model Index', fontsize=9)
        axes[0, 1].set_ylabel('F1-Score', fontsize=9)
        axes[0, 1].legend(fontsize=8)
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim(0.0, 1.05)
        
        # 3. Model category distribution by dataset (improved bar chart)
        category_dataset = all_results.groupby(['dataset', 'category']).size().unstack(fill_value=0)
        category_colors_list = [self.colors['primary'], self.colors['secondary'], self.colors['accent']]
        category_dataset.plot(kind='bar', ax=axes[1, 0], color=category_colors_list, width=0.8)
        axes[1, 0].set_title('Model Categories by Dataset', fontweight='bold', fontsize=10)
        axes[1, 0].set_xlabel('Dataset', fontsize=9)
        axes[1, 0].set_ylabel('Number of Models', fontsize=9)
        axes[1, 0].legend(title='Model Category', fontsize=8, title_fontsize=8)
        axes[1, 0].tick_params(axis='x', rotation=0, labelsize=8)
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # 4. Performance distribution by dataset (improved boxplot)
        accuracy_by_dataset = [all_results[all_results['dataset'] == d]['accuracy'].values for d in datasets]
        bp = axes[1, 1].boxplot(accuracy_by_dataset, labels=datasets, patch_artist=True)
        
        # Color the boxplots
        for patch, color in zip(bp['boxes'], [dataset_colors[d] for d in datasets]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        axes[1, 1].set_title('Accuracy Distribution by Dataset', fontweight='bold', fontsize=10)
        axes[1, 1].set_xlabel('Dataset', fontsize=9)
        axes[1, 1].set_ylabel('Accuracy', fontsize=9)
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        axes[1, 1].tick_params(axis='x', labelsize=8)
        axes[1, 1].set_ylim(0.0, 1.05)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Leave space for suptitle
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        else:
            plt.savefig(self.output_dir / "dataset_comparison_overview.png", dpi=300, bbox_inches='tight', facecolor='white')
        
        return fig
    
    def create_performance_summary_table(self, save_path: Optional[str] = None) -> pd.DataFrame:
        """
        Create publication-ready performance summary table
        
        Args:
            save_path: Optional path to save table
            
        Returns:
            DataFrame with formatted results
        """
        baseline_results, advanced_results = self.load_all_results()
        
        if baseline_results.empty or advanced_results.empty:
            print("❌ No results data available")
            return pd.DataFrame()
        
        # Combine and format results
        all_results = pd.concat([baseline_results, advanced_results], ignore_index=True)
        
        # Create summary table
        summary_table = pd.DataFrame({
            'Model': all_results['model_name'].str.replace('_', ' ').str.title(),
            'Category': all_results['category'],
            'Accuracy': all_results['accuracy'].apply(lambda x: f"{x:.4f}"),
            'Precision': all_results['precision'].apply(lambda x: f"{x:.4f}"),
            'Recall': all_results['recall'].apply(lambda x: f"{x:.4f}"),
            'F1-Score': all_results['f1_score'].apply(lambda x: f"{x:.4f}"),
        })
        
        # Add ROC-AUC if available
        if 'roc_auc' in all_results.columns:
            summary_table['ROC-AUC'] = all_results['roc_auc'].apply(lambda x: f"{x:.4f}")
        
        # Sort by accuracy
        summary_table['_accuracy_sort'] = all_results['accuracy']
        summary_table = summary_table.sort_values('_accuracy_sort', ascending=False)
        summary_table = summary_table.drop('_accuracy_sort', axis=1)
        summary_table = summary_table.reset_index(drop=True)
        
        # Add ranking
        summary_table.insert(0, 'Rank', range(1, len(summary_table) + 1))
        
        # Save to CSV and LaTeX
        if save_path is None:
            dataset_prefix = "nsl_cic" if self._has_both_datasets() else ("nsl" if self._has_nsl_data() else "cic")
            save_path = self.output_dir / f"{dataset_prefix}_performance_summary_table"
        
        summary_table.to_csv(f"{save_path}.csv", index=False)
        
        # Generate LaTeX table
        latex_table = self._generate_latex_table(summary_table)
        with open(f"{save_path}.tex", 'w') as f:
            f.write(latex_table)
        
        print(f"📊 Performance summary table saved to {save_path}.*")
        
        return summary_table
    
    def _generate_latex_table(self, df: pd.DataFrame) -> str:
        """Generate professional LaTeX table code"""
        
        # Create column specification with proper alignment
        col_alignments = {
            'Rank': 'c',
            'Model': 'l', 
            'Category': 'l',
            'Accuracy': 'c',
            'Precision': 'c', 
            'Recall': 'c',
            'F1-Score': 'c',
            'ROC-AUC': 'c'
        }
        
        col_spec = ''
        for col in df.columns:
            alignment = col_alignments.get(col, 'c')
            col_spec += alignment
        
        # Use booktabs for professional appearance
        latex_code = """\\begin{table}[htbp]
\\centering
\\caption{Machine Learning Models Performance Comparison on NSL-KDD Dataset}
\\label{tab:model_performance}
\\begin{tabular}{""" + col_spec + """}
\\toprule
"""
        
        # Add header with proper formatting
        header_mapping = {
            'Rank': '\\textbf{Rank}',
            'Model': '\\textbf{Model}',
            'Category': '\\textbf{Category}',
            'Accuracy': '\\textbf{Accuracy}',
            'Precision': '\\textbf{Precision}',
            'Recall': '\\textbf{Recall}',
            'F1-Score': '\\textbf{F1-Score}',
            'ROC-AUC': '\\textbf{ROC-AUC}'
        }
        
        headers = [header_mapping.get(col, f'\\textbf{{{col}}}') for col in df.columns]
        header = ' & '.join(headers) + ' \\\\\n'
        latex_code += header
        latex_code += '\\midrule\n'
        
        # Add rows with proper formatting
        for i, (_, row) in enumerate(df.iterrows()):
            # Format values appropriately
            formatted_values = []
            for col, val in zip(df.columns, row.values):
                if col == 'Rank':
                    formatted_values.append(str(val))
                elif col in ['Model', 'Category']:
                    # Escape underscores and special characters
                    formatted_val = str(val).replace('_', '\\_').replace('&', '\\&')
                    formatted_values.append(formatted_val)
                else:
                    # Numeric values
                    formatted_values.append(str(val))
            
            row_str = ' & '.join(formatted_values) + ' \\\\\n'
            latex_code += row_str
            
            # Add midrule every 5 rows for better readability
            if (i + 1) % 10 == 0 and i < len(df) - 1:
                latex_code += '\\midrule\n'
        
        latex_code += """\\bottomrule
\\end{tabular}
\\end{table}
"""
        
        return latex_code
    
    def generate_all_figures(self):
        """
        Generate all publication-ready figures and tables following scientific standards
        Creates publication-quality visualizations optimized for academic journals
        """
        print("🎨 GENERATING SCIENTIFIC PUBLICATION FIGURES")
        print("=" * 60)
        print("📐 Using IEEE/ACM publication standards")
        print("🎨 Colorblind-friendly palette with statistical rigor")
        print("=" * 60)
        
        success_count = 0
        total_figures = 0
        
        try:
            # 1. Model performance comparison
            total_figures += 1
            print("📊 Creating model performance comparison...")
            try:
                self.create_model_performance_comparison()
                success_count += 1
                print("   ✅ Model performance comparison completed")
            except Exception as e:
                print(f"   ⚠️ Model performance comparison failed: {e}")
            
            # 2. Dataset comparison overview (NEW)
            total_figures += 1
            print("📊 Creating dataset comparison overview...")
            try:
                self.create_dataset_comparison_overview()
                success_count += 1
                print("   ✅ Dataset comparison overview completed")
            except Exception as e:
                print(f"   ⚠️ Dataset comparison overview failed: {e}")
            
            # 3. Cross-validation comparison (NEW)
            total_figures += 1
            print("📊 Creating cross-validation comparison...")
            try:
                self.create_cross_validation_comparison()
                success_count += 1
                print("   ✅ Cross-validation comparison completed")
            except Exception as e:
                print(f"   ⚠️ Cross-validation comparison failed: {e}")
            
            # 4. Cross-dataset transfer analysis (NEW)
            total_figures += 1
            print("📊 Creating cross-dataset transfer analysis...")
            try:
                self.create_cross_dataset_transfer_analysis()
                success_count += 1
                print("   ✅ Cross-dataset transfer analysis completed")
            except Exception as e:
                print(f"   ⚠️ Cross-dataset transfer analysis failed: {e}")
            
            # 5. Harmonized evaluation summary (NEW)
            total_figures += 1
            print("📊 Creating harmonized evaluation summary...")
            try:
                self.create_harmonized_evaluation_summary()
                success_count += 1
                print("   ✅ Harmonized evaluation summary completed")
            except Exception as e:
                print(f"   ⚠️ Harmonized evaluation summary failed: {e}")
            
            # 6. Attack distribution analysis (NSL-KDD)
            total_figures += 1
            print("📊 Creating NSL-KDD attack distribution analysis...")
            try:
                self.create_attack_distribution_analysis()
                success_count += 1
                print("   ✅ NSL-KDD attack distribution analysis completed")
            except Exception as e:
                print(f"   ⚠️ NSL-KDD attack distribution analysis failed: {e}")
            
            # 7. CIC-IDS-2017 attack distribution analysis
            total_figures += 1
            print("📊 Creating CIC-IDS-2017 attack distribution analysis...")
            try:
                self.create_cic_attack_distribution_analysis()
                success_count += 1
                print("   ✅ CIC-IDS-2017 attack distribution analysis completed")
            except Exception as e:
                print(f"   ⚠️ CIC-IDS-2017 attack distribution analysis failed: {e}")
            
            # 8. Performance summary table
            total_figures += 1
            print("📊 Creating performance summary table...")
            try:
                summary_table = self.create_performance_summary_table()
                if not summary_table.empty:
                    print("\n📋 Performance Summary (Top 5):")
                    print(summary_table.head().to_string(index=False))
                success_count += 1
                print("   ✅ Performance summary table completed")
            except Exception as e:
                print(f"   ⚠️ Performance summary table failed: {e}")
            
            print(f"\n📊 FIGURE GENERATION SUMMARY")
            print(f"✅ Successfully generated: {success_count}/{total_figures} figures")
            print(f"📁 Output directory: {self.output_dir}")
            
            # List generated files
            if self.output_dir.exists():
                generated_files = list(self.output_dir.glob("*"))
                if generated_files:
                    print("\n📄 Generated files:")
                    for file in generated_files:
                        print(f"   📄 {file.name}")
            
            return success_count > 0
            
        except Exception as e:
            print(f"❌ Critical error in figure generation: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main function to generate all figures"""
    generator = PaperFigureGenerator()
    success = generator.generate_all_figures()
    return success


if __name__ == "__main__":
    main()