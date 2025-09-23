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

from src.preprocessing import CICIDSPreprocessor

warnings.filterwarnings('ignore')

# Set publication-ready style
plt.style.use('default')
sns.set_style("whitegrid")
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 11,
    'figure.titlesize': 16,
    'font.family': 'serif',
    'figure.dpi': 300
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
        
        # Define color palette for consistent styling
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72', 
            'accent': '#F18F01',
            'success': '#C73E1D',
            'neutral': '#6C757D'
        }
        
        # Model categories for grouping
        self.model_categories = {
            'Traditional ML': ['random_forest', 'decision_tree', 'logistic_regression', 'naive_bayes', 'knn'],
            'Ensemble Methods': ['xgboost', 'lightgbm', 'gradient_boosting', 'extra_trees', 'voting_classifier'],
            'Neural Networks': ['mlp']
        }
    
    def load_all_results(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load baseline and advanced model results
        
        Returns:
            Tuple of (baseline_results, advanced_results) DataFrames
        """
        try:
            baseline_results = pd.read_csv("data/results/baseline_results.csv")
            # Try multiple locations for advanced results
            try:
                advanced_results = pd.read_csv("data/models/advanced/advanced_results.csv")
            except FileNotFoundError:
                try:
                    advanced_results = pd.read_csv("data/results/advanced_results.csv")
                except FileNotFoundError:
                    # If no advanced results, create empty DataFrame
                    print("⚠️ Advanced results not found, using baseline results only")
                    advanced_results = pd.DataFrame()
            
            # Add model category
            def categorize_model(model_name):
                for category, models in self.model_categories.items():
                    if model_name in models:
                        return category
                return 'Other'
            
            baseline_results['category'] = baseline_results['model_name'].apply(categorize_model)
            advanced_results['category'] = advanced_results['model_name'].apply(categorize_model)
            
            return baseline_results, advanced_results
            
        except FileNotFoundError as e:
            print(f"❌ Could not load results: {e}")
            return pd.DataFrame(), pd.DataFrame()
    
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
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Machine Learning Models Performance Comparison\nNSL-KDD Dataset', 
                     fontsize=16, fontweight='bold')
        
        # Sort by accuracy for consistent ordering
        all_results = all_results.sort_values('accuracy', ascending=True)
        
        # Color mapping by category
        category_colors = {
            'Traditional ML': self.colors['primary'],
            'Ensemble Methods': self.colors['secondary'],
            'Neural Networks': self.colors['accent']
        }
        colors = [category_colors.get(cat, self.colors['neutral']) for cat in all_results['category']]
        
        # 1. Accuracy comparison
        axes[0, 0].barh(all_results['model_name'], all_results['accuracy'], color=colors)
        axes[0, 0].set_title('Model Accuracy Comparison', fontweight='bold')
        axes[0, 0].set_xlabel('Accuracy')
        axes[0, 0].set_xlim(0.8, 1.0)
        
        # Add value labels
        for i, v in enumerate(all_results['accuracy']):
            axes[0, 0].text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=9)
        
        # 2. F1-Score comparison
        axes[0, 1].barh(all_results['model_name'], all_results['f1_score'], color=colors)
        axes[0, 1].set_title('F1-Score Comparison', fontweight='bold')
        axes[0, 1].set_xlabel('F1-Score')
        axes[0, 1].set_xlim(0.8, 1.0)
        
        for i, v in enumerate(all_results['f1_score']):
            axes[0, 1].text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=9)
        
        # 3. Precision vs Recall scatter
        axes[1, 0].scatter(all_results['recall'], all_results['precision'], 
                          c=[category_colors.get(cat, self.colors['neutral']) for cat in all_results['category']], 
                          s=100, alpha=0.7)
        
        # Add model name labels
        for i, row in all_results.iterrows():
            axes[1, 0].annotate(row['model_name'], 
                              (row['recall'], row['precision']),
                              xytext=(5, 5), textcoords='offset points',
                              fontsize=8, alpha=0.7)
        
        axes[1, 0].set_title('Precision vs Recall', fontweight='bold')
        axes[1, 0].set_xlabel('Recall')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].set_xlim(0.8, 1.02)
        axes[1, 0].set_ylim(0.8, 1.02)
        
        # Add diagonal line (perfect balance)
        axes[1, 0].plot([0.8, 1.0], [0.8, 1.0], 'k--', alpha=0.3)
        
        # 4. ROC-AUC comparison (if available)
        if 'roc_auc' in all_results.columns:
            axes[1, 1].barh(all_results['model_name'], all_results['roc_auc'], color=colors)
            axes[1, 1].set_title('ROC-AUC Comparison', fontweight='bold')
            axes[1, 1].set_xlabel('ROC-AUC')
            axes[1, 1].set_xlim(0.9, 1.0)
            
            for i, v in enumerate(all_results['roc_auc']):
                axes[1, 1].text(v + 0.002, i, f'{v:.3f}', va='center', fontsize=9)
        else:
            # Alternative: Model category distribution
            category_counts = all_results['category'].value_counts()
            axes[1, 1].pie(category_counts.values, labels=category_counts.index, 
                          colors=[category_colors.get(cat, self.colors['neutral']) for cat in category_counts.index],
                          autopct='%1.0f%%')
            axes[1, 1].set_title('Model Categories Distribution', fontweight='bold')
        
        # Create legend
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, label=category) 
                          for category, color in category_colors.items()]
        fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
        
        plt.tight_layout()
        
        # Save figure
        if save_path is None:
            save_path = self.output_dir / "model_performance_comparison.png"
        
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
            save_path = self.output_dir / "attack_distribution_analysis.png"
        
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
            save_path = self.output_dir / "performance_summary_table"
        
        summary_table.to_csv(f"{save_path}.csv", index=False)
        
        # Generate LaTeX table
        latex_table = self._generate_latex_table(summary_table)
        with open(f"{save_path}.tex", 'w') as f:
            f.write(latex_table)
        
        print(f"📊 Performance summary table saved to {save_path}.*")
        
        return summary_table
    
    def _generate_latex_table(self, df: pd.DataFrame) -> str:
        """Generate LaTeX table code"""
        
        # Create column specification
        num_cols = len(df.columns)
        col_spec = '|' + 'c|' * num_cols
        
        latex_code = f"""
\\begin{{table}}[htbp]
\\centering
\\caption{{Machine Learning Models Performance Comparison on NSL-KDD Dataset}}
\\label{{tab:model_performance}}
\\begin{{tabular}}{{{col_spec}}}
\\hline
"""
        
        # Add header
        header = ' & '.join([f"\\textbf{{{col}}}" for col in df.columns]) + ' \\\\\n'
        latex_code += header
        latex_code += '\\hline\n'
        
        # Add rows
        for _, row in df.iterrows():
            row_str = ' & '.join([str(val) for val in row.values]) + ' \\\\\n'
            latex_code += row_str
        
        latex_code += """\\hline
\\end{tabular}
\\end{table}
"""
        
        return latex_code
    
    def generate_all_figures(self):
        """
        Generate all publication-ready figures and tables
        """
        print("🎨 Generating Publication-Ready Figures")
        print("=" * 50)
        
        try:
            # 1. Model performance comparison
            print("📊 Creating model performance comparison...")
            self.create_model_performance_comparison()
            
            # 2. Attack distribution analysis (NSL-KDD)
            print("📊 Creating NSL-KDD attack distribution analysis...")
            self.create_attack_distribution_analysis()
            
            # 3. CIC-IDS-2017 attack distribution analysis
            print("📊 Creating CIC-IDS-2017 attack distribution analysis...")
            self.create_cic_attack_distribution_analysis()
            
            # 4. Performance summary table
            print("📊 Creating performance summary table...")
            summary_table = self.create_performance_summary_table()
            
            if not summary_table.empty:
                print("\n📋 Performance Summary (Top 5):")
                print(summary_table.head().to_string(index=False))
            
            print(f"\n✅ All figures generated successfully!")
            print(f"📁 Output directory: {self.output_dir}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error generating figures: {e}")
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