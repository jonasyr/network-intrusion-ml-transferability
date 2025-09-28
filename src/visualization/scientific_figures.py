# src/visualization/scientific_figures.py
"""
Publication-ready scientific figures for network anomaly detection research
High-quality visualizations following academic publication standards
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
from scipy import stats
from scipy.stats import pearsonr
import json

warnings.filterwarnings('ignore')

# Scientific publication settings (IEEE/ACM standards)
plt.style.use('default')
sns.set_style("whitegrid", {'grid.linestyle': '-', 'grid.linewidth': 0.5, 'grid.alpha': 0.4})

plt.rcParams.update({
    # Typography - Arial/Sans-Serif fonts as required
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica', 'sans-serif'],
    'mathtext.fontset': 'dejavusans',
    
    # Font sizes (optimized for 2-column journal format)
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.titlesize': 11,
    
    # Quality and layout
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.format': 'pdf',
    'savefig.bbox': 'tight',
    'figure.constrained_layout.use': True,
    
    # Professional appearance
    'lines.linewidth': 1.2,
    'lines.markersize': 4,
    'patch.linewidth': 0.8,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.8
})

class ScientificFigureGenerator:
    """Generate publication-quality scientific figures"""
    
    def __init__(self, output_dir: str = "data/results/paper_figures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Scientific color palette (colorblind-friendly)
        self.colors = {
            'primary': '#1f77b4',    # Blue
            'secondary': '#ff7f0e',  # Orange  
            'accent': '#2ca02c',     # Green
            'warning': '#d62728',    # Red
            'neutral': '#7f7f7f',    # Gray
            'purple': '#9467bd',     # Purple
            'brown': '#8c564b',      # Brown
            'pink': '#e377c2'        # Pink
        }
        
        self.qual_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
                           '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    
    def load_all_results(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load baseline and advanced results from both datasets"""
        datasets = ['nsl', 'cic']
        all_baseline = []
        all_advanced = []
        
        for dataset in datasets:
            # Load baseline
            baseline_path = f"data/results/{dataset}_baseline_results.csv"
            if Path(baseline_path).exists():
                df = pd.read_csv(baseline_path)
                df['dataset'] = dataset.upper()
                all_baseline.append(df)
            
            # Load advanced
            advanced_path = f"data/results/{dataset}_advanced_results.csv"
            if Path(advanced_path).exists():
                df = pd.read_csv(advanced_path)
                df['dataset'] = dataset.upper()
                all_advanced.append(df)
        
        combined_baseline = pd.concat(all_baseline, ignore_index=True) if all_baseline else pd.DataFrame()
        combined_advanced = pd.concat(all_advanced, ignore_index=True) if all_advanced else pd.DataFrame()
        
        return combined_baseline, combined_advanced
    
    def load_cross_validation_results(self) -> Dict[str, pd.DataFrame]:
        """Load cross-validation results"""
        cv_results = {}
        
        # NSL-KDD CV
        nsl_path = "data/results/cross_validation/cv_summary_table.csv"
        if Path(nsl_path).exists():
            cv_results['NSL-KDD'] = pd.read_csv(nsl_path)
        
        # CIC-IDS-2017 CV
        cic_path = "data/results/cross_validation/cic/cv_summary_table.csv"
        if Path(cic_path).exists():
            cv_results['CIC-IDS-2017'] = pd.read_csv(cic_path)
        
        return cv_results
    
    def load_cross_dataset_results(self) -> Dict[str, pd.DataFrame]:
        """Load cross-dataset transfer results"""
        results = {}
        
        paths = {
            'NSL→CIC': 'data/results/nsl_trained_tested_on_cic.csv',
            'CIC→NSL': 'data/results/cic_trained_tested_on_nsl.csv',
            'Bidirectional': 'data/results/bidirectional_cross_dataset_analysis.csv'
        }
        
        for name, path in paths.items():
            if Path(path).exists():
                results[name] = pd.read_csv(path)
        
        return results
    
    def create_figure1_performance_overview(self) -> plt.Figure:
        """Figure 1: Comprehensive Performance Overview"""
        baseline_results, advanced_results = self.load_all_results()
        
        if baseline_results.empty and advanced_results.empty:
            print("❌ No performance results available")
            return None
        
        # Combine results
        all_results = pd.concat([baseline_results, advanced_results], ignore_index=True)
        if all_results.empty:
            return None
        
        # Create figure (IEEE 2-column format)
        fig = plt.figure(figsize=(7.16, 5.4))
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.1], hspace=0.35, wspace=0.25)
        
        # Check if we have multiple datasets
        has_datasets = 'dataset' in all_results.columns and len(all_results['dataset'].unique()) > 1
        
        if has_datasets:
            fig.suptitle('Model Performance Comparison Across Datasets', fontweight='bold', y=0.98)
            datasets = all_results['dataset'].unique()
        else:
            fig.suptitle('Model Performance Analysis', fontweight='bold', y=0.98)
            datasets = ['Combined']
            all_results['dataset'] = 'Combined'
        
        # (a) F1-Score Performance Ranking
        ax1 = fig.add_subplot(gs[0, :])
        
        if has_datasets:
            # Multi-dataset bar chart
            x_pos = 0
            positions = []
            labels = []
            values = []
            colors = []
            
            for i, dataset in enumerate(datasets):
                dataset_data = all_results[all_results['dataset'] == dataset]
                if dataset_data.empty:
                    continue
                    
                # Top 5 models per dataset
                top_models = dataset_data.nlargest(5, 'f1_score')
                
                for _, model in top_models.iterrows():
                    positions.append(x_pos)
                    labels.append(f"{model['model_name']}\n({dataset})")
                    values.append(model['f1_score'])
                    colors.append(self.qual_colors[i])
                    x_pos += 1
                    
                x_pos += 0.5  # Space between datasets
            
            bars = ax1.bar(positions, values, color=colors, alpha=0.8, width=0.8)
            ax1.set_xticks(positions)
            ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
            
        else:
            # Single dataset
            top_models = all_results.nlargest(8, 'f1_score')
            bars = ax1.bar(range(len(top_models)), top_models['f1_score'], 
                          color=self.colors['primary'], alpha=0.8)
            ax1.set_xticks(range(len(top_models)))
            ax1.set_xticklabels(top_models['model_name'], rotation=45, ha='right')
            values = top_models['f1_score']
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=7, fontweight='bold')
        
        ax1.set_ylabel('F1-Score')
        ax1.set_title('(a) Model Performance Ranking', loc='left', fontweight='bold', pad=10)
        ax1.set_ylim(0.85, 1.02)
        ax1.grid(axis='y', alpha=0.3)
        
        # (b) Precision-Recall Analysis
        ax2 = fig.add_subplot(gs[1, 0])
        
        if has_datasets:
            for i, dataset in enumerate(datasets):
                dataset_data = all_results[all_results['dataset'] == dataset]
                if dataset_data.empty:
                    continue
                ax2.scatter(dataset_data['recall'], dataset_data['precision'], 
                           c=self.qual_colors[i], label=dataset, s=30, alpha=0.7, 
                           edgecolors='black', linewidths=0.5)
        else:
            ax2.scatter(all_results['recall'], all_results['precision'], 
                       c=self.colors['primary'], s=30, alpha=0.7, 
                       edgecolors='black', linewidths=0.5)
        
        # Perfect balance line
        lims = [0.85, 1.0]
        ax2.plot(lims, lims, 'k--', alpha=0.4, linewidth=1, label='Perfect Balance')
        
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('(b) Precision-Recall Trade-off', loc='left', fontweight='bold')
        ax2.set_xlim(0.85, 1.02)
        ax2.set_ylim(0.85, 1.02)
        ax2.grid(alpha=0.3)
        if has_datasets:
            ax2.legend(loc='lower right', frameon=True, fancybox=False)
        
        # (c) Performance Distribution
        ax3 = fig.add_subplot(gs[1, 1])
        
        if has_datasets and len(datasets) == 2:
            # Box plot comparison
            data_by_dataset = [all_results[all_results['dataset'] == d]['f1_score'] for d in datasets]
            bp = ax3.boxplot(data_by_dataset, labels=datasets, patch_artist=True,
                           boxprops=dict(facecolor=self.colors['primary'], alpha=0.6),
                           medianprops=dict(color='black', linewidth=1.5))
            ax3.set_ylabel('F1-Score')
            ax3.set_title('(c) Performance Distribution', loc='left', fontweight='bold')
        else:
            # Accuracy vs F1 correlation
            if len(all_results) > 2:
                corr, _ = pearsonr(all_results['accuracy'], all_results['f1_score'])
                ax3.scatter(all_results['accuracy'], all_results['f1_score'], 
                           c=self.colors['accent'], s=30, alpha=0.7, 
                           edgecolors='black', linewidths=0.5)
                
                # Correlation line
                z = np.polyfit(all_results['accuracy'], all_results['f1_score'], 1)
                p = np.poly1d(z)
                ax3.plot(all_results['accuracy'], p(all_results['accuracy']), 
                        'r--', alpha=0.6, linewidth=1.5)
                
                ax3.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax3.transAxes, 
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                        fontsize=8, fontweight='bold')
            
            ax3.set_xlabel('Accuracy')
            ax3.set_ylabel('F1-Score')
            ax3.set_title('(c) Accuracy-F1 Correlation', loc='left', fontweight='bold')
        
        ax3.grid(alpha=0.3)
        
        # Save figure
        plt.savefig(self.output_dir / "figure1_performance_overview.pdf", 
                   dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / "figure1_performance_overview.png", 
                   dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_figure2_statistical_analysis(self) -> plt.Figure:
        """Figure 2: Statistical Significance Analysis"""
        cv_results = self.load_cross_validation_results()
        
        if not cv_results:
            print("❌ No cross-validation results available")
            return None
        
        # Create figure
        fig = plt.figure(figsize=(7.16, 4.0))
        
        if len(cv_results) == 2:
            gs = fig.add_gridspec(1, 2, wspace=0.25)
        else:
            gs = fig.add_gridspec(1, 1)
        
        fig.suptitle('Statistical Analysis with Confidence Intervals', fontweight='bold', y=0.95)
        
        for i, (dataset, df) in enumerate(cv_results.items()):
            if df.empty or i >= 2:
                continue
                
            ax = fig.add_subplot(gs[0, i] if len(cv_results) > 1 else gs[0])
            
            # Parse accuracy and confidence intervals
            acc_means = []
            ci_lower = []
            ci_upper = []
            
            for _, row in df.iterrows():
                # Parse accuracy mean
                acc_str = str(row['Accuracy'])
                if '±' in acc_str:
                    mean_val = float(acc_str.split(' ±')[0])
                else:
                    mean_val = float(acc_str)
                acc_means.append(mean_val)
                
                # Parse confidence interval
                if 'Accuracy_CI' in row and pd.notna(row['Accuracy_CI']):
                    try:
                        ci_str = str(row['Accuracy_CI']).strip('[]"')
                        lower, upper = map(float, ci_str.split(', '))
                        ci_lower.append(lower)
                        ci_upper.append(upper)
                    except:
                        # Fallback to mean ± 2*std approximation
                        if '±' in acc_str:
                            std_val = float(acc_str.split(' ±')[1])
                            ci_lower.append(mean_val - 1.96*std_val)
                            ci_upper.append(mean_val + 1.96*std_val)
                        else:
                            ci_lower.append(mean_val)
                            ci_upper.append(mean_val)
                else:
                    ci_lower.append(mean_val)
                    ci_upper.append(mean_val)
            
            # Sort by performance and take top models
            sort_idx = np.argsort(acc_means)[::-1]
            top_n = min(8, len(df))
            
            top_indices = sort_idx[:top_n]
            top_models = df.iloc[top_indices]
            top_means = [acc_means[j] for j in top_indices]
            top_lower = [ci_lower[j] for j in top_indices]
            top_upper = [ci_upper[j] for j in top_indices]
            
            # Calculate error bars
            err_lower = [mean - lower for mean, lower in zip(top_means, top_lower)]
            err_upper = [upper - mean for mean, upper in zip(top_means, top_upper)]
            
            # Create horizontal bar chart with error bars
            y_pos = range(len(top_models))
            bars = ax.barh(y_pos, top_means, 
                          xerr=[err_lower, err_upper],
                          color=self.qual_colors[i % len(self.qual_colors)],
                          alpha=0.7, capsize=3, 
                          error_kw={'linewidth': 1.2, 'capthick': 1.2})
            
            # Labels and formatting
            ax.set_yticks(y_pos)
            model_labels = [model.replace('_', ' ').title() for model in top_models['Model']]
            ax.set_yticklabels(model_labels, fontsize=8)
            ax.set_xlabel('Accuracy')
            
            title_prefix = f'({chr(97+i)}) ' if len(cv_results) > 1 else ''
            ax.set_title(f'{title_prefix}{dataset}\n95% Confidence Intervals', 
                        loc='left', fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            
            # Add significance markers and precise values
            for j, (mean, lower, upper) in enumerate(zip(top_means, top_lower, top_upper)):
                if j == 0:  # Best model
                    ax.text(0.02, j, '★', transform=ax.get_yaxis_transform(),
                           ha='left', va='center', fontsize=10, color='gold', weight='bold')
                
                # Add precise value
                ax.text(mean + (upper-lower)/6, j, f'{mean:.4f}',
                       ha='left', va='center', fontsize=7, weight='bold')
        
        # Save figure
        plt.savefig(self.output_dir / "figure2_statistical_analysis.pdf", 
                   dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / "figure2_statistical_analysis.png", 
                   dpi=300, bbox_inches='tight')
        
        return fig
    
    def generate_all_scientific_figures(self) -> bool:
        """Generate all publication-quality figures"""
        print("🎨 GENERATING SCIENTIFIC PUBLICATION FIGURES")
        print("=" * 60)
        
        success_count = 0
        total_figures = 2
        
        # Figure 1: Performance Overview
        print("📊 Creating Figure 1: Performance Overview...")
        try:
            fig1 = self.create_figure1_performance_overview()
            if fig1 is not None:
                success_count += 1
                print("   ✅ Figure 1 completed")
                plt.close(fig1)
            else:
                print("   ⚠️ Figure 1 skipped - no data")
        except Exception as e:
            print(f"   ❌ Figure 1 failed: {e}")
        
        # Figure 2: Statistical Analysis
        print("📊 Creating Figure 2: Statistical Analysis...")
        try:
            fig2 = self.create_figure2_statistical_analysis()
            if fig2 is not None:
                success_count += 1
                print("   ✅ Figure 2 completed")
                plt.close(fig2)
            else:
                print("   ⚠️ Figure 2 skipped - no data")
        except Exception as e:
            print(f"   ❌ Figure 2 failed: {e}")
        
        print(f"\n📊 GENERATION COMPLETE")
        print(f"✅ Successfully generated: {success_count}/{total_figures} figures")
        print(f"📁 Output directory: {self.output_dir}")
        
        # List generated files
        if self.output_dir.exists():
            files = list(self.output_dir.glob("figure*.pdf"))
            if files:
                print("\n📄 Generated scientific figures:")
                for file in sorted(files):
                    print(f"   📄 {file.name}")
        
        return success_count > 0

def main():
    """Generate all scientific figures"""
    generator = ScientificFigureGenerator()
    return generator.generate_all_scientific_figures()

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)