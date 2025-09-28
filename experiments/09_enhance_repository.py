"""
Scientific Visualization Enhancement for Network Anomaly Detection Research
Generates publication-quality learning curves, performance analytics, and scientific visualizations

This experiment focuses ONLY on creating scientific visualizations from existing result files.
No model loading, no file manipulation, no memory-intensive operations.
"""
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
import json
import warnings
warnings.filterwarnings('ignore')

# Scientific publication settings with Arial/Sans-Serif fonts (user requirement)
plt.style.use('default')
sns.set_style("whitegrid", {'grid.linestyle': '-', 'grid.linewidth': 0.5, 'grid.alpha': 0.4})

# Publication-quality matplotlib settings
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
    'axes.grid.axis': 'both',
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.8
})


class ScientificVisualizationEnhancer:
    """
    Generate scientific visualizations for network anomaly detection research.
    
    This class focuses ONLY on reading existing CSV/JSON result files and creating
    publication-quality visualizations. No model loading or file manipulation.
    """
    
    def __init__(self, output_dir: str = "data/results/scientific_analysis"):
        """Initialize the scientific visualization enhancer"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create organized subdirectories
        self.subdirs = {
            'learning_curves': self.output_dir / 'learning_curves',
            'convergence_analysis': self.output_dir / 'convergence_analysis',
            'comparative_analysis': self.output_dir / 'comparative_analysis',
            'cross_dataset_analysis': self.output_dir / 'cross_dataset_analysis'
        }
        
        for subdir in self.subdirs.values():
            subdir.mkdir(parents=True, exist_ok=True)
        
        # Scientific color palette (colorblind-friendly)
        self.colors = {
            'nsl_baseline': '#1f77b4',     # Blue
            'nsl_advanced': '#ff7f0e',     # Orange
            'cic_baseline': '#2ca02c',     # Green
            'cic_advanced': '#d62728',     # Red
            'cross_dataset': '#9467bd',    # Purple
            'neutral': '#7f7f7f',          # Gray
            'accent1': '#8c564b',          # Brown
            'accent2': '#e377c2'           # Pink
        }
        
        # Dataset detection
        self.available_datasets = self._detect_available_datasets()
        
        print(f"📊 Scientific Visualization Enhancer initialized")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"🔍 Available datasets: {', '.join(self.available_datasets)}")
    
    def _detect_available_datasets(self) -> List[str]:
        """Detect which datasets have results available"""
        datasets = []
        
        # Check for NSL-KDD results
        nsl_paths = [
            "data/results/nsl_baseline_results.csv",
            "data/results/nsl_advanced_results.csv"
        ]
        if any(Path(path).exists() for path in nsl_paths):
            datasets.append('NSL-KDD')
        
        # Check for CIC-IDS-2017 results  
        cic_paths = [
            "data/results/cic_baseline_results.csv",
            "data/results/cic_advanced_results.csv"
        ]
        if any(Path(path).exists() for path in cic_paths):
            datasets.append('CIC-IDS-2017')
        
        return datasets
    
    def generate_learning_curves_analysis(self) -> bool:
        """
        Generate learning curves analysis from existing result files.
        Shows model performance vs complexity/training data size.
        """
        print("\n📈 Generating learning curves analysis...")
        
        try:
            # Load existing result files
            results_data = {}
            
            # Load NSL-KDD results
            if Path("data/results/nsl_baseline_results.csv").exists():
                results_data['NSL-KDD Baseline'] = pd.read_csv("data/results/nsl_baseline_results.csv")
                
            if Path("data/results/nsl_advanced_results.csv").exists():
                results_data['NSL-KDD Advanced'] = pd.read_csv("data/results/nsl_advanced_results.csv")
            
            # Load CIC-IDS-2017 results
            if Path("data/results/cic_baseline_results.csv").exists():
                results_data['CIC-IDS-2017 Baseline'] = pd.read_csv("data/results/cic_baseline_results.csv")
                
            if Path("data/results/cic_advanced_results.csv").exists():
                results_data['CIC-IDS-2017 Advanced'] = pd.read_csv("data/results/cic_advanced_results.csv")
            
            if not results_data:
                print("❌ No result files found. Run experiments 02-03 first.")
                return False
            
            # Create learning curves visualization
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            axes = axes.flatten()
            
            metrics = ['accuracy', 'f1_score', 'precision', 'recall']
            
            for i, metric in enumerate(metrics):
                ax = axes[i]
                
                for dataset_name, df in results_data.items():
                    if metric in df.columns:
                        # Use model names as x-axis (complexity proxy)
                        models = df['model'].unique() if 'model' in df.columns else range(len(df))
                        values = df[metric].values
                        
                        # Determine color based on dataset
                        if 'NSL-KDD' in dataset_name and 'Baseline' in dataset_name:
                            color = self.colors['nsl_baseline']
                        elif 'NSL-KDD' in dataset_name and 'Advanced' in dataset_name:
                            color = self.colors['nsl_advanced']
                        elif 'CIC' in dataset_name and 'Baseline' in dataset_name:
                            color = self.colors['cic_baseline']
                        elif 'CIC' in dataset_name and 'Advanced' in dataset_name:
                            color = self.colors['cic_advanced']
                        else:
                            color = self.colors['neutral']
                        
                        ax.plot(range(len(values)), values, 'o-', 
                               label=dataset_name, color=color, linewidth=1.5, markersize=5)
                
                ax.set_title(f'{metric.replace("_", " ").title()} Learning Curve')
                ax.set_xlabel('Model Complexity')
                ax.set_ylabel(metric.replace("_", " ").title())
                ax.legend(fontsize=7)
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save the figure
            output_path = self.subdirs['learning_curves'] / 'model_learning_curves.pdf'
            plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
            plt.close()
            
            print(f"✅ Learning curves saved: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error generating learning curves: {e}")
            return False
    
    def generate_convergence_analysis(self) -> bool:
        """
        Generate cross-validation convergence analysis.
        Shows statistical stability of model performance.
        """
        print("\n📊 Generating convergence analysis...")
        
        try:
            # Check for cross-validation results
            cv_json_path = Path("data/results/harmonized_cross_validation.json")
            cv_dir = Path("data/results/cross_validation")
            
            cv_data = None
            
            # Try to load harmonized CV results first
            if cv_json_path.exists():
                with open(cv_json_path, 'r') as f:
                    cv_data = json.load(f)
                print(f"✅ Loaded harmonized CV results: {len(cv_data)} entries")
            
            # Try to load individual CV files
            elif cv_dir.exists():
                cv_files = list(cv_dir.glob("*.csv"))
                if cv_files:
                    cv_data = {}
                    for cv_file in cv_files:
                        dataset_name = cv_file.stem.replace('_cv_results', '')
                        cv_data[dataset_name] = pd.read_csv(cv_file)
                    print(f"✅ Loaded {len(cv_files)} CV result files")
            
            if not cv_data:
                print("❌ No cross-validation results found. Run experiment 04 first.")
                return False
            
            # Create convergence analysis
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            axes = axes.flatten()
            
            if isinstance(cv_data, dict) and 'cross_validation_results' in cv_data:
                # Handle harmonized format
                results = cv_data['cross_validation_results']
                
                for i, (dataset_name, dataset_results) in enumerate(results.items()):
                    if i >= 4:  # Limit to 4 subplots
                        break
                    
                    ax = axes[i]
                    
                    for model_name, model_results in dataset_results.items():
                        if 'fold_scores' in model_results:
                            fold_scores = model_results['fold_scores']
                            
                            # Calculate cumulative mean (convergence)
                            cumulative_mean = np.cumsum(fold_scores) / np.arange(1, len(fold_scores) + 1)
                            
                            ax.plot(range(1, len(cumulative_mean) + 1), cumulative_mean, 
                                   'o-', label=model_name, linewidth=1.5, markersize=4)
                    
                    ax.set_title(f'{dataset_name} CV Convergence')
                    ax.set_xlabel('CV Fold')
                    ax.set_ylabel('Cumulative Mean F1-Score')
                    ax.legend(fontsize=7)
                    ax.grid(True, alpha=0.3)
            
            else:
                # Handle individual file format
                for i, (dataset_name, df) in enumerate(cv_data.items()):
                    if i >= 4:  # Limit to 4 subplots
                        break
                    
                    ax = axes[i]
                    
                    if 'f1_score' in df.columns:
                        # Assume each row is a fold result
                        fold_scores = df['f1_score'].values
                        cumulative_mean = np.cumsum(fold_scores) / np.arange(1, len(fold_scores) + 1)
                        
                        ax.plot(range(1, len(cumulative_mean) + 1), cumulative_mean, 
                               'o-', color=self.colors['primary'], linewidth=1.5, markersize=4)
                    
                    ax.set_title(f'{dataset_name} CV Convergence')
                    ax.set_xlabel('CV Fold')
                    ax.set_ylabel('Cumulative Mean F1-Score')
                    ax.grid(True, alpha=0.3)
            
            # Hide empty subplots
            for i in range(len(cv_data), 4):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            
            # Save the figure
            output_path = self.subdirs['convergence_analysis'] / 'cv_convergence_analysis.pdf'
            plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
            plt.close()
            
            print(f"✅ Convergence analysis saved: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error generating convergence analysis: {e}")
            return False
    
    def generate_cross_dataset_analysis(self) -> bool:
        """
        Generate cross-dataset transfer learning analysis.
        Shows how models trained on one dataset perform on another.
        """
        print("\n🔄 Generating cross-dataset analysis...")
        
        try:
            # Load bidirectional cross-dataset results
            cross_results_path = Path("data/results/bidirectional_cross_dataset_analysis.csv")
            nsl_on_cic_path = Path("data/results/nsl_trained_tested_on_cic.csv")
            cic_on_nsl_path = Path("data/results/cic_trained_tested_on_nsl.csv")
            
            cross_data = {}
            
            if cross_results_path.exists():
                cross_data['bidirectional'] = pd.read_csv(cross_results_path)
                
            if nsl_on_cic_path.exists():
                cross_data['nsl_on_cic'] = pd.read_csv(nsl_on_cic_path)
                
            if cic_on_nsl_path.exists():
                cross_data['cic_on_nsl'] = pd.read_csv(cic_on_nsl_path)
            
            if not cross_data:
                print("❌ No cross-dataset results found. Run experiment 05 first.")
                return False
            
            # Create cross-dataset visualization
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            axes = axes.flatten()
            
            # Plot 1: Transfer performance comparison
            ax1 = axes[0]
            if 'bidirectional' in cross_data:
                df = cross_data['bidirectional']
                if 'source_dataset' in df.columns and 'f1_score' in df.columns:
                    # Group by source dataset
                    grouped = df.groupby(['source_dataset', 'target_dataset'])['f1_score'].mean().unstack()
                    grouped.plot(kind='bar', ax=ax1, color=[self.colors['nsl_baseline'], self.colors['cic_baseline']])
                    ax1.set_title('Cross-Dataset Transfer Performance')
                    ax1.set_xlabel('Source Dataset')
                    ax1.set_ylabel('F1-Score')
                    ax1.legend(title='Target Dataset')
                    ax1.tick_params(axis='x', rotation=45)
            
            # Plot 2: Performance degradation analysis
            ax2 = axes[1]
            if 'nsl_on_cic' in cross_data and 'cic_on_nsl' in cross_data:
                # Calculate performance drops
                nsl_cic_f1 = cross_data['nsl_on_cic']['f1_score'].mean() if 'f1_score' in cross_data['nsl_on_cic'].columns else 0
                cic_nsl_f1 = cross_data['cic_on_nsl']['f1_score'].mean() if 'f1_score' in cross_data['cic_on_nsl'].columns else 0
                
                transfer_performance = [nsl_cic_f1, cic_nsl_f1]
                transfer_labels = ['NSL→CIC', 'CIC→NSL']
                
                bars = ax2.bar(transfer_labels, transfer_performance, 
                              color=[self.colors['cross_dataset'], self.colors['accent1']])
                ax2.set_title('Transfer Learning Performance')
                ax2.set_ylabel('F1-Score')
                
                # Add value labels on bars
                for bar, value in zip(bars, transfer_performance):
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{value:.3f}', ha='center', va='bottom')
            
            # Plot 3: Model comparison across datasets
            ax3 = axes[2]
            if 'bidirectional' in cross_data:
                df = cross_data['bidirectional']
                if 'model' in df.columns and 'f1_score' in df.columns:
                    model_performance = df.groupby('model')['f1_score'].mean().sort_values(ascending=True)
                    model_performance.plot(kind='barh', ax=ax3, color=self.colors['primary'])
                    ax3.set_title('Average Model Performance')
                    ax3.set_xlabel('F1-Score')
            
            # Plot 4: Dataset difficulty analysis
            ax4 = axes[3]
            if len(cross_data) >= 2:
                # Calculate average performance when each dataset is used as target
                target_scores = {}
                for name, df in cross_data.items():
                    if 'target_dataset' in df.columns and 'f1_score' in df.columns:
                        targets = df.groupby('target_dataset')['f1_score'].mean()
                        for target, score in targets.items():
                            if target not in target_scores:
                                target_scores[target] = []
                            target_scores[target].append(score)
                
                if target_scores:
                    avg_scores = {k: np.mean(v) for k, v in target_scores.items()}
                    datasets = list(avg_scores.keys())
                    scores = list(avg_scores.values())
                    
                    bars = ax4.bar(datasets, scores, 
                                  color=[self.colors['nsl_advanced'], self.colors['cic_advanced']])
                    ax4.set_title('Dataset Difficulty (as Target)')
                    ax4.set_ylabel('Average F1-Score')
                    
                    # Add value labels
                    for bar, value in zip(bars, scores):
                        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                                f'{value:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            # Save the figure
            output_path = self.subdirs['cross_dataset_analysis'] / 'cross_dataset_transfer_analysis.pdf'
            plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
            plt.close()
            
            print(f"✅ Cross-dataset analysis saved: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error generating cross-dataset analysis: {e}")
            return False
    
    def generate_comparative_analysis(self) -> bool:
        """
        Generate comparative analysis across all models and datasets.
        """
        print("\n📊 Generating comparative analysis...")
        
        try:
            # Load all available result files
            all_results = []
            
            # NSL-KDD results
            for result_type in ['baseline', 'advanced']:
                result_path = Path(f"data/results/nsl_{result_type}_results.csv")
                if result_path.exists():
                    df = pd.read_csv(result_path)
                    df['dataset'] = 'NSL-KDD'
                    df['type'] = result_type
                    all_results.append(df)
            
            # CIC-IDS-2017 results  
            for result_type in ['baseline', 'advanced']:
                result_path = Path(f"data/results/cic_{result_type}_results.csv")
                if result_path.exists():
                    df = pd.read_csv(result_path)
                    df['dataset'] = 'CIC-IDS-2017'
                    df['type'] = result_type
                    all_results.append(df)
            
            if not all_results:
                print("❌ No result files found for comparison")
                return False
            
            # Combine all results
            combined_df = pd.concat(all_results, ignore_index=True)
            
            # Create comprehensive comparison
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()
            
            metrics = ['accuracy', 'f1_score', 'precision', 'recall']
            
            # Plot each metric
            for i, metric in enumerate(metrics):
                if i < 4 and metric in combined_df.columns:
                    ax = axes[i]
                    
                    # Create grouped bar plot
                    grouped_data = combined_df.groupby(['dataset', 'type'])[metric].mean().unstack()
                    grouped_data.plot(kind='bar', ax=ax, 
                                     color=[self.colors['nsl_baseline'], self.colors['nsl_advanced']])
                    
                    ax.set_title(f'{metric.replace("_", " ").title()} Comparison')
                    ax.set_ylabel(metric.replace("_", " ").title())
                    ax.legend(title='Model Type')
                    ax.tick_params(axis='x', rotation=45)
                    ax.grid(True, alpha=0.3)
            
            # Overall performance heatmap
            if len(metrics) <= 4:
                ax5 = axes[4]
                
                # Create performance matrix
                perf_matrix = combined_df.groupby(['dataset', 'type'])[metrics[:4]].mean()
                
                if not perf_matrix.empty:
                    # Create heatmap
                    sns.heatmap(perf_matrix, annot=True, fmt='.3f', cmap='RdYlBu_r', ax=ax5)
                    ax5.set_title('Performance Heatmap')
            
            # Model ranking
            ax6 = axes[5]
            if 'model' in combined_df.columns and 'f1_score' in combined_df.columns:
                model_ranking = combined_df.groupby('model')['f1_score'].mean().sort_values(ascending=True)
                model_ranking.plot(kind='barh', ax=ax6, color=self.colors['primary'])
                ax6.set_title('Model Ranking (F1-Score)')
                ax6.set_xlabel('F1-Score')
            
            plt.tight_layout()
            
            # Save the figure
            output_path = self.subdirs['comparative_analysis'] / 'comprehensive_model_comparison.pdf'
            plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
            plt.close()
            
            print(f"✅ Comparative analysis saved: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error generating comparative analysis: {e}")
            return False
    
    def generate_scientific_report(self) -> str:
        """
        Generate comprehensive scientific visualization report.
        """
        report_path = self.output_dir / "scientific_visualization_report.md"
        
        with open(report_path, 'w') as f:
            f.write("# Scientific Visualization Report\n\n")
            f.write("Generated automatically by the Scientific Visualization Enhancer.\n\n")
            
            f.write("## Overview\n\n")
            f.write(f"- **Analysis timestamp**: {pd.Timestamp.now()}\n")
            f.write(f"- **Available datasets**: {', '.join(self.available_datasets)}\n")
            f.write(f"- **Output directory**: {self.output_dir}\n\n")
            
            f.write("## Generated Visualizations\n\n")
            
            # List generated files for each category
            for category, subdir in self.subdirs.items():
                f.write(f"### {category.replace('_', ' ').title()}\n\n")
                
                if subdir.exists():
                    pdf_files = list(subdir.glob("*.pdf"))
                    if pdf_files:
                        for pdf_file in sorted(pdf_files):
                            f.write(f"- `{pdf_file.name}`\n")
                    else:
                        f.write("- No visualizations generated\n")
                f.write("\n")
            
            f.write("## Key Features\n\n")
            f.write("✅ **Arial/Sans-Serif Fonts**: All visualizations use publication-standard typography\n")
            f.write("✅ **300 DPI PDF Output**: High-quality vector graphics for print\n")
            f.write("✅ **Colorblind-Friendly Palette**: Accessible scientific color scheme\n")
            f.write("✅ **IEEE/ACM Journal Format**: Optimized for 2-column layout\n")
            f.write("✅ **Safe File Operations**: Only reads existing results, no file manipulation\n\n")
            
            f.write("## Visualization Types\n\n")
            f.write("1. **Learning Curves**: Model performance vs complexity analysis\n")
            f.write("2. **Convergence Analysis**: Cross-validation statistical stability\n")
            f.write("3. **Cross-Dataset Analysis**: Transfer learning performance\n")
            f.write("4. **Comparative Analysis**: Comprehensive model comparison\n\n")
            
            f.write("## Usage in Publications\n\n")
            f.write("All generated PDFs are publication-ready and can be directly included in:\n")
            f.write("- IEEE conference papers\n")
            f.write("- ACM journal submissions\n")
            f.write("- Springer/Elsevier publications\n")
            f.write("- Academic presentations\n\n")
            
            f.write("## File Organization\n\n")
            f.write("```\n")
            f.write("data/results/scientific_analysis/\n")
            f.write("├── learning_curves/         # Model performance analysis\n")
            f.write("├── convergence_analysis/    # CV stability analysis\n") 
            f.write("├── cross_dataset_analysis/  # Transfer learning analysis\n")
            f.write("├── comparative_analysis/    # Multi-model comparison\n")
            f.write("└── scientific_visualization_report.md\n")
            f.write("```\n\n")
        
        print(f"📄 Scientific report saved: {report_path}")
        return str(report_path)
    
    def run_all_analyses(self) -> bool:
        """
        Run all scientific visualization analyses.
        """
        print("\n🚀 Running all scientific visualization analyses...")
        
        success_count = 0
        total_analyses = 4
        
        # Run each analysis
        if self.generate_learning_curves_analysis():
            success_count += 1
        
        if self.generate_convergence_analysis():
            success_count += 1
            
        if self.generate_cross_dataset_analysis():
            success_count += 1
            
        if self.generate_comparative_analysis():
            success_count += 1
        
        # Generate report
        self.generate_scientific_report()
        
        print(f"\n📊 Analysis Summary:")
        print(f"✅ Successful analyses: {success_count}/{total_analyses}")
        print(f"📁 All outputs saved to: {self.output_dir}")
        
        return success_count == total_analyses


def main():
    """
    Main function to generate scientific visualizations.
    """
    print("🔬 SCIENTIFIC VISUALIZATION ENHANCEMENT")
    print("=" * 60)
    print("Generating publication-quality scientific visualizations from existing results...")
    
    try:
        # Initialize enhancer
        enhancer = ScientificVisualizationEnhancer()
        
        # Check if we have any datasets
        if not enhancer.available_datasets:
            print("❌ No datasets found. Run experiments 02-03 first to generate results.")
            return False
        
        # Run all analyses
        success = enhancer.run_all_analyses()
        
        if success:
            print("\n🎉 SCIENTIFIC VISUALIZATION COMPLETE!")
            print("=" * 60)
            print("✅ All visualizations generated successfully")
            print("✅ Publication-quality PDF outputs created")
            print("✅ Arial/Sans-Serif fonts applied")
            print("✅ 300 DPI resolution for print quality")
            print("✅ Colorblind-friendly color palette used")
            print(f"✅ Results organized in: {enhancer.output_dir}")
            
            print("\n📊 READY FOR PUBLICATION!")
            print("All generated PDFs can be directly used in:")
            print("• IEEE/ACM conference papers")
            print("• Journal submissions")
            print("• Academic presentations")
        else:
            print("\n⚠️ Some analyses failed")
            print("Check the logs above for details.")
        
        return success
        
    except Exception as e:
        print(f"\n❌ Error during scientific visualization: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)