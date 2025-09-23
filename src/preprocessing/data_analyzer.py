# src/nsl_kdd_analyzer.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import warnings
import os
from pathlib import Path
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

class NSLKDDAnalyzer:
    """
    Comprehensive analyzer for NSL-KDD dataset
    Designed for the intrusion detection research project
    """
    
    def __init__(self, data_dir="data/raw/nsl-kdd", output_dir="data/results"):
        """
        Initialize the analyzer with project directory structure
        
        Args:
            data_dir (str): Path to raw data directory
            output_dir (str): Path to results directory
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define the 41 feature names based on NSL-KDD documentation
        self.feature_names = [
            # Basic features (1-9)
            'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 
            'dst_bytes', 'land', 'wrong_fragment', 'urgent',
            
            # Content features (10-22)
            'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 
            'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 
            'num_shells', 'num_access_files', 'num_outbound_cmds', 
            'is_hot_login', 'is_guest_login',
            
            # Time-based traffic features (23-31)
            'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 
            'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 
            'diff_srv_rate', 'srv_diff_host_rate',
            
            # Host-based traffic features (32-41)
            'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
            'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 
            'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 
            'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 
            'dst_host_srv_rerror_rate',
            
            # Labels
            'attack_type', 'difficulty_level'
        ]
        
        # Feature categories
        self.feature_categories = {
            'basic': list(range(0, 9)),
            'content': list(range(9, 22)),
            'time_based': list(range(22, 31)),
            'host_based': list(range(31, 41))
        }
        
        # Attack type to category mapping
        self.attack_categories = {
            'normal': 'Normal',
            # DoS attacks
            'back': 'DoS', 'land': 'DoS', 'neptune': 'DoS', 'pod': 'DoS', 
            'smurf': 'DoS', 'teardrop': 'DoS', 'mailbomb': 'DoS', 
            'apache2': 'DoS', 'processtable': 'DoS', 'udpstorm': 'DoS',
            # Probe attacks
            'ipsweep': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe', 
            'satan': 'Probe', 'mscan': 'Probe', 'saint': 'Probe',
            # R2L attacks
            'ftp_write': 'R2L', 'guess_passwd': 'R2L', 'imap': 'R2L', 
            'multihop': 'R2L', 'phf': 'R2L', 'spy': 'R2L', 
            'warezclient': 'R2L', 'warezmaster': 'R2L', 'sendmail': 'R2L', 
            'named': 'R2L', 'snmpgetattack': 'R2L', 'snmpguess': 'R2L', 
            'xlock': 'R2L', 'xsnoop': 'R2L', 'worm': 'R2L',
            # U2R attacks
            'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'perl': 'U2R', 
            'rootkit': 'U2R', 'httptunnel': 'U2R', 'ps': 'U2R', 
            'sqlattack': 'U2R'
        }
        
    def load_data(self, filename):
        """
        Load NSL-KDD data from file in the data/raw directory
        
        Args:
            filename (str): Name of the file (e.g., 'KDDTrain+.txt')
            
        Returns:
            pd.DataFrame: Loaded and processed data
        """
        file_path = self.data_dir / filename
        
        try:
            # Load data without headers (CSV format)
            data = pd.read_csv(file_path, names=self.feature_names, header=None)
            
            # Add attack category column
            data['attack_category'] = data['attack_type'].map(self.attack_categories)
            
            print(f"✓ Successfully loaded data from {filename}")
            print(f"  Shape: {data.shape}")
            print(f"  Features: {len(self.feature_names)} (41 features + 2 labels)")
            
            return data
            
        except Exception as e:
            print(f"✗ Error loading data from {filename}: {e}")
            return None
    
    def save_processed_data(self, data, filename_prefix):
        """
        Save processed data to the processed directory
        
        Args:
            data (pd.DataFrame): Processed data
            filename_prefix (str): Prefix for the saved file
        """
        processed_dir = Path("data/processed")
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as CSV
        csv_path = processed_dir / f"{filename_prefix}_processed.csv"
        data.to_csv(csv_path, index=False)
        
        # Save as pickle for faster loading
        pkl_path = processed_dir / f"{filename_prefix}_processed.pkl"
        data.to_pickle(pkl_path)
        
        print(f"✓ Saved processed data to:")
        print(f"  CSV: {csv_path}")
        print(f"  Pickle: {pkl_path}")
    
    def basic_info(self, data):
        """
        Display basic information about the dataset
        """
        print("\n" + "="*60)
        print("BASIC DATASET INFORMATION")
        print("="*60)
        
        print(f"Dataset Shape: {data.shape}")
        print(f"Total Records: {len(data):,}")
        print(f"Total Features: {data.shape[1]-2} (excluding labels)")
        print(f"Memory Usage: {data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Check for missing values
        missing_values = data.isnull().sum()
        if missing_values.sum() > 0:
            print(f"\nMissing Values: {missing_values.sum()}")
            print(missing_values[missing_values > 0])
        else:
            print("\n✓ No missing values found")
        
        # Data types
        print(f"\nData Types:")
        dtype_counts = data.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            print(f"  {dtype}: {count} columns")
            
    def analyze_features(self, data):
        """
        Analyze feature types and characteristics
        """
        print("\n" + "="*60)
        print("FEATURE ANALYSIS")
        print("="*60)
        
        numeric_features = []
        categorical_features = []
        
        # Separate features by type (excluding labels)
        for i, col in enumerate(data.columns[:-2]):  # Exclude attack_type and difficulty_level
            if data[col].dtype in ['int64', 'float64']:
                # Check if it's binary/categorical numeric
                unique_vals = data[col].nunique()
                if unique_vals <= 10:
                    categorical_features.append(col)
                else:
                    numeric_features.append(col)
            else:
                categorical_features.append(col)
        
        print(f"Numeric Features: {len(numeric_features)}")
        print(f"Categorical Features: {len(categorical_features)}")
        
        # Feature categories breakdown
        print(f"\nFeature Categories:")
        for category, indices in self.feature_categories.items():
            features = [self.feature_names[i] for i in indices]
            print(f"  {category.title()}: {len(features)} features")
            print(f"    {', '.join(features[:5])}{'...' if len(features) > 5 else ''}")
        
        # Analyze categorical features
        print(f"\nCategorical Feature Details:")
        categorical_data = data[categorical_features]
        for col in categorical_data.columns:
            unique_count = data[col].nunique()
            print(f"  {col}: {unique_count} unique values")
            if unique_count <= 10:
                print(f"    Values: {list(data[col].unique())}")
                
        return numeric_features, categorical_features
    
    def analyze_class_distribution(self, data):
        """
        Analyze attack types and categories distribution
        """
        print("\n" + "="*60)
        print("CLASS DISTRIBUTION ANALYSIS")
        print("="*60)
        
        # Attack type distribution
        attack_type_dist = data['attack_type'].value_counts()
        print(f"Attack Types Distribution:")
        print(f"Total unique attack types: {len(attack_type_dist)}")
        
        # Show top 10 most frequent
        print(f"\nTop 10 Most Frequent Attack Types:")
        for i, (attack, count) in enumerate(attack_type_dist.head(10).items(), 1):
            percentage = (count / len(data)) * 100
            print(f"  {i:2}. {attack:<15} {count:>8,} ({percentage:5.2f}%)")
        
        # Attack category distribution
        attack_cat_dist = data['attack_category'].value_counts()
        print(f"\nAttack Categories Distribution:")
        for category, count in attack_cat_dist.items():
            percentage = (count / len(data)) * 100
            print(f"  {category:<8} {count:>8,} ({percentage:5.2f}%)")
        
        # Difficulty level analysis
        if 'difficulty_level' in data.columns:
            diff_dist = data['difficulty_level'].value_counts().sort_index()
            print(f"\nDifficulty Level Distribution:")
            print(f"  Range: {data['difficulty_level'].min()} - {data['difficulty_level'].max()}")
            print(f"  Most common: Level {diff_dist.idxmax()} ({diff_dist.max():,} records)")
            
        return attack_type_dist, attack_cat_dist
    
    def statistical_summary(self, data, numeric_features):
        """
        Generate statistical summary for numeric features
        """
        print("\n" + "="*60)
        print("STATISTICAL SUMMARY")
        print("="*60)
        
        numeric_data = data[numeric_features]
        
        print("Numeric Features Statistics:")
        summary = numeric_data.describe()
        
        # Display summary in a more readable format
        for col in numeric_data.columns[:5]:  # Show first 5 features
            print(f"\n{col}:")
            print(f"  Mean: {summary.loc['mean', col]:.4f}")
            print(f"  Std:  {summary.loc['std', col]:.4f}")
            print(f"  Min:  {summary.loc['min', col]:.4f}")
            print(f"  Max:  {summary.loc['max', col]:.4f}")
            
        # Zero variance features
        zero_var_features = numeric_data.columns[numeric_data.var() == 0].tolist()
        if zero_var_features:
            print(f"\n⚠️  Zero Variance Features: {len(zero_var_features)}")
            print(f"   {', '.join(zero_var_features)}")
        
        # High correlation features
        correlation_matrix = numeric_data.corr()
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_val = correlation_matrix.iloc[i, j]
                if abs(corr_val) > 0.9:
                    high_corr_pairs.append((
                        correlation_matrix.columns[i],
                        correlation_matrix.columns[j],
                        corr_val
                    ))
        
        if high_corr_pairs:
            print(f"\n⚠️  Highly Correlated Features (|r| > 0.9): {len(high_corr_pairs)} pairs")
            for feat1, feat2, corr in high_corr_pairs[:5]:
                print(f"   {feat1} - {feat2}: {corr:.3f}")
                
        return summary, high_corr_pairs
                
    def create_visualizations(self, data, attack_type_dist, attack_cat_dist, filename_prefix="analysis"):
        """
        Create comprehensive visualizations and save to results directory
        """
        print("\n" + "="*60)
        print("CREATING VISUALIZATIONS")
        print("="*60)
        
        # Set up the plotting
        plt.figure(figsize=(20, 15))
        
        # 1. Attack Category Distribution (Pie Chart)
        plt.subplot(2, 3, 1)
        attack_cat_dist.plot(kind='pie', autopct='%1.1f%%', startangle=90)
        plt.title('Attack Categories Distribution', fontsize=14, fontweight='bold')
        plt.ylabel('')
        
        # 2. Attack Category Distribution (Bar Chart)
        plt.subplot(2, 3, 2)
        attack_cat_dist.plot(kind='bar', color='skyblue', alpha=0.7)
        plt.title('Attack Categories Count', fontsize=14, fontweight='bold')
        plt.xlabel('Attack Category')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        # 3. Top 15 Attack Types
        plt.subplot(2, 3, 3)
        attack_type_dist.head(15).plot(kind='bar', color='lightcoral', alpha=0.7)
        plt.title('Top 15 Attack Types', fontsize=14, fontweight='bold')
        plt.xlabel('Attack Type')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        # 4. Difficulty Level Distribution
        plt.subplot(2, 3, 4)
        if 'difficulty_level' in data.columns:
            difficulty_dist = data['difficulty_level'].value_counts().sort_index()
            difficulty_dist.plot(kind='bar', color='lightgreen', alpha=0.7)
            plt.title('Difficulty Level Distribution', fontsize=14, fontweight='bold')
            plt.xlabel('Difficulty Level')
            plt.ylabel('Count')
        
        # 5. Protocol Type Distribution
        plt.subplot(2, 3, 5)
        if 'protocol_type' in data.columns:
            protocol_dist = data['protocol_type'].value_counts()
            protocol_dist.plot(kind='bar', color='orange', alpha=0.7)
            plt.title('Protocol Type Distribution', fontsize=14, fontweight='bold')
            plt.xlabel('Protocol Type')
            plt.ylabel('Count')
        
        # 6. Service Distribution (Top 10)
        plt.subplot(2, 3, 6)
        if 'service' in data.columns:
            service_dist = data['service'].value_counts().head(10)
            service_dist.plot(kind='bar', color='purple', alpha=0.7)
            plt.title('Top 10 Services', fontsize=14, fontweight='bold')
            plt.xlabel('Service')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = self.output_dir / f'{filename_prefix}_overview.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✓ Visualizations saved to: {plot_path}")
    
    def comprehensive_analysis(self, filename, save_processed=True):
        """
        Run complete analysis pipeline for a specific file
        
        Args:
            filename (str): Name of the NSL-KDD file to analyze
            save_processed (bool): Whether to save processed data
            
        Returns:
            pd.DataFrame: Processed data
        """
        print(f"🔍 Starting Comprehensive Analysis for {filename}")
        print("="*60)
        
        # Load data
        data = self.load_data(filename)
        if data is None:
            return None
        
        # Basic information
        self.basic_info(data)
        
        # Feature analysis
        numeric_features, categorical_features = self.analyze_features(data)
        
        # Class distribution
        attack_type_dist, attack_cat_dist = self.analyze_class_distribution(data)
        
        # Statistical summary
        summary, high_corr_pairs = self.statistical_summary(data, numeric_features)
        
        # Create visualizations
        filename_prefix = filename.replace('.txt', '').replace('+', '_plus')
        self.create_visualizations(data, attack_type_dist, attack_cat_dist, filename_prefix)
        
        # Save processed data if requested
        if save_processed:
            self.save_processed_data(data, filename_prefix)
        
        print("\n" + "="*60)
        print("✅ ANALYSIS COMPLETE")
        print("="*60)
        
        return data
    
    def compare_datasets(self, train_data, test_data):
        """
        Compare training and test datasets
        """
        print("\n" + "="*60)
        print("DATASET COMPARISON")
        print("="*60)
        
        print(f"Training Set:")
        print(f"  Records: {len(train_data):,}")
        print(f"  Attack types: {train_data['attack_type'].nunique()}")
        
        print(f"\nTest Set:")
        print(f"  Records: {len(test_data):,}")
        print(f"  Attack types: {test_data['attack_type'].nunique()}")
        
        # Find novel attacks in test set
        train_attacks = set(train_data['attack_type'].unique())
        test_attacks = set(test_data['attack_type'].unique())
        novel_attacks = test_attacks - train_attacks
        
        if novel_attacks:
            print(f"\n⚠️  Novel attacks in test set (not in training): {len(novel_attacks)}")
            for attack in sorted(novel_attacks):
                count = (test_data['attack_type'] == attack).sum()
                print(f"   {attack}: {count} records")
        
        return novel_attacks

# Utility functions for the project
def setup_project_directories():
    """
    Ensure all necessary directories exist
    """
    directories = [
        "data/raw/nsl-kdd",
        "data/raw/cic-ids-2017",
        "data/processed",
        "data/models",
        "data/results",
        "docs/figures",
        "src/features",
        "src/preprocessing",
        "src/metrics",
        "src/visualization",
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        
    print("✓ Project directories set up successfully")

if __name__ == "__main__":
    # Example usage
    setup_project_directories()
    
    analyzer = NSLKDDAnalyzer()
    
    print("🚀 NSL-KDD Analyzer initialized!")
    print("Available files:")
    for file in analyzer.data_dir.glob("*.txt"):
        print(f"  - {file.name}")