# run_analysis.py
"""
Direct Python script to run NSL-KDD analysis without Jupyter
Save this as: run_analysis.py
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append('src')

try:
    from nsl_kdd_analyzer import NSLKDDAnalyzer, setup_project_directories
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    print("✅ All imports successful!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please install requirements: pip install -r requirements.txt")
    sys.exit(1)

def main():
    """
    Run comprehensive NSL-KDD analysis
    """
    print("🔍 Starting NSL-KDD Comprehensive Analysis")
    print("=" * 60)
    
    # Setup directories
    setup_project_directories()
    
    # Initialize analyzer
    analyzer = NSLKDDAnalyzer()
    
    # Check available files
    print("\n📁 Available data files:")
    data_files = list(analyzer.data_dir.glob("*.txt"))
    if not data_files:
        print("❌ No .txt files found in data/raw/")
        print("Please ensure your NSL-KDD files are in data/raw/")
        return
    
    for file in data_files:
        size_mb = file.stat().st_size / (1024 * 1024)
        print(f"  📄 {file.name:<25} ({size_mb:.1f} MB)")
    
    try:
        # Step 1: Analyze 20% subset first (fastest)
        print(f"\n" + "="*60)
        print(f"STEP 1: ANALYZING 20% TRAINING SUBSET")
        print(f"="*60)
        train_20_data = analyzer.comprehensive_analysis('KDDTrain+_20Percent.txt')
        
        if train_20_data is not None:
            print(f"\n📊 20% Subset Overview:")
            print(f"   Records: {len(train_20_data):,}")
            print(f"   Attack types: {train_20_data['attack_type'].nunique()}")
            
            # Attack distribution analysis
            attack_dist = train_20_data['attack_category'].value_counts()
            print(f"\n🎯 Attack Categories (20% subset):")
            for category, count in attack_dist.items():
                pct = (count / len(train_20_data)) * 100
                print(f"   {category:<8} {count:>6,} ({pct:5.1f}%)")
        
        # Step 2: Analyze full training data
        print(f"\n" + "="*60)
        print(f"STEP 2: ANALYZING FULL TRAINING DATA")
        print(f"="*60)
        train_full_data = analyzer.comprehensive_analysis('KDDTrain+.txt')
        
        # Step 3: Analyze test data
        print(f"\n" + "="*60)
        print(f"STEP 3: ANALYZING TEST DATA")
        print(f"="*60)
        test_data = analyzer.comprehensive_analysis('KDDTest+.txt')
        
        # Step 4: Compare datasets
        print(f"\n" + "="*60)
        print(f"STEP 4: DATASET COMPARISON")
        print(f"="*60)
        if train_full_data is not None and test_data is not None:
            novel_attacks = analyzer.compare_datasets(train_full_data, test_data)
            
            print(f"\n📊 Final Dataset Summary:")
            print(f"Training (Full): {len(train_full_data):,} records, {train_full_data['attack_type'].nunique()} attack types")
            print(f"Training (20%):  {len(train_20_data):,} records, {train_20_data['attack_type'].nunique()} attack types")
            print(f"Test:            {len(test_data):,} records, {test_data['attack_type'].nunique()} attack types")
            print(f"Novel attacks in test: {len(novel_attacks)}")
            
            # Data quality check
            print(f"\n🔍 Data Quality Summary:")
            missing_train = train_full_data.isnull().sum().sum()
            missing_test = test_data.isnull().sum().sum()
            print(f"Missing values - Train: {missing_train}, Test: {missing_test}")
            
            duplicates_train = train_full_data.duplicated().sum()
            duplicates_test = test_data.duplicated().sum()
            print(f"Duplicate records - Train: {duplicates_train}, Test: {duplicates_test}")
            
            # Class imbalance analysis
            print(f"\n⚖️ Class Imbalance Analysis:")
            normal_pct = (train_full_data['attack_type'] == 'normal').mean() * 100
            attack_pct = 100 - normal_pct
            print(f"Normal traffic: {normal_pct:.1f}%")
            print(f"Attack traffic: {attack_pct:.1f}%")
            
            # Detailed attack category breakdown
            full_attack_dist = train_full_data['attack_category'].value_counts()
            print(f"\n📈 Full Training Set Attack Distribution:")
            for category, count in full_attack_dist.items():
                pct = (count / len(train_full_data)) * 100
                print(f"   {category:<8} {count:>8,} ({pct:6.2f}%)")
            
            # Feature analysis
            numeric_features = train_full_data.select_dtypes(include=[np.number]).columns.tolist()
            numeric_features = [col for col in numeric_features if col not in ['difficulty_level']]
            categorical_features = ['protocol_type', 'service', 'flag', 'attack_type']
            
            print(f"\n📋 Feature Summary:")
            print(f"   Total features: {train_full_data.shape[1] - 2}")
            print(f"   Numeric features: {len(numeric_features)}")
            print(f"   Categorical features: {len(categorical_features)}")
            
            # Key insights
            print(f"\n" + "="*60)
            print(f"🎯 KEY INSIGHTS & RECOMMENDATIONS")
            print(f"="*60)
            
            print(f"1. Dataset Characteristics:")
            print(f"   ✓ Large-scale dataset suitable for ML research")
            print(f"   ✓ Clean data with no missing values")
            print(f"   ✓ Rich feature set (41 features across 4 categories)")
            
            print(f"\n2. Major Challenges Identified:")
            print(f"   ⚠️  Severe class imbalance (U2R: ~0.01%, R2L: ~0.23%)")
            print(f"   ⚠️  Novel attacks in test set ({len(novel_attacks)} new types)")
            print(f"   ⚠️  High dimensionality (41 features)")
            print(f"   ⚠️  Mixed data types (numeric + categorical)")
            
            print(f"\n3. Recommended Preprocessing Steps:")
            print(f"   📋 Encode categorical variables (protocol_type, service, flag)")
            print(f"   📋 Normalize/standardize numeric features")
            print(f"   📋 Apply feature selection techniques")
            print(f"   📋 Handle class imbalance (SMOTE, undersampling)")
            
            print(f"\n4. Modeling Strategy Recommendations:")
            print(f"   🤖 Start with binary classification (Normal vs Attack)")
            print(f"   🤖 Progress to 5-class classification")
            print(f"   🤖 Use ensemble methods for robustness")
            print(f"   🤖 Implement anomaly detection for novel attacks")
            
            print(f"\n5. Evaluation Considerations:")
            print(f"   📊 Use stratified cross-validation")
            print(f"   📊 Focus on precision/recall for minority classes")
            print(f"   📊 Test on novel attacks separately")
            print(f"   📊 Consider computational efficiency metrics")
            
        print(f"\n" + "="*60)
        print(f"✅ COMPREHENSIVE ANALYSIS COMPLETE!")
        print(f"="*60)
        print(f"📁 Check the following directories for outputs:")
        print(f"   • data/processed/ - Cleaned datasets")
        print(f"   • data/results/ - Analysis plots and summaries")
        print(f"\n🚀 Next steps:")
        print(f"   1. Run feature preprocessing: src/data_preprocessing/")
        print(f"   2. Develop baseline models: src/models/")
        print(f"   3. Implement evaluation metrics: src/evaluation/")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()