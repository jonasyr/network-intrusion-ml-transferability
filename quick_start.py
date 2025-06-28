# quick_start.py
"""
Quick start script for NSL-KDD analysis
Run this to get immediate insights into your dataset
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
    print("✅ All imports successful!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please install requirements: pip install -r requirements.txt")
    sys.exit(1)

def main():
    print("🚀 NSL-KDD Quick Start Analysis")
    print("=" * 50)
    
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
    
    # Quick analysis of 20% subset (fastest)
    print(f"\n🔍 Starting quick analysis with 20% training subset...")
    
    try:
        # Load and analyze 20% subset
        data_20 = analyzer.load_data('KDDTrain+_20Percent.txt')
        
        if data_20 is not None:
            print(f"\n📊 Quick Dataset Overview:")
            print(f"   Records: {len(data_20):,}")
            print(f"   Features: {data_20.shape[1] - 2} + 2 labels")
            
            # Attack distribution
            attack_dist = data_20['attack_category'].value_counts()
            print(f"\n🎯 Attack Categories:")
            for category, count in attack_dist.items():
                pct = (count / len(data_20)) * 100
                print(f"   {category:<8} {count:>6,} ({pct:5.1f}%)")
            
            # Protocol distribution  
            protocol_dist = data_20['protocol_type'].value_counts()
            print(f"\n📡 Protocols: {', '.join(protocol_dist.index.tolist())}")
            
            # Save quick summary
            summary_path = Path("data/results/quick_summary.txt")
            with open(summary_path, 'w') as f:
                f.write("NSL-KDD Quick Analysis Summary\n")
                f.write("=" * 40 + "\n\n")
                f.write(f"Dataset: KDDTrain+_20Percent.txt\n")
                f.write(f"Records: {len(data_20):,}\n")
                f.write(f"Features: {data_20.shape[1] - 2}\n\n")
                f.write("Attack Distribution:\n")
                for category, count in attack_dist.items():
                    pct = (count / len(data_20)) * 100
                    f.write(f"  {category}: {count:,} ({pct:.1f}%)\n")
            
            print(f"\n✅ Quick analysis complete!")
            print(f"📄 Summary saved to: {summary_path}")
            
            # Show next steps
            print(f"\n🎯 What to do next:")
            print(f"   1. Install Jupyter: pip install jupyter")
            print(f"   2. Run notebook: jupyter notebook notebooks/01_initial_data_analysis.ipynb")
            print(f"   3. Or continue with Python:")
            print(f"      analyzer.comprehensive_analysis('KDDTrain+.txt')")
            print(f"      analyzer.comprehensive_analysis('KDDTest+.txt')")
            
        else:
            print("❌ Failed to load data")
            
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        print(f"💡 Try: python -c \"from src.nsl_kdd_analyzer import NSLKDDAnalyzer; print('OK')\"")

if __name__ == "__main__":
    main()