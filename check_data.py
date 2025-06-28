# check_data.py
"""
Data Overview & Insights Script

Quick analysis script for NSL-KDD dataset that provides:
- Dataset file overview and sizes
- Attack category distribution
- Data quality assessment
- Protocol analysis
- Summary file generation

Perfect for: "What does my data look like?" and demo purposes.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append('src')

def main():
    print("🚀 NSL-KDD Quick Start Analysis")
    print("=" * 50)
    
    try:
        from nsl_kdd_analyzer import NSLKDDAnalyzer
        
        # Initialize analyzer
        analyzer = NSLKDDAnalyzer()
        
        # Check available files
        print("\n📁 Available data files:")
        data_files = list(analyzer.data_dir.glob("*.txt"))
        if not data_files:
            print("❌ No .txt files found in data/raw/")
            print("💡 Please ensure your NSL-KDD files are in data/raw/")
            return
        
        for file in data_files:
            size_mb = file.stat().st_size / (1024 * 1024)
            print(f"  📄 {file.name:<25} ({size_mb:.1f} MB)")
        
        # Quick analysis of 20% subset
        print(f"\n🔍 Quick analysis with 20% training subset...")
        
        try:
            # Load and analyze 20% subset
            data_20 = analyzer.load_data('KDDTrain+_20Percent.txt')
            
            if data_20 is not None:
                print(f"\n📊 Dataset Overview:")
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
                print(f"\n📡 Top Protocols: {', '.join(protocol_dist.head(3).index.tolist())}")
                
                # Data quality check
                missing_values = data_20.isnull().sum().sum()
                duplicates = data_20.duplicated().sum()
                print(f"\n🔍 Data Quality:")
                print(f"   Missing values: {missing_values}")
                print(f"   Duplicates: {duplicates}")
                print(f"   Status: {'✅ Clean' if missing_values == 0 and duplicates == 0 else '⚠️ Needs cleaning'}")
                
                # Save quick summary
                summary_path = Path("data/results/quick_start_summary.txt")
                summary_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(summary_path, 'w') as f:
                    f.write("NSL-KDD Quick Start Summary\n")
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
                print(f"   1. Run full baseline: python scripts/run_baseline.py")
                print(f"   2. Open Jupyter notebook: jupyter lab notebooks/01_data_exploration.ipynb")
                print(f"   3. Run detailed analysis: python -c \"from src.nsl_kdd_analyzer import *; analyzer = NSLKDDAnalyzer(); analyzer.comprehensive_analysis('KDDTrain+.txt')\"")
                
            else:
                print("❌ Failed to load data")
                
        except Exception as e:
            print(f"❌ Error during analysis: {e}")
            print("💡 Try running the test script: python check_setup.py")
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print(f"💡 Make sure you have:")
        print(f"   • Installed packages: pip install -r requirements.txt")
        print(f"   • Created src/nsl_kdd_analyzer.py")
        print(f"   • Have NSL-KDD data in data/raw/")

if __name__ == "__main__":
    main()
