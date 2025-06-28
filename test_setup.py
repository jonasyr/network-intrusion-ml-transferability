# test_setup.py
"""
Quick test to verify your setup is working
Run this after creating the missing files
"""

import sys
from pathlib import Path

def test_imports():
    """Test if all imports work"""
    print("🧪 Testing imports...")
    
    try:
        # Test basic imports
        import pandas as pd
        import numpy as np
        import sklearn
        print("✅ Basic packages OK")
        
        # Test project imports
        sys.path.append('src')
        from nsl_kdd_analyzer import NSLKDDAnalyzer
        print("✅ NSLKDDAnalyzer OK")
        
        try:
            from data.preprocessor import NSLKDDPreprocessor
            print("✅ Preprocessor OK")
        except ImportError as e:
            print(f"❌ Preprocessor import failed: {e}")
            print("💡 Create src/data/preprocessor.py from the artifact")
        
        try:
            from models.baseline import BaselineModels
            print("✅ BaselineModels OK")
        except ImportError as e:
            print(f"❌ BaselineModels import failed: {e}")
            print("💡 Create src/models/baseline.py from the artifact")
            
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("💡 Run: pip install -r requirements.txt")
        return False

def test_data():
    """Test if data files are accessible"""
    print("\n📁 Testing data access...")
    
    sys.path.append('src')
    from nsl_kdd_analyzer import NSLKDDAnalyzer
    
    analyzer = NSLKDDAnalyzer()
    data_files = list(analyzer.data_dir.glob("*.txt"))
    
    if data_files:
        print(f"✅ Found {len(data_files)} data files:")
        for file in data_files:
            size_mb = file.stat().st_size / (1024 * 1024)
            print(f"   📄 {file.name} ({size_mb:.1f} MB)")
        return True
    else:
        print("❌ No data files found!")
        print("💡 Make sure NSL-KDD .txt files are in data/raw/")
        return False

def test_quick_analysis():
    """Test quick data analysis"""
    print("\n🔍 Testing quick analysis...")
    
    try:
        sys.path.append('src')
        from nsl_kdd_analyzer import NSLKDDAnalyzer
        
        analyzer = NSLKDDAnalyzer()
        data = analyzer.load_data("KDDTrain+_20Percent.txt")
        
        if data is not None:
            print(f"✅ Loaded data: {data.shape}")
            print(f"✅ Attack categories: {data['attack_category'].value_counts().to_dict()}")
            return True
        else:
            print("❌ Failed to load data")
            return False
            
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 TESTING YOUR SETUP")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 3
    
    if test_imports():
        tests_passed += 1
    
    if test_data():
        tests_passed += 1
        
    if test_quick_analysis():
        tests_passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 TEST RESULTS: {tests_passed}/{total_tests} passed")
    
    if tests_passed == total_tests:
        print("✅ ALL TESTS PASSED! 🎉")
        print("\n🎯 You're ready to run:")
        print("   • jupyter lab notebooks/01_initial_data_analysis.ipynb")
        print("   • python quick_start.py")
        print("   • python run_analysis.py")
    else:
        print("❌ Some tests failed. Check the messages above.")
        print("\n💡 Quick fixes:")
        print("   • Install packages: pip install -r requirements.txt")
        print("   • Create missing files from the artifacts above")
        print("   • Ensure NSL-KDD data is in data/raw/")

if __name__ == "__main__":
    main()