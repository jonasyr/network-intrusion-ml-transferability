#!/usr/bin/env python3
"""
Validate and fix results file naming conflicts
"""

import pandas as pd
from pathlib import Path
import sys

def main():
    print("🔍 RESULTS FILE VALIDATION")
    print("="*50)
    
    results_dir = Path("data/results")
    
    # Expected dataset-specific files
    expected_files = {
        "nsl_baseline_results.csv": "NSL-KDD Baseline Results",
        "nsl_advanced_results.csv": "NSL-KDD Advanced Results", 
        "cic_baseline_results.csv": "CIC-IDS-2017 Baseline Results",
        "cic_advanced_results.csv": "CIC-IDS-2017 Advanced Results"
    }
    
    # Check for conflicts and duplicates
    conflicts = []
    found_files = {}
    
    for filename, description in expected_files.items():
        filepath = results_dir / filename
        if filepath.exists():
            try:
                df = pd.read_csv(filepath)
                found_files[filename] = {
                    'path': filepath,
                    'rows': len(df),
                    'description': description,
                    'dataset_column': df.get('dataset', ['Unknown']).iloc[0] if len(df) > 0 else 'Unknown'
                }
                print(f"✅ {description}: {len(df)} models")
            except Exception as e:
                print(f"❌ Error reading {filename}: {e}")
        else:
            print(f"⚠️  Missing: {description}")
    
    # Check for legacy files that might cause conflicts
    legacy_files = [
        "baseline_results.csv",
        "advanced_results.csv"
    ]
    
    for legacy_file in legacy_files:
        legacy_path = results_dir / legacy_file
        if legacy_path.exists():
            try:
                df = pd.read_csv(legacy_path)
                dataset_info = "Unknown"
                if 'dataset' in df.columns and len(df) > 0:
                    dataset_info = df['dataset'].iloc[0]
                
                conflicts.append({
                    'file': legacy_file,
                    'path': legacy_path,
                    'rows': len(df),
                    'dataset': dataset_info
                })
                print(f"🔄 Legacy file found: {legacy_file} ({len(df)} rows, dataset: {dataset_info})")
            except Exception as e:
                print(f"❌ Error reading legacy {legacy_file}: {e}")
    
    # Check subdirectories for additional conflicts
    subdir_conflicts = []
    for subdir in ["nsl", "cic"]:
        subdir_path = results_dir / subdir
        if subdir_path.exists():
            for result_file in ["baseline_results.csv", "advanced_results.csv"]:
                subfile_path = subdir_path / result_file
                if subfile_path.exists():
                    try:
                        df = pd.read_csv(subfile_path)
                        subdir_conflicts.append({
                            'file': f"{subdir}/{result_file}",
                            'path': subfile_path, 
                            'rows': len(df),
                            'expected_dataset': subdir.upper()
                        })
                        print(f"📁 Subdirectory file: {subdir}/{result_file} ({len(df)} rows)")
                    except Exception as e:
                        print(f"❌ Error reading {subdir}/{result_file}: {e}")
    
    print(f"\n📊 SUMMARY")
    print(f"Dataset-specific files found: {len(found_files)}/4")
    print(f"Legacy files found: {len(conflicts)}")
    print(f"Subdirectory files found: {len(subdir_conflicts)}")
    
    if conflicts or subdir_conflicts:
        print(f"\n⚠️  POTENTIAL CONFLICTS DETECTED")
        print("These files may cause confusion:")
        
        for conflict in conflicts:
            print(f"  - {conflict['file']}: {conflict['rows']} rows ({conflict['dataset']} dataset)")
        
        for subconf in subdir_conflicts:
            print(f"  - {subconf['file']}: {subconf['rows']} rows (expected {subconf['expected_dataset']})")
            
        print(f"\n💡 RECOMMENDATIONS:")
        print("1. Stop current pipeline if running")
        print("2. Clean up duplicate files") 
        print("3. Restart pipeline with fixed naming")
        return False
    else:
        print(f"\n✅ NO CONFLICTS DETECTED")
        print("File naming is consistent and dataset-specific")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)