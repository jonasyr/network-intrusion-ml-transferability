#!/usr/bin/env python3
"""
RECOVERY SCRIPT - Recover from temp saves if pipeline crashes
"""

import pandas as pd
from pathlib import Path
import joblib
import sys

def recover_from_crash():
    """Recover models and results from temporary saves"""
    
    print("🔄 CRASH RECOVERY SYSTEM")
    print("=" * 50)
    
    recovery_data = {
        "models_recovered": 0,
        "results_recovered": 0,
        "baseline_models": [],
        "advanced_models": []
    }
    
    # Check for temp baseline models
    temp_baseline_dir = Path("data/models/temp")
    if temp_baseline_dir.exists():
        for model_file in temp_baseline_dir.glob("*.joblib"):
            try:
                model = joblib.load(model_file)
                model_name = model_file.stem.replace("_temp", "")
                
                # Move to proper location
                proper_dir = Path("data/models/baseline") 
                proper_dir.mkdir(parents=True, exist_ok=True)
                proper_path = proper_dir / f"{model_name}.joblib"
                
                joblib.dump(model, proper_path)
                recovery_data["baseline_models"].append(model_name)
                recovery_data["models_recovered"] += 1
                
                print(f"✅ Recovered baseline model: {model_name}")
                
            except Exception as e:
                print(f"❌ Failed to recover {model_file}: {e}")
    
    # Check for temp advanced models
    temp_advanced_dir = Path("data/models/temp_advanced") 
    if temp_advanced_dir.exists():
        for model_file in temp_advanced_dir.glob("*.joblib"):
            try:
                model = joblib.load(model_file)
                model_name = model_file.stem.replace("_temp", "")
                
                # Move to proper location
                proper_dir = Path("data/models/advanced")
                proper_dir.mkdir(parents=True, exist_ok=True)
                proper_path = proper_dir / f"{model_name}.joblib"
                
                joblib.dump(model, proper_path)
                recovery_data["advanced_models"].append(model_name)
                recovery_data["models_recovered"] += 1
                
                print(f"✅ Recovered advanced model: {model_name}")
                
            except Exception as e:
                print(f"❌ Failed to recover {model_file}: {e}")
    
    # Check for temp results
    temp_baseline_results = Path("data/results/temp_baseline_results.csv")
    if temp_baseline_results.exists():
        try:
            df = pd.read_csv(temp_baseline_results)
            
            # Determine dataset from context or filename
            final_path = Path("data/results/recovered_baseline_results.csv")
            df.to_csv(final_path, index=False)
            recovery_data["results_recovered"] += len(df)
            
            print(f"✅ Recovered {len(df)} baseline results")
            
        except Exception as e:
            print(f"❌ Failed to recover baseline results: {e}")
    
    temp_advanced_results = Path("data/results/temp_advanced_results.csv")
    if temp_advanced_results.exists():
        try:
            df = pd.read_csv(temp_advanced_results)
            
            final_path = Path("data/results/recovered_advanced_results.csv")
            df.to_csv(final_path, index=False)
            recovery_data["results_recovered"] += len(df)
            
            print(f"✅ Recovered {len(df)} advanced results")
            
        except Exception as e:
            print(f"❌ Failed to recover advanced results: {e}")
    
    # Summary
    print(f"\n📊 RECOVERY SUMMARY")
    print(f"Models recovered: {recovery_data['models_recovered']}")
    print(f"Results recovered: {recovery_data['results_recovered']}")
    
    if recovery_data['baseline_models']:
        print(f"Baseline models: {', '.join(recovery_data['baseline_models'])}")
    if recovery_data['advanced_models']:
        print(f"Advanced models: {', '.join(recovery_data['advanced_models'])}")
    
    # Cleanup temp files after successful recovery
    if recovery_data["models_recovered"] > 0:
        print(f"\n🧹 Cleaning up temp files...")
        try:
            if temp_baseline_dir.exists():
                for f in temp_baseline_dir.glob("*"):
                    f.unlink()
                temp_baseline_dir.rmdir()
                
            if temp_advanced_dir.exists():
                for f in temp_advanced_dir.glob("*"):
                    f.unlink()  
                temp_advanced_dir.rmdir()
                
            if temp_baseline_results.exists():
                temp_baseline_results.unlink()
            if temp_advanced_results.exists():
                temp_advanced_results.unlink()
                
            print("✅ Temp files cleaned up")
            
        except Exception as e:
            print(f"⚠️ Cleanup failed: {e}")
    
    if recovery_data["models_recovered"] == 0 and recovery_data["results_recovered"] == 0:
        print("ℹ️ No crash recovery data found - system was clean")
        return False
    else:
        print("✅ Recovery completed successfully")
        return True

if __name__ == "__main__":
    success = recover_from_crash()
    sys.exit(0 if success else 1)