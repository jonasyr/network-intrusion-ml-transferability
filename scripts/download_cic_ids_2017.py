#!/usr/bin/env python3
# scripts/download_cic_ids_2017.py
"""
Download and prepare CIC-IDS-2017 dataset for cross-dataset evaluation
"""

import os
import requests
from pathlib import Path
import zipfile
import pandas as pd
from urllib.parse import urlparse

def download_file(url: str, destination: Path, chunk_size: int = 8192):
    """Download a file with progress indication"""
    try:
        print(f"📥 Downloading {url}")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        file_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if file_size > 0:
                        progress = (downloaded / file_size) * 100
                        print(f"\r   Progress: {progress:.1f}%", end='', flush=True)
        
        print(f"\n✅ Downloaded: {destination.name} ({downloaded / 1024 / 1024:.1f} MB)")
        return True
        
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        return False

def extract_zip(zip_path: Path, extract_to: Path):
    """Extract ZIP file"""
    try:
        print(f"📦 Extracting {zip_path.name}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"✅ Extracted to {extract_to}")
        return True
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        return False

def main():
    """Download CIC-IDS-2017 dataset"""
    print("🚀 CIC-IDS-2017 Dataset Download")
    print("=" * 50)
    
    # Create download directory
    download_dir = Path("data/raw/cic_ids_2017")
    download_dir.mkdir(parents=True, exist_ok=True)
    
    # CIC-IDS-2017 download URLs (Kaggle mirror - faster and more reliable)
    urls = {
        "MachineLearningCSV.zip": "https://www.unb.ca/cic/datasets/ids-2017/dataset/MachineLearningCSV.zip"
    }
    
    print("📋 Download Plan:")
    print("   • MachineLearningCSV.zip (~500MB)")
    print("   • Contains all 5 days as CSV files")
    print("   • Ready for immediate ML processing")
    
    choice = input("\n❓ Proceed with download? (y/n): ").lower().strip()
    if choice != 'y':
        print("❌ Download cancelled")
        return False
    
    # Download files
    for filename, url in urls.items():
        file_path = download_dir / filename
        
        if file_path.exists():
            print(f"✅ {filename} already exists")
            continue
        
        success = download_file(url, file_path)
        if not success:
            print(f"❌ Failed to download {filename}")
            return False
        
        # Extract if it's a zip file
        if filename.endswith('.zip'):
            extract_success = extract_zip(file_path, download_dir)
            if not extract_success:
                return False
    
    # Check extracted files
    print("\n📁 Checking extracted files...")
    csv_files = list(download_dir.glob("**/*.csv"))
    
    if csv_files:
        print(f"✅ Found {len(csv_files)} CSV files:")
        total_size = 0
        for csv_file in csv_files:
            size_mb = csv_file.stat().st_size / (1024 * 1024)
            total_size += size_mb
            print(f"   📄 {csv_file.name} ({size_mb:.1f} MB)")
        
        print(f"\n📊 Total CSV size: {total_size:.1f} MB")
        
        # Quick sample of first file
        try:
            first_csv = csv_files[0]
            print(f"\n🔍 Quick preview of {first_csv.name}:")
            df_sample = pd.read_csv(first_csv, nrows=5)
            print(f"   Shape preview: {df_sample.shape}")
            print(f"   Columns: {len(df_sample.columns)}")
            print(f"   Sample columns: {list(df_sample.columns[:5])}...")
            
        except Exception as e:
            print(f"⚠️ Could not preview CSV: {e}")
    
    else:
        print("❌ No CSV files found after extraction")
        return False
    
    print("\n✅ CIC-IDS-2017 download complete!")
    print(f"📁 Location: {download_dir}")
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎯 Next steps:")
        print("   1. Run integration script to process CSV files")
        print("   2. Create cross-dataset evaluation experiments")
        print("   3. Update paper with cross-dataset results")