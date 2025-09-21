# src/data/cic_ids_preprocessor.py
"""
CIC-IDS-2017 dataset preprocessor for cross-dataset evaluation
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Optional, Dict
import warnings
warnings.filterwarnings('ignore')

class CICIDSPreprocessor:
    """
    Preprocessor for CIC-IDS-2017 dataset
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize CIC-IDS-2017 preprocessor
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.scaler: Optional[StandardScaler] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.feature_names: list = []
        
        # CIC-IDS-2017 attack mapping to binary
        self.attack_mapping = {
            'BENIGN': 0,  # Normal traffic
            # All attacks mapped to 1
            'DoS Hulk': 1, 'PortScan': 1, 'DDoS': 1, 'DoS GoldenEye': 1,
            'FTP-Patator': 1, 'SSH-Patator': 1, 'DoS slowloris': 1, 
            'DoS Slowhttptest': 1, 'Bot': 1, 'Web Attack – Brute Force': 1,
            'Web Attack – XSS': 1, 'Infiltration': 1, 
            'Web Attack – Sql Injection': 1, 'Heartbleed': 1
        }
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load CIC-IDS-2017 data from CSV
        
        Args:
            file_path: Path to CIC-IDS-2017 CSV file
            
        Returns:
            Loaded DataFrame
        """
        try:
            print(f"📁 Loading CIC-IDS-2017 data from {file_path}...")
            data = pd.read_csv(file_path)
            
            # Basic data cleaning
            data = self._clean_data(data)
            
            print(f"✅ Loaded CIC-IDS-2017 data: {data.shape}")
            print(f"📊 Features: {data.shape[1] - 1}")
            print(f"🏷️ Labels: {data['Label'].nunique()} unique")
            
            return data
            
        except Exception as e:
            print(f"❌ Error loading CIC-IDS-2017 data: {e}")
            return None
    
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean CIC-IDS-2017 data"""
        
        # Handle infinite values
        data = data.replace([np.inf, -np.inf], np.nan)
        
        # Handle missing values
        data = data.fillna(0)
        
        # Remove duplicate rows
        initial_shape = data.shape[0]
        data = data.drop_duplicates()
        removed = initial_shape - data.shape[0]
        
        if removed > 0:
            print(f"🧹 Removed {removed} duplicate rows")
        
        return data
    
    def prepare_features(self, data: pd.DataFrame, fit: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and labels for ML
        
        Args:
            data: CIC-IDS-2017 DataFrame
            fit: Whether to fit scalers (True for training, False for test)
            
        Returns:
            Tuple of (features, labels)
        """
        
        # Separate features and labels
        feature_cols = [col for col in data.columns if col != 'Label']
        X = data[feature_cols].copy()
        
        # Store feature names
        if fit:
            self.feature_names = feature_cols
        
        # Handle any remaining non-numeric data
        for col in X.columns:
            if X[col].dtype == 'object':
                print(f"⚠️ Converting object column {col} to numeric")
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
        
        # Scale features
        if fit:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            print(f"✅ Fitted scaler on {X.shape[1]} features")
        else:
            if self.scaler is None:
                raise ValueError("Scaler not fitted. Call with fit=True first.")
            X_scaled = self.scaler.transform(X)
        
        # Prepare labels
        y = data['Label'].copy()
        
        # Map to binary classification
        y_binary = y.map(self.attack_mapping)
        
        # Handle unknown attacks
        unknown_mask = y_binary.isna()
        if unknown_mask.sum() > 0:
            print(f"⚠️ Found {unknown_mask.sum()} unknown attack types, marking as attacks")
            y_binary = y_binary.fillna(1)  # Unknown attacks treated as attacks
        
        y_binary = y_binary.astype(int)
        
        print(f"✅ Prepared features: {X_scaled.shape}")
        print(f"✅ Label distribution: Normal={np.sum(y_binary == 0)}, Attack={np.sum(y_binary == 1)}")
        
        return X_scaled, y_binary.values
    
    def fit_transform(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit preprocessor and transform data
        
        Args:
            data: CIC-IDS-2017 DataFrame
            
        Returns:
            Tuple of (features, labels)
        """
        return self.prepare_features(data, fit=True)
    
    def transform(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform data using fitted preprocessor
        
        Args:
            data: CIC-IDS-2017 DataFrame
            
        Returns:
            Tuple of (features, labels)
        """
        return self.prepare_features(data, fit=False)
    
    def get_feature_info(self) -> Dict:
        """Get information about processed features"""
        return {
            'n_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'attack_mapping': self.attack_mapping
        }


def test_cic_preprocessor():
    """Test the CIC-IDS-2017 preprocessor"""
    print("🧪 Testing CIC-IDS-2017 Preprocessor")
    print("=" * 50)
    
    # Load sample data
    preprocessor = CICIDSPreprocessor()
    data = preprocessor.load_data("data/raw/cic_ids_2017/cic_ids_sample.csv")
    
    if data is not None:
        # Test preprocessing
        X, y = preprocessor.fit_transform(data)
        
        print(f"\n📊 Preprocessing Results:")
        print(f"   Features shape: {X.shape}")
        print(f"   Labels shape: {y.shape}")
        print(f"   Feature range: [{X.min():.3f}, {X.max():.3f}]")
        print(f"   Class distribution: {np.bincount(y)}")
        
        return True
    
    return False

if __name__ == "__main__":
    test_cic_preprocessor()