# src/data/preprocessor.py
"""
Baseline preprocessing pipeline for NSL-KDD dataset
Handles encoding, scaling, and class balancing
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import pickle
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

class NSLKDDPreprocessor:
    """
    Complete preprocessing pipeline for NSL-KDD dataset
    """
    
    def __init__(self, 
                 test_size: float = 0.2,
                 random_state: int = 42,
                 balance_method: str = 'smote'):
        """
        Initialize preprocessor
        
        Args:
            test_size: Proportion for train/validation split
            random_state: Random seed for reproducibility
            balance_method: 'smote', 'undersample', or 'none'
        """
        self.test_size = test_size
        self.random_state = random_state
        self.balance_method = balance_method
        
        # Preprocessing components
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: list = []
        self.attack_mapping: Dict[str, str] = {}
        
        # Define attack category mapping
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
        
        # Categorical features that need encoding
        self.categorical_features = ['protocol_type', 'service', 'flag']
        
    def add_attack_categories(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add attack category column based on attack_type"""
        data = data.copy()
        data['attack_category'] = data['attack_type'].map(self.attack_categories)
        
        # Handle unknown attack types
        unknown_attacks = data[data['attack_category'].isna()]['attack_type'].unique()
        if len(unknown_attacks) > 0:
            print(f"⚠️ Unknown attack types found: {unknown_attacks}")
            # Default unknown attacks to 'Unknown'
            data['attack_category'] = data['attack_category'].fillna('Unknown')
            
        return data
    
    def encode_categorical_features(self, data: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical features using LabelEncoder
        
        Args:
            data: Input dataframe
            fit: Whether to fit encoders (True for training, False for test)
            
        Returns:
            DataFrame with encoded categorical features
        """
        data = data.copy()
        
        for feature in self.categorical_features:
            if feature in data.columns:
                if fit:
                    # Fit and transform
                    self.label_encoders[feature] = LabelEncoder()
                    data[feature] = self.label_encoders[feature].fit_transform(data[feature])
                else:
                    # Transform only (using fitted encoder)
                    if feature in self.label_encoders:
                        # Handle unseen categories
                        le = self.label_encoders[feature]
                        data[feature] = data[feature].apply(
                            lambda x: le.transform([x])[0] if x in le.classes_ else -1
                        )
                    else:
                        print(f"⚠️ No fitted encoder found for {feature}")
                        
        return data
    
    def scale_numerical_features(self, data: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Scale numerical features using StandardScaler
        
        Args:
            data: Input dataframe
            fit: Whether to fit scaler (True for training, False for test)
            
        Returns:
            DataFrame with scaled numerical features
        """
        data = data.copy()
        
        # Identify numerical features (exclude labels and categorical)
        exclude_cols = ['attack_type', 'attack_category', 'difficulty_level'] + self.categorical_features
        numerical_features = [col for col in data.columns if col not in exclude_cols]
        
        if fit:
            # Fit and transform
            self.scaler = StandardScaler()
            data[numerical_features] = self.scaler.fit_transform(data[numerical_features])
            self.feature_names = numerical_features
        else:
            # Transform only
            if self.scaler is not None:
                data[numerical_features] = self.scaler.transform(data[numerical_features])
            else:
                print("⚠️ No fitted scaler found")
                
        return data
    
    def create_binary_labels(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create binary classification labels (Normal vs Attack)"""
        data = data.copy()
        data['is_attack'] = (data['attack_type'] != 'normal').astype(int)
        
        # Ensure labels are integers for classification
        data['is_attack'] = data['is_attack'].astype('int32')
        
        return data
    
    def balance_classes(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Balance classes using specified method
        
        Args:
            X: Feature matrix
            y: Target labels
            
        Returns:
            Balanced X and y
        """
        if self.balance_method == 'smote':
            # Use SMOTE for oversampling
            smote = SMOTE(random_state=self.random_state)
            X_balanced, y_balanced = smote.fit_resample(X, y)
            print(f"✓ Applied SMOTE: {len(X)} → {len(X_balanced)} samples")
            
        elif self.balance_method == 'undersample':
            # Use random undersampling
            undersampler = RandomUnderSampler(random_state=self.random_state)
            X_balanced, y_balanced = undersampler.fit_resample(X, y)
            print(f"✓ Applied undersampling: {len(X)} → {len(X_balanced)} samples")
            
        else:
            # No balancing
            X_balanced, y_balanced = X, y
            print("✓ No class balancing applied")
            
        return X_balanced, y_balanced
    
    def fit_transform(self, data: pd.DataFrame, target_type: str = 'binary') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Fit preprocessor and transform data for training
        
        Args:
            data: Training data
            target_type: 'binary', 'multiclass', or 'category'
            
        Returns:
            X_train, X_val, y_train, y_val
        """
        print("🔄 Fitting and transforming training data...")
        
        # Add attack categories
        data = self.add_attack_categories(data)
        
        # Create binary labels
        data = self.create_binary_labels(data)
        
        # Encode categorical features
        data = self.encode_categorical_features(data, fit=True)
        
        # Scale numerical features
        data = self.scale_numerical_features(data, fit=True)
        
        # Prepare features and labels
        exclude_cols = ['attack_type', 'attack_category', 'difficulty_level', 'is_attack']
        feature_cols = [col for col in data.columns if col not in exclude_cols]
        
        X = data[feature_cols].values
        
        # Select target based on type
        if target_type == 'binary':
            y = data['is_attack'].values.astype('int32')  # Ensure integer labels
        elif target_type == 'category':
            # Encode attack categories
            le_target = LabelEncoder()
            y = le_target.fit_transform(data['attack_category']).astype('int32')
            self.label_encoders['target'] = le_target
        else:  # multiclass
            # Encode attack types
            le_target = LabelEncoder()
            y = le_target.fit_transform(data['attack_type']).astype('int32')
            self.label_encoders['target'] = le_target
        
        # Train/validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.test_size, 
            random_state=self.random_state, 
            stratify=y
        )
        
        # Debug: Check label types and values
        print(f"✓ Labels - Type: {y_train.dtype}, Unique values: {np.unique(y_train)}")
        print(f"✓ Label distribution: {np.bincount(y_train)}")
        
        # Balance training data
        X_train, y_train = self.balance_classes(X_train, y_train)
        
        print(f"✓ Training set: {X_train.shape}")
        print(f"✓ Validation set: {X_val.shape}")
        print(f"✓ Features: {len(feature_cols)}")
        
        return X_train, X_val, y_train, y_val
    
    def transform(self, data: pd.DataFrame, target_type: str = 'binary') -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform test data using fitted preprocessor
        
        Args:
            data: Test data
            target_type: 'binary', 'multiclass', or 'category'
            
        Returns:
            X_test, y_test
        """
        print("🔄 Transforming test data...")
        
        # Add attack categories
        data = self.add_attack_categories(data)
        
        # Create binary labels
        data = self.create_binary_labels(data)
        
        # Encode categorical features (no fitting)
        data = self.encode_categorical_features(data, fit=False)
        
        # Scale numerical features (no fitting)
        data = self.scale_numerical_features(data, fit=False)
        
        # Prepare features and labels
        exclude_cols = ['attack_type', 'attack_category', 'difficulty_level', 'is_attack']
        feature_cols = [col for col in data.columns if col not in exclude_cols]
        
        X = data[feature_cols].values
        
        # Select target based on type
        if target_type == 'binary':
            y = data['is_attack'].values.astype('int32')
        elif target_type == 'category':
            if 'target' in self.label_encoders:
                y = self.label_encoders['target'].transform(data['attack_category']).astype('int32')
            else:
                raise ValueError("No fitted target encoder found")
        else:  # multiclass
            if 'target' in self.label_encoders:
                # Handle unknown attack types in test set
                known_attacks = set(self.label_encoders['target'].classes_)
                y = []
                for attack in data['attack_type']:
                    if attack in known_attacks:
                        y.append(self.label_encoders['target'].transform([attack])[0])
                    else:
                        y.append(-1)  # Unknown attack
                y = np.array(y, dtype='int32')
            else:
                raise ValueError("No fitted target encoder found")
        
        print(f"✓ Test set: {X.shape}")
        
        return X, y
    
    def save(self, filepath: str):
        """Save the fitted preprocessor"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'label_encoders': self.label_encoders,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'attack_categories': self.attack_categories,
                'categorical_features': self.categorical_features,
                'test_size': self.test_size,
                'random_state': self.random_state,
                'balance_method': self.balance_method
            }, f)
        print(f"✓ Preprocessor saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'NSLKDDPreprocessor':
        """Load a fitted preprocessor"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        preprocessor = cls(
            test_size=data['test_size'],
            random_state=data['random_state'],
            balance_method=data['balance_method']
        )
        
        preprocessor.label_encoders = data['label_encoders']
        preprocessor.scaler = data['scaler']
        preprocessor.feature_names = data['feature_names']
        preprocessor.attack_categories = data['attack_categories']
        preprocessor.categorical_features = data['categorical_features']
        
        print(f"✓ Preprocessor loaded from {filepath}")
        return preprocessor

# Example usage and testing
if __name__ == "__main__":
    # Test the preprocessor
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    
    from nsl_kdd_analyzer import NSLKDDAnalyzer
    
    # Load data
    analyzer = NSLKDDAnalyzer()
    train_data = analyzer.load_data("KDDTrain+_20Percent.txt")
    test_data = analyzer.load_data("KDDTest+.txt")
    
    if train_data is not None and test_data is not None:
        # Initialize and test preprocessor
        preprocessor = NSLKDDPreprocessor(balance_method='smote')
        
        # Fit on training data
        X_train, X_val, y_train, y_val = preprocessor.fit_transform(train_data, target_type='binary')
        
        # Transform test data
        X_test, y_test = preprocessor.transform(test_data, target_type='binary')
        
        print(f"\n📊 Final shapes:")
        print(f"Training: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")
        print(f"Class distribution (train): {np.bincount(y_train)}")
        print(f"Class distribution (val): {np.bincount(y_val)}")
        print(f"Class distribution (test): {np.bincount(y_test)}")
        
        # Save preprocessor
        save_path = Path("data/processed/preprocessor.pkl")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        preprocessor.save(str(save_path))
        
        print("\n✅ Preprocessor test completed successfully!")