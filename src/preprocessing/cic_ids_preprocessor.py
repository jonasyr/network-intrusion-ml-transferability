# src/data/cic_ids_preprocessor.py
"""
CIC-IDS-2017 dataset preprocessor for cross-dataset evaluation
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Optional, Dict
import warnings

warnings.filterwarnings("ignore")


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
            "BENIGN": 0,  # Normal traffic
            # All attacks mapped to 1
            "DoS Hulk": 1,
            "PortScan": 1,
            "DDoS": 1,
            "DoS GoldenEye": 1,
            "FTP-Patator": 1,
            "SSH-Patator": 1,
            "DoS slowloris": 1,
            "DoS Slowhttptest": 1,
            "Bot": 1,
            "Web Attack – Brute Force": 1,
            "Web Attack – XSS": 1,
            "Infiltration": 1,
            "Web Attack – Sql Injection": 1,
            "Heartbleed": 1,
        }

    @staticmethod
    def _augment_features(features: pd.DataFrame) -> pd.DataFrame:
        """Add shared traffic statistics used for cross-dataset alignment."""

        df = features.copy()

        # Handle both formats: with and without underscores
        forward_candidates = [
            "Total_Length_of_Fwd_Packets",
            "Total Length of Fwd Packets",
        ]
        backward_candidates = [
            "Total_Length_of_Bwd_Packets",
            "Total Length of Bwd Packets",
        ]
        duration_candidates = ["Flow_Duration", "Flow Duration"]

        # Find the actual column names
        forward = next((col for col in forward_candidates if col in df.columns), None)
        backward = next((col for col in backward_candidates if col in df.columns), None)
        duration = next((col for col in duration_candidates if col in df.columns), None)

        if forward and backward:
            df["total_bytes"] = df[forward] + df[backward]
            backward_safe = np.where(df[backward] <= 0, 1.0, df[backward])
            df["byte_ratio"] = df[forward] / backward_safe

        if forward and backward and duration:
            duration_safe = np.where(df[duration] <= 0, 1.0, df[duration])
            df["bytes_per_second"] = (df[forward] + df[backward]) / duration_safe

        return df

    def load_data(
        self, file_path: str = None, use_full_dataset: bool = True
    ) -> pd.DataFrame:
        """
        Load CIC-IDS-2017 data from CSV file(s)

        Args:
            file_path: Path to specific CSV file, or None to load full dataset
            use_full_dataset: If True and file_path is None, load all files from full dataset

        Returns:
            Loaded DataFrame
        """
        try:
            if file_path is None and use_full_dataset:
                # Load full dataset from multiple files
                return self._load_full_dataset()
            elif file_path is None:
                # Load sample dataset (backward compatibility)
                file_path = "data/raw/cic-ids-2017/cic_ids_sample_backup.csv"

            print(f"📁 Loading CIC-IDS-2017 data from {file_path}...")
            data = pd.read_csv(file_path)

            # Clean column names (remove leading/trailing spaces)
            data.columns = data.columns.str.strip()

            # Basic data cleaning
            data = self._clean_data(data)

            print(f"✅ Loaded CIC-IDS-2017 data: {data.shape}")
            print(f"📊 Features: {data.shape[1] - 1}")
            print(f"🏷️ Labels: {data['Label'].nunique()} unique")

            return data

        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None

    def _load_full_dataset(self) -> pd.DataFrame:
        """
        Load and combine all files from the full CIC-IDS-2017 dataset

        Returns:
            Combined DataFrame from all daily files
        """
        from pathlib import Path

        dataset_dir = Path("data/raw/cic-ids-2017/full_dataset")
        csv_files = list(dataset_dir.glob("*.csv"))

        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {dataset_dir}")

        print(f"📁 Loading full CIC-IDS-2017 dataset from {len(csv_files)} files...")

        combined_data = pd.DataFrame()
        total_rows = 0

        for i, csv_file in enumerate(sorted(csv_files)):
            print(f"   Loading {csv_file.name}... ({i+1}/{len(csv_files)})")

            try:
                # Load individual file
                df = pd.read_csv(csv_file)

                # Clean column names (remove leading/trailing spaces)
                df.columns = df.columns.str.strip()

                # Combine with main dataset
                combined_data = pd.concat([combined_data, df], ignore_index=True)
                total_rows += len(df)

                print(f"     ✅ {len(df):,} rows added (total: {total_rows:,})")

            except Exception as e:
                print(f"     ❌ Error loading {csv_file.name}: {e}")
                continue

        print(f"✅ Full dataset loaded: {combined_data.shape}")

        # Clean the combined data
        combined_data = self._clean_data(combined_data)

        return combined_data

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

    def prepare_features(
        self, data: pd.DataFrame, fit: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and labels for ML

        Args:
            data: CIC-IDS-2017 DataFrame
            fit: Whether to fit scalers (True for training, False for test)

        Returns:
            Tuple of (features, labels)
        """

        # Separate features and labels
        feature_cols = [col for col in data.columns if col != "Label"]
        X = data[feature_cols].copy()
        X = self._augment_features(X)

        if fit:
            feature_cols = list(X.columns)

        if not fit and self.feature_names:
            missing = [col for col in self.feature_names if col not in X.columns]
            if missing:
                raise ValueError(
                    f"Missing expected CIC-IDS-2017 features: {', '.join(missing)}"
                )
            X = X[self.feature_names]

        # Store feature names
        if fit:
            self.feature_names = feature_cols

        # Handle any remaining non-numeric data
        for col in X.columns:
            if X[col].dtype == "object":
                print(f"⚠️ Converting object column {col} to numeric")
                X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

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
        y = data["Label"].copy()

        # Map to binary classification
        y_binary = y.map(self.attack_mapping)

        # Handle unknown attacks
        unknown_mask = y_binary.isna()
        if unknown_mask.sum() > 0:
            print(
                f"⚠️ Found {unknown_mask.sum()} unknown attack types, marking as attacks"
            )
            y_binary = y_binary.fillna(1)  # Unknown attacks treated as attacks

        y_binary = y_binary.astype(int)

        print(f"✅ Prepared features: {X_scaled.shape}")
        print(
            f"✅ Label distribution: Normal={np.sum(y_binary == 0)}, Attack={np.sum(y_binary == 1)}"
        )

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
            "n_features": len(self.feature_names),
            "feature_names": self.feature_names,
            "attack_mapping": self.attack_mapping,
        }


def test_cic_preprocessor():
    """Test the CIC-IDS-2017 preprocessor with full dataset"""
    print("🧪 Testing CIC-IDS-2017 Preprocessor (Full Dataset)")
    print("=" * 50)

    preprocessor = CICIDSPreprocessor()

    # Test with full dataset
    print("📊 Loading full dataset...")
    data = preprocessor.load_data(use_full_dataset=True)

    if data is not None:
        # Test preprocessing
        X, y = preprocessor.prepare_features(data)

        print(f"\n📊 Preprocessing Results:")
        print(f"   Features shape: {X.shape}")
        print(f"   Labels shape: {y.shape}")
        print(f"   Feature range: [{X.min():.3f}, {X.max():.3f}]")
        print(f"   Class distribution: {np.bincount(y)}")

        # Show attack type distribution
        print(f"\n🏷️ Attack Type Distribution:")
        labels_df = pd.DataFrame({"Label": data["Label"]})
        distribution = labels_df["Label"].value_counts()
        for label, count in distribution.items():
            percentage = (count / len(data)) * 100
            print(f"   {label}: {count:,} ({percentage:.1f}%)")

        # Compare with sample if available
        print(f"\n🔍 Testing sample dataset for comparison...")
        try:
            sample_data = preprocessor.load_data(
                "data/raw/cic_ids_2017/cic_ids_sample.csv", use_full_dataset=False
            )
            if sample_data is not None:
                print(f"   Sample size: {sample_data.shape}")
                print(f"   Full dataset: {data.shape}")
                print(f"   Size increase: {len(data) / len(sample_data):.1f}x")
        except:
            print("   Sample file not available")

        return True

    return False


if __name__ == "__main__":
    test_cic_preprocessor()
