"""Feature alignment utilities for cross-dataset evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass
class AlignmentResult:
    """Container for aligned feature spaces."""

    source: np.ndarray
    target: np.ndarray
    metadata: Dict[str, object]


class FeatureAligner:
    """Align feature spaces between NSL-KDD and CIC-IDS-2017 datasets."""

    def __init__(self) -> None:
        # Semantic mappings between the two datasets. The mapping is not exhaustive,
        # but it captures the most widely used traffic statistics that appear in both
        # corpora. This allows reproducible experiments while acknowledging the
        # structural differences between the datasets.
        self.feature_mappings: Dict[str, Dict[str, object]] = {
            "duration": {
                "nsl_name": "duration",
                "cic_name": ["Flow_Duration", "Flow Duration"],  # Handle both formats
            },
            "forward_bytes": {
                "nsl_name": "src_bytes",
                "cic_name": [
                    "Total_Length_of_Fwd_Packets",
                    "Total Length of Fwd Packets",
                ],
            },
            "backward_bytes": {
                "nsl_name": "dst_bytes",
                "cic_name": [
                    "Total_Length_of_Bwd_Packets",
                    "Total Length of Bwd Packets",
                ],
            },
            "total_bytes": {
                "nsl_name": "total_bytes",
                "cic_name": ["total_bytes", "total bytes"],  # Handle both formats
            },
            "bytes_per_second": {
                "nsl_name": "bytes_per_second",
                "cic_name": [
                    "bytes_per_second",
                    "bytes per second",
                ],  # Handle both formats
            },
            "byte_ratio": {
                "nsl_name": "byte_ratio",
                "cic_name": ["byte_ratio", "byte ratio"],  # Handle both formats
            },
        }

    # ------------------------------------------------------------------
    # Utilities
    def _to_dataframe(
        self, data: np.ndarray | pd.DataFrame, feature_names: Sequence[str]
    ) -> pd.DataFrame:
        """Convert input data to a DataFrame with explicit feature names."""

        if isinstance(data, pd.DataFrame):
            return data.loc[:, feature_names].copy()
        return pd.DataFrame(data, columns=list(feature_names)).copy()

    # ------------------------------------------------------------------
    # Public API
    def extract_common_features(
        self,
        nsl_data: np.ndarray | pd.DataFrame,
        cic_data: np.ndarray | pd.DataFrame,
        nsl_features: Sequence[str],
        cic_features: Sequence[str],
    ) -> AlignmentResult:
        """Extract semantically aligned features from both datasets.

        Parameters
        ----------
        nsl_data, cic_data:
            Feature matrices from NSL-KDD and CIC-IDS-2017 respectively.
        nsl_features, cic_features:
            Ordered feature names corresponding to the provided matrices.

        Returns
        -------
        AlignmentResult
            Source and target matrices restricted to common semantic features. The
            ``metadata`` field contains the selected feature names for each dataset
            and the semantic mappings used to construct the alignment.
        """

        nsl_df = self._to_dataframe(nsl_data, nsl_features)
        cic_df = self._to_dataframe(cic_data, cic_features)

        selected_nsl: List[str] = []
        selected_cic: List[str] = []
        feature_pairs: List[Dict[str, str]] = []
        for semantic_key, mapping in self.feature_mappings.items():
            nsl_name = mapping["nsl_name"]
            cic_names = mapping["cic_name"]

            # Handle both single name and list of possible names
            if isinstance(cic_names, str):
                cic_names = [cic_names]

            # Find first matching CIC name
            matched_cic_name = None
            for cic_name in cic_names:
                if cic_name in cic_df.columns:
                    matched_cic_name = cic_name
                    break

            if nsl_name in nsl_df.columns and matched_cic_name is not None:
                selected_nsl.append(nsl_name)
                selected_cic.append(matched_cic_name)
                feature_pairs.append(
                    {
                        "semantic_feature": semantic_key,
                        "source_feature": nsl_name,
                        "target_feature": matched_cic_name,
                    }
                )

        if not selected_nsl or not selected_cic:
            raise ValueError(
                "No overlapping features found. Ensure that preprocessors expose"
                " appropriate feature names before calling extract_common_features()."
            )

        nsl_common = nsl_df[selected_nsl].to_numpy(dtype=float)
        cic_common = cic_df[selected_cic].to_numpy(dtype=float)

        return AlignmentResult(
            source=nsl_common,
            target=cic_common,
            metadata={
                "selected_features": list(selected_nsl),
                "source_features": list(selected_nsl),
                "target_features": list(selected_cic),
                "feature_pairs": feature_pairs,
            },
        )

    def pca_alignment(
        self,
        source_data: np.ndarray,
        target_data: np.ndarray,
        n_components: int = 20,
    ) -> AlignmentResult:
        """Project both datasets into a shared latent space using PCA."""

        if n_components <= 0:
            raise ValueError("n_components must be positive")

        if source_data.shape[1] != target_data.shape[1]:
            raise ValueError(
                "Source and target must have the same number of features for PCA alignment. "
                "Call extract_common_features() before performing dimensionality reduction."
            )

        scaler = StandardScaler()
        scaler.fit(source_data)
        source_scaled = scaler.transform(source_data)
        target_scaled = scaler.transform(target_data)

        n_components = min(n_components, source_scaled.shape[1])
        pca = PCA(n_components=n_components, random_state=42)
        pca.fit(source_scaled)

        source_aligned = pca.transform(source_scaled)
        target_aligned = pca.transform(target_scaled)

        return AlignmentResult(
            source=source_aligned,
            target=target_aligned,
            metadata={
                "scaler": scaler,
                "pca": pca,
                "explained_variance_ratio": pca.explained_variance_ratio_,
            },
        )

    def statistical_alignment(
        self, source_data: np.ndarray, target_data: np.ndarray
    ) -> AlignmentResult:
        """Standardize the distributions of source and target features."""

        if source_data.shape[1] != target_data.shape[1]:
            raise ValueError(
                "Source and target must have the same number of features for statistical "
                "alignment. Use extract_common_features() to identify semantically "
                "consistent columns before scaling."
            )

        scaler = StandardScaler()
        scaler.fit(source_data)
        source_scaled = scaler.transform(source_data)
        target_scaled = scaler.transform(target_data)

        return AlignmentResult(
            source=source_scaled,
            target=target_scaled,
            metadata={"scaler": scaler},
        )

    def transform_dataset(
        self,
        data: np.ndarray | pd.DataFrame,
        feature_names: Sequence[str],
        selected_features: Sequence[str],
    ) -> np.ndarray:
        """Project a dataset to the selected feature subset."""

        df = self._to_dataframe(data, feature_names)
        missing = [
            feature for feature in selected_features if feature not in df.columns
        ]
        if missing:
            raise ValueError("Missing expected features: " + ", ".join(missing))
        return df[list(selected_features)].to_numpy(dtype=float)

    def estimate_domain_divergence(
        self, source: np.ndarray, target: np.ndarray
    ) -> float:
        """Estimate Wasserstein distance between aligned feature distributions."""

        if source.shape[1] != target.shape[1]:
            raise ValueError("Source and target must have the same number of features")

        divergences: List[float] = []
        for i in range(source.shape[1]):
            divergences.append(wasserstein_distance(source[:, i], target[:, i]))
        return float(np.mean(divergences))


__all__ = ["FeatureAligner", "AlignmentResult"]
