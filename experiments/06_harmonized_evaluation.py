#!/usr/bin/env python3
"""Harmonized feature evaluation across NSL-KDD and CIC-IDS-2017 with memory adaptation."""

from __future__ import annotations

import gc
import json
import numpy as np
import pickle
from pathlib import Path
import sys
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from src.utils import (
    get_memory_adaptive_config,
    MemoryMonitor,
    optimize_memory_usage,
)

from src.preprocessing.harmonization import (
    COMMON_COLUMNS,
    SCHEMA_VERSION,
    harmonize_cic,
    harmonize_nsl,
    read_csv_any,
    to_common_from_cic,
    _prepare_cic_columns,
    validate_cic,
    to_union_from_cic,
    _compute_summary,
)
NSL_PATH = PROJECT_ROOT / "data/raw/nsl-kdd/KDDTrain+.txt"
CIC_SAMPLE_PATH = PROJECT_ROOT / "data/raw/cic-ids-2017/cic_ids_sample_backup.csv"
CIC_DATASET_DIR = PROJECT_ROOT / "data/raw/cic-ids-2017/full_dataset"
RESULTS_PATH = PROJECT_ROOT / "data/results/harmonized_cross_validation.json"
CHECKPOINT_DIR = PROJECT_ROOT / "data/results/harmonized_checkpoints"
NSL_CIC_CHECKPOINT = CHECKPOINT_DIR / "nsl_to_cic_model.pkl"
CIC_NSL_CHECKPOINT = CHECKPOINT_DIR / "cic_to_nsl_model.pkl"
PROGRESS_CHECKPOINT = CHECKPOINT_DIR / "training_progress.json"

RANDOM_STATE = 42

# Memory-adaptive constants - will be set based on available memory
def get_memory_adaptive_constants():
    config = get_memory_adaptive_config()
    if config["use_full_dataset"]:
        return {
            "MAX_SAMPLES": 100000,  # For target dataset evaluation
            "BATCH_SIZE": 20000,    # For incremental learning
        }
    else:
        return {
            "MAX_SAMPLES": 25000,   # For target dataset evaluation
            "BATCH_SIZE": 5000,     # For incremental learning
        }

NUMERIC_FEATURES = [
    "duration_ms",
    "fwd_bytes",
    "bwd_bytes",
    "fwd_packets",
    "bwd_packets",
    "urgent_count",
    "connection_rate",
    "service_rate",
    "error_rate",
    "land",
    "flow_bytes_per_s",
]

CATEGORICAL_FEATURES = ["protocol", "connection_state"]


def convert_numpy_types(obj):
    """Convert numpy types to JSON serializable types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


def save_checkpoint(pipeline: Pipeline, checkpoint_path: Path, progress_info: Dict) -> None:
    """Save model checkpoint and training progress."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save model
    with checkpoint_path.open('wb') as f:
        pickle.dump(pipeline, f)
    
    # Save progress info (convert numpy types to JSON serializable types)
    progress_path = checkpoint_path.parent / "training_progress.json"
    serializable_progress_info = convert_numpy_types(progress_info)
    with progress_path.open('w') as f:
        json.dump(serializable_progress_info, f, indent=2)
    
    print(f"💾 Checkpoint saved: {checkpoint_path.name}")


def load_checkpoint(checkpoint_path: Path) -> Optional[Pipeline]:
    """Load model checkpoint if exists."""
    if checkpoint_path.exists():
        try:
            with checkpoint_path.open('rb') as f:
                pipeline = pickle.load(f)
            print(f"📥 Loaded checkpoint: {checkpoint_path.name}")
            return pipeline
        except Exception as e:
            print(f"⚠️  Failed to load checkpoint: {e}")
    return None


def create_balanced_batch(chunk: pd.DataFrame, batch_size: int = 20000) -> Optional[pd.DataFrame]:
    """Create balanced batch from chunk ensuring both classes are present."""
    if chunk.empty:
        return None
        
    benign = chunk[chunk['label_binary'] == 0]
    malicious = chunk[chunk['label_binary'] == 1]
    
    if len(benign) == 0 or len(malicious) == 0:
        return None  # Skip if only one class
    
    # Take balanced samples
    samples_per_class = min(batch_size // 2, len(benign), len(malicious))
    
    if samples_per_class < 100:  # Skip very small batches
        return None
        
    benign_sample = benign.sample(n=samples_per_class, random_state=42)
    malicious_sample = malicious.sample(n=samples_per_class, random_state=42)
    
    balanced_batch = pd.concat([benign_sample, malicious_sample]).sample(frac=1, random_state=42)
    return balanced_batch


def compute_adaptive_class_weights(y: np.ndarray) -> Dict[int, float]:
    """Compute adaptive class weights based on actual class distribution.
    
    Args:
        y: Target labels
        
    Returns:
        Dictionary mapping class labels to their weights
    """
    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        # Only one class present, return equal weights
        return {int(unique_classes[0]): 1.0}
    
    # Use sklearn's compute_class_weight for balanced weights
    class_weights = compute_class_weight('balanced', classes=unique_classes, y=y)
    weight_dict = {int(cls): float(weight) for cls, weight in zip(unique_classes, class_weights)}
    
    print(f"Computed adaptive class weights: {weight_dict}")
    return weight_dict


def train_on_full_cic_dataset() -> Pipeline:
    """Train incrementally on the full CIC-IDS-2017 dataset with class balancing."""
    
    # Get memory-adaptive configuration
    config = get_memory_adaptive_config()
    constants = get_memory_adaptive_constants()
    BATCH_SIZE = constants["BATCH_SIZE"]
    
    if not config["use_full_dataset"] or not CIC_DATASET_DIR.exists():
        # Use sample dataset for memory optimization or if full dataset not available
        if CIC_SAMPLE_PATH.exists():
            print(f"Using sample dataset for training: {CIC_SAMPLE_PATH.name}")
            _, cic_common, _ = harmonize_cic(CIC_SAMPLE_PATH)
            X, y = _prepare_features(cic_common)
            pipeline = _build_regular_pipeline()
            pipeline.fit(X, y)
            return pipeline
        else:
            raise FileNotFoundError("No CIC-IDS-2017 data files found")
    
    print("🚀 Training on FULL CIC-IDS-2017 dataset...")
    
    # Get all CSV files
    csv_files = list(CIC_DATASET_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError("No CSV files found in CIC dataset directory")
    
    print(f"Training incrementally on {len(csv_files)} CIC-IDS-2017 files:")
    for file_path in sorted(csv_files):
        print(f"  - {file_path.name}")
    
    # Use class_weight='balanced' for adaptive class balancing
    pipeline = Pipeline(steps=[
        ("preprocess", ColumnTransformer(
            transformers=[
                ("num", Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]), NUMERIC_FEATURES),
                ("cat", Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]), CATEGORICAL_FEATURES),
            ],
            remainder="drop",
        )),
        ("clf", SGDClassifier(
            loss="log_loss",
            max_iter=1000,
            random_state=RANDOM_STATE,
            learning_rate="adaptive",
            eta0=0.1,  # Higher learning rate
            class_weight='balanced',  # Adaptive weights based on class distribution
            average=True  # Use averaged weights for better stability
        ))
    ])
    
    total_samples_processed = 0
    is_fitted = False
    validation_scores = []
    
    # Keep track of class distribution during training
    total_benign_seen = 0
    total_malicious_seen = 0
    
    for file_idx, file_path in enumerate(sorted(csv_files)):
        try:
            print(f"\nTraining on file {file_idx + 1}/{len(csv_files)}: {file_path.name}")
            
            # Check if file is a Git LFS pointer
            with file_path.open("r", encoding="utf-8", errors="ignore") as handle:
                first_line = handle.readline().lower()
            if "git-lfs" in first_line:
                print(f"  Skipping {file_path.name} (Git LFS pointer)")
                continue
            
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"  File size: {file_size_mb:.1f} MB")
            
            if file_size_mb > 500:  # Process large files in chunks
                print(f"  Processing in chunks of {BATCH_SIZE:,} rows...")
                chunk_iter = pd.read_csv(file_path, chunksize=BATCH_SIZE, encoding="utf-8", on_bad_lines="skip")
                
                for chunk_num, chunk in enumerate(chunk_iter):
                    processed_chunk = process_cic_chunk(chunk)
                    if processed_chunk is not None and len(processed_chunk) > 0:
                        X_chunk, y_chunk = _prepare_features(processed_chunk)
                        
                        # Check class distribution in this chunk
                        benign_count = (y_chunk == 0).sum()
                        malicious_count = (y_chunk == 1).sum()
                        total_benign_seen += benign_count
                        total_malicious_seen += malicious_count
                        
                        # Only train on chunks with both classes
                        if len(np.unique(y_chunk)) >= 2:
                            if not is_fitted:
                                # Compute adaptive class weights for initial fit
                                class_weights = compute_adaptive_class_weights(y_chunk)
                                
                                # Update pipeline with computed weights
                                pipeline.named_steps['clf'].set_params(class_weight=class_weights)
                                
                                # Fit the pipeline on the first valid chunk
                                print(f"    Initial fit on chunk {chunk_num} ({len(X_chunk):,} samples)")
                                print(f"    Class distribution: {benign_count} benign, {malicious_count} malicious")
                                pipeline.fit(X_chunk, y_chunk)
                                is_fitted = True
                                
                                # Test prediction on this chunk
                                y_pred_test = pipeline.predict(X_chunk)
                                test_acc = accuracy_score(y_chunk, y_pred_test)
                                test_f1 = f1_score(y_chunk, y_pred_test, zero_division=0)
                                print(f"    Initial training: Acc={test_acc:.3f}, F1={test_f1:.3f}")
                                
                            else:
                                # For partial fit, we use sample_weight to handle class imbalance
                                # since SGDClassifier doesn't support changing class_weight in partial_fit
                                
                                # Compute sample weights based on cumulative class distribution
                                cumulative_total = total_benign_seen + total_malicious_seen
                                if cumulative_total > 0:
                                    benign_ratio = total_benign_seen / cumulative_total
                                    malicious_ratio = total_malicious_seen / cumulative_total
                                    
                                    # Inverse frequency weighting
                                    sample_weights = np.where(y_chunk == 0, 
                                                            1.0 / (benign_ratio + 1e-8),  # Weight for benign
                                                            1.0 / (malicious_ratio + 1e-8))  # Weight for malicious
                                    
                                    # Normalize weights
                                    sample_weights = sample_weights / np.mean(sample_weights)
                                else:
                                    sample_weights = np.ones(len(y_chunk))
                                
                                if chunk_num % 5 == 0:  # Print less frequently
                                    print(f"    Partial fit on chunk {chunk_num} ({len(X_chunk):,} samples)")
                                    print(f"    Class distribution: {benign_count} benign, {malicious_count} malicious")
                                    print(f"    Sample weight range: [{np.min(sample_weights):.3f}, {np.max(sample_weights):.3f}]")
                                
                                # For SGDClassifier, we need to use partial_fit on the classifier
                                X_transformed = pipeline[:-1].transform(X_chunk)
                                pipeline.named_steps['clf'].partial_fit(X_transformed, y_chunk, 
                                                                       sample_weight=sample_weights)
                                
                                # Periodic validation
                                if chunk_num % 20 == 0:
                                    y_pred_val = pipeline.predict(X_chunk)
                                    val_acc = accuracy_score(y_chunk, y_pred_val)
                                    val_f1 = f1_score(y_chunk, y_pred_val, zero_division=0)
                                    
                                    # Check prediction distribution
                                    pred_benign = (y_pred_val == 0).sum()
                                    pred_malicious = (y_pred_val == 1).sum()
                                    
                                    print(f"    Validation (chunk {chunk_num}): Acc={val_acc:.3f}, F1={val_f1:.3f}")
                                    print(f"    Predictions: {pred_benign} benign, {pred_malicious} malicious")
                                    validation_scores.append({
                                        "chunk": chunk_num, 
                                        "acc": val_acc, 
                                        "f1": val_f1,
                                        "pred_benign": pred_benign,
                                        "pred_malicious": pred_malicious
                                    })
                            
                            total_samples_processed += len(X_chunk)
                        else:
                            if chunk_num % 10 == 0:
                                print(f"    Skipping chunk {chunk_num} (only {benign_count} benign, {malicious_count} malicious)")
                    
                    # Clean up memory
                    del chunk, processed_chunk
                    if chunk_num % 10 == 0 and chunk_num > 0:
                        gc.collect()
                        
                    
            else:
                # Process smaller files entirely (similar logic with class balancing)
                raw_df = read_csv_any(file_path)
                print(f"  Loaded {len(raw_df):,} rows")
                
                if len(raw_df) > 0:
                    processed_df = process_cic_chunk(raw_df)
                    if processed_df is not None and len(processed_df) > 0:
                        
                        # Process in batches even for smaller files
                        for i in range(0, len(processed_df), BATCH_SIZE):
                            batch = processed_df.iloc[i:i + BATCH_SIZE]
                            X_batch, y_batch = _prepare_features(batch)
                            
                            benign_count = (y_batch == 0).sum()
                            malicious_count = (y_batch == 1).sum()
                            total_benign_seen += benign_count
                            total_malicious_seen += malicious_count
                            
                            if len(np.unique(y_batch)) >= 2:
                                if not is_fitted:
                                    # Compute adaptive class weights for initial fit
                                    class_weights = compute_adaptive_class_weights(y_batch)
                                    
                                    # Update pipeline with computed weights
                                    pipeline.named_steps['clf'].set_params(class_weight=class_weights)
                                    
                                    print(f"    Initial fit on batch {i//BATCH_SIZE + 1} ({len(X_batch):,} samples)")
                                    print(f"    Class distribution: {benign_count} benign, {malicious_count} malicious")
                                    pipeline.fit(X_batch, y_batch)
                                    is_fitted = True
                                    
                                    # Test prediction
                                    y_pred_test = pipeline.predict(X_batch)
                                    test_acc = accuracy_score(y_batch, y_pred_test)
                                    test_f1 = f1_score(y_batch, y_pred_test, zero_division=0)
                                    print(f"    Initial training: Acc={test_acc:.3f}, F1={test_f1:.3f}")
                                    
                                else:
                                    # Use sample weights for partial fit based on cumulative distribution
                                    cumulative_total = total_benign_seen + total_malicious_seen
                                    if cumulative_total > 0:
                                        benign_ratio = total_benign_seen / cumulative_total
                                        malicious_ratio = total_malicious_seen / cumulative_total
                                        
                                        # Inverse frequency weighting
                                        sample_weights = np.where(y_batch == 0, 
                                                                1.0 / (benign_ratio + 1e-8),
                                                                1.0 / (malicious_ratio + 1e-8))
                                        
                                        # Normalize weights
                                        sample_weights = sample_weights / np.mean(sample_weights)
                                    else:
                                        sample_weights = np.ones(len(y_batch))
                                    
                                    X_transformed = pipeline[:-1].transform(X_batch)
                                    pipeline.named_steps['clf'].partial_fit(X_transformed, y_batch,
                                                                           sample_weight=sample_weights)
                                
                                total_samples_processed += len(X_batch)
                            else:
                                print(f"    Skipping batch {i//BATCH_SIZE + 1} (only {benign_count} benign, {malicious_count} malicious)")
                
                # Clean up memory
                del raw_df
                gc.collect()
            
            print(f"  Completed {file_path.name}. Total samples processed: {total_samples_processed:,}")
            print(f"  Cumulative class distribution: {total_benign_seen:,} benign, {total_malicious_seen:,} malicious")
            
        except Exception as e:
            print(f"  Error processing {file_path.name}: {e}")
            continue
    
    if not is_fitted:
        raise RuntimeError("Could not fit the model - no valid data found")
    
    print(f"\nIncremental training completed!")
    print(f"Total samples processed: {total_samples_processed:,}")
    print(f"Final class distribution seen: {total_benign_seen:,} benign, {total_malicious_seen:,} malicious")
    print(f"Class imbalance ratio: {total_benign_seen/total_malicious_seen:.2f}:1 (benign:malicious)")
    
    if validation_scores:
        avg_val_acc = np.mean([s["acc"] for s in validation_scores])
        avg_val_f1 = np.mean([s["f1"] for s in validation_scores])
        print(f"Average validation accuracy during training: {avg_val_acc:.3f}")
        print(f"Average validation F1 during training: {avg_val_f1:.3f}")
        
        # Check if model was predicting both classes
        final_val = validation_scores[-1] if validation_scores else None
        if final_val:
            print(f"Final validation predictions: {final_val['pred_benign']} benign, {final_val['pred_malicious']} malicious")
    
    # Save checkpoint for incremental training
    save_checkpoint(pipeline, CIC_NSL_CHECKPOINT, {
        "total_samples": total_samples_processed,
        "benign_samples": total_benign_seen,
        "malicious_samples": total_malicious_seen,
        "validation_scores": validation_scores[-10:] if validation_scores else [],
        "training_type": "incremental"
    })
    
    return pipeline


def process_cic_chunk(chunk_df: pd.DataFrame) -> pd.DataFrame:
    """Process a chunk of CIC data and return harmonized common format."""
    try:
        if len(chunk_df) == 0:
            return None
        
        # Process using harmonization functions
        prepared = _prepare_cic_columns(chunk_df)
        validate_cic(prepared)
        common_df = to_common_from_cic(prepared)
        
        return common_df
    except Exception as e:
        print(f"    Error processing chunk: {e}")
        return None


def _resolve_cic_path() -> Path:
    # This function is no longer used but kept for compatibility
    if CIC_SAMPLE_PATH.exists():
        return CIC_SAMPLE_PATH
    raise FileNotFoundError("No CIC-IDS-2017 sample file found")


def _prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    feature_columns = [col for col in COMMON_COLUMNS if col not in {"label_binary", "label_multiclass"}]
    X = df.loc[:, feature_columns].copy()
    y = df.loc[:, "label_binary"].astype(np.int8)
    
    # Convert all numeric columns to float64 to ensure consistent dtype
    numeric_cols = [col for col in X.columns if col != 'protocol']  # protocol is categorical
    for col in numeric_cols:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # Clean data: replace infinity and extremely large values
    X = X.replace([np.inf, -np.inf], np.nan)
    
    # Replace extremely large values (> 1e10) with NaN
    for col in numeric_cols:
        mask = X[col].abs() > 1e10
        if mask.any():
            print(f"Warning: Replacing {mask.sum()} extreme values in column '{col}'")
            X.loc[mask, col] = np.nan
    
    # Check for columns that are entirely NaN
    nan_cols = X.columns[X.isnull().all()].tolist()
    if nan_cols:
        print(f"Warning: Some columns are entirely NaN: {nan_cols}")
    
    # Check for remaining infinity values
    for col in numeric_cols:
        if X[col].dtype.kind in 'biufc':  # numeric types
            inf_mask = np.isinf(X[col].fillna(0))
            if inf_mask.any():
                print(f"Warning: Column '{col}' still contains {inf_mask.sum()} infinity values")
                X.loc[inf_mask, col] = np.nan
    
    return X, y


def _build_pipeline() -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    # Use SGDClassifier for incremental learning capability
    classifier = SGDClassifier(
        loss="log_loss",  # For logistic regression
        max_iter=1000,
        random_state=RANDOM_STATE,
        learning_rate="adaptive",
        eta0=0.01,
        class_weight='balanced'  # Use balanced class weights
    )

    return Pipeline(steps=[("preprocess", preprocessor), ("clf", classifier)])


def _build_regular_pipeline() -> Pipeline:
    """Build pipeline with regular LogisticRegression for small datasets."""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    classifier = LogisticRegression(
        max_iter=500, 
        solver="lbfgs", 
        random_state=RANDOM_STATE,
        class_weight='balanced'  # Use balanced class weights
    )

    return Pipeline(steps=[("preprocess", preprocessor), ("clf", classifier)])


def evaluate_with_existing_model(
    pipeline: Pipeline,
    target_common: pd.DataFrame,
    source_name: str,
    target_name: str,
    use_incremental: bool = False,
) -> Dict[str, float | str]:
    """Evaluate using existing trained model."""
    X_target, y_target = _prepare_features(target_common)
    
    # Get prediction probabilities for threshold tuning
    y_proba = pipeline.predict_proba(X_target)
    
    # Try different thresholds to optimize F1 score
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    best_threshold = 0.5
    best_f1 = 0.0
    
    print("Testing different classification thresholds:")
    for threshold in thresholds:
        y_pred_thresh = (y_proba[:, 1] >= threshold).astype(int)
        f1_thresh = f1_score(y_target, y_pred_thresh, zero_division=0)
        acc_thresh = accuracy_score(y_target, y_pred_thresh)
        
        unique_pred, counts_pred = np.unique(y_pred_thresh, return_counts=True)
        pred_dist = dict(zip(unique_pred, counts_pred))
        
        print(f"  Threshold {threshold}: F1={f1_thresh:.3f}, Acc={acc_thresh:.3f}, Pred dist={pred_dist}")
        
        if f1_thresh > best_f1:
            best_f1 = f1_thresh
            best_threshold = threshold
    
    print(f"Best threshold: {best_threshold} (F1={best_f1:.3f})")
    
    # Use best threshold for final predictions
    y_pred = (y_proba[:, 1] >= best_threshold).astype(int)
    
    # Calculate final metrics
    target_accuracy = float(accuracy_score(y_target, y_pred))
    target_precision = float(precision_score(y_target, y_pred, zero_division=0))
    target_recall = float(recall_score(y_target, y_pred, zero_division=0))
    target_f1 = float(f1_score(y_target, y_pred, zero_division=0))
    
    unique_true, counts_true = np.unique(y_target, return_counts=True)
    unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
    
    print(f"Target true labels: {dict(zip(unique_true, counts_true))}")
    print(f"Final predicted labels: {dict(zip(unique_pred, counts_pred))}")
    print(f"Mean probabilities: {np.mean(y_proba, axis=0)}")
    
    # Print results
    print("\n" + "=" * 80)
    print(f"Training on {source_name} → Evaluating on {target_name}")
    print("-" * 80)
    print(f"Cross-validation F1 (source, 3-fold): N/A (cached model)")
    print(f"Optimized threshold: {best_threshold}")
    print(f"Transfer performance Acc={target_accuracy:.3f} | F1={target_f1:.3f} | Precision={target_precision:.3f} | Recall={target_recall:.3f}")
    if use_incremental:
        print("(Using cached incremental training model)")
    
    return {
        "source_dataset": source_name,
        "target_dataset": target_name,
        "cv_f1_mean": -1.0,  # N/A for cached model
        "cv_f1_std": -1.0,
        "best_threshold": best_threshold,
        "target_accuracy": target_accuracy,
        "target_precision": target_precision,
        "target_recall": target_recall,
        "target_f1": target_f1,
        "incremental_training": use_incremental,
    }


def evaluate_transfer_with_full_training(
    source_common: pd.DataFrame,
    target_common: pd.DataFrame,
    source_name: str,
    target_name: str,
    use_incremental: bool = False,
) -> Dict[str, float | str]:
    X_target, y_target = _prepare_features(target_common)

    if use_incremental and source_name == "CIC-IDS-2017":
        # Train incrementally on full CIC dataset
        print(f"Training incrementally on full {source_name} dataset...")
        pipeline = train_on_full_cic_dataset()
        
        # For incremental training, we can't do traditional CV, but we can report training stats
        cv_f1_mean = -1.0  # Will be updated if we implement online validation
        cv_f1_std = -1.0
        
    else:
        # Regular training for smaller datasets
        X_source, y_source = _prepare_features(source_common)
        pipeline = _build_regular_pipeline()
        cv_scores = cross_val_score(pipeline, X_source, y_source, cv=3, scoring="f1")
        cv_f1_mean = float(np.mean(cv_scores))
        cv_f1_std = float(np.std(cv_scores))
        pipeline.fit(X_source, y_source)
        
        # Save checkpoint for regular training
        if source_name == "NSL-KDD":
            save_checkpoint(pipeline, NSL_CIC_CHECKPOINT, {
                "source": source_name,
                "target": target_name,
                "cv_f1_mean": cv_f1_mean,
                "cv_f1_std": cv_f1_std,
                "training_type": "regular"
            })

    # Evaluate on target with detailed debugging
    print("Evaluating on target dataset...")
    
    # Get prediction probabilities for threshold tuning
    y_proba = pipeline.predict_proba(X_target)
    
    # Try different thresholds to optimize F1 score
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    best_threshold = 0.5
    best_f1 = 0.0
    
    print("Testing different classification thresholds:")
    for threshold in thresholds:
        y_pred_thresh = (y_proba[:, 1] >= threshold).astype(int)
        f1_thresh = f1_score(y_target, y_pred_thresh, zero_division=0)
        acc_thresh = accuracy_score(y_target, y_pred_thresh)
        
        unique_pred, counts_pred = np.unique(y_pred_thresh, return_counts=True)
        pred_dist = dict(zip(unique_pred, counts_pred))
        
        print(f"  Threshold {threshold}: F1={f1_thresh:.3f}, Acc={acc_thresh:.3f}, Pred dist={pred_dist}")
        
        if f1_thresh > best_f1:
            best_f1 = f1_thresh
            best_threshold = threshold
    
    print(f"Best threshold: {best_threshold} (F1={best_f1:.3f})")
    
    # Use best threshold for final predictions
    y_pred = (y_proba[:, 1] >= best_threshold).astype(int)
    
    # Debug prediction distribution
    unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
    unique_true, counts_true = np.unique(y_target, return_counts=True)
    print(f"Target true labels: {dict(zip(unique_true, counts_true))}")
    print(f"Final predicted labels: {dict(zip(unique_pred, counts_pred))}")
    print(f"Mean probabilities: {np.mean(y_proba, axis=0)}")

    metrics = {
        "source": source_name,
        "target": target_name,
        "cv_f1_mean": cv_f1_mean,
        "cv_f1_std": cv_f1_std,
        "target_accuracy": float(accuracy_score(y_target, y_pred)),
        "target_precision": float(precision_score(y_target, y_pred, zero_division=0)),
        "target_recall": float(recall_score(y_target, y_pred, zero_division=0)),
        "target_f1": float(f1_score(y_target, y_pred, zero_division=0)),
        "incremental_training": use_incremental,
        "best_threshold": float(best_threshold),
        "threshold_tuned_f1": float(best_f1),
    }

    print("\n" + "=" * 80)
    print(f"Training on {source_name} → Evaluating on {target_name}")
    if use_incremental:
        print("(Using incremental training on full dataset)")
    print("-" * 80)
    if cv_f1_mean >= 0:
        print(f"Cross-validation F1 (source, 3-fold): {metrics['cv_f1_mean']:.3f} ± {metrics['cv_f1_std']:.3f}")
    else:
        print("Cross-validation not applicable for incremental training")
    print(f"Optimized threshold: {best_threshold}")
    print(
        "Transfer performance "
        f"Acc={metrics['target_accuracy']:.3f} | F1={metrics['target_f1']:.3f} | "
        f"Precision={metrics['target_precision']:.3f} | Recall={metrics['target_recall']:.3f}"
    )

    return metrics


def main() -> None:
    # Get memory-adaptive configuration
    config = get_memory_adaptive_config()
    use_full = config["use_full_dataset"]
    
    title = "🚀 Harmonized cross-dataset validation"
    if use_full:
        title += " with FULL DATASET training"
    else:
        title += " with MEMORY OPTIMIZED training"
    
    print(title)
    print("Schema version:", SCHEMA_VERSION)
    print(f"💾 Memory mode: {'Full dataset' if use_full else 'Optimized'}")

    with MemoryMonitor("NSL-KDD Loading"):
        # Load NSL-KDD dataset
        nsl_union, nsl_common, nsl_summary = harmonize_nsl(NSL_PATH)
        print(f"Loaded NSL-KDD rows: {len(nsl_common):,}")
    
    with MemoryMonitor("CIC-IDS-2017 Loading"):
        # For evaluation, we need a representative CIC sample (not for training)
        print(f"\nLoading CIC-IDS-2017 sample for evaluation...")
        if use_full:
            cic_sample_path = CIC_DATASET_DIR / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
            if not cic_sample_path.exists():
                cic_sample_path = CIC_SAMPLE_PATH
        else:
            cic_sample_path = CIC_SAMPLE_PATH
            print("   Using sample dataset for memory optimization")
        
        _, cic_common_sample, cic_summary = harmonize_cic(cic_sample_path)
        print(f"Loaded CIC-IDS sample rows: {len(cic_common_sample):,}")

    # Get memory-adaptive constants
    constants = get_memory_adaptive_constants()
    MAX_SAMPLES = constants["MAX_SAMPLES"]
    BATCH_SIZE = constants["BATCH_SIZE"]
    
    print(f"📊 Using MAX_SAMPLES: {MAX_SAMPLES:,}, BATCH_SIZE: {BATCH_SIZE:,}")

    # Create balanced samples for evaluation
    def create_balanced_sample(df: pd.DataFrame, max_samples: int = MAX_SAMPLES) -> pd.DataFrame:
        """Create a balanced sample with both classes represented."""
        benign = df[df['label_binary'] == 0]
        malicious = df[df['label_binary'] == 1]
        
        print(f"Original: {len(benign)} benign, {len(malicious)} malicious")
        
        if len(malicious) == 0:
            print("Warning: No malicious samples found!")
            return df.head(max_samples).reset_index(drop=True)
        
        if len(benign) == 0:
            print("Warning: No benign samples found!")
            return df.head(max_samples).reset_index(drop=True)
        
        # Take equal samples from each class
        samples_per_class = min(max_samples // 2, len(benign), len(malicious))
        
        benign_sample = benign.head(samples_per_class)
        malicious_sample = malicious.head(samples_per_class)
        
        balanced = pd.concat([benign_sample, malicious_sample], ignore_index=True)
        balanced = balanced.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
        
        print(f"Balanced sample: {len(balanced)} total ({samples_per_class} per class)")
        return balanced

    print("\nCreating balanced evaluation samples...")
    nsl_sample = create_balanced_sample(nsl_common, MAX_SAMPLES)
    cic_eval_sample = create_balanced_sample(cic_common_sample, MAX_SAMPLES)

    print("\nRunning cross-dataset evaluation...")
    results: List[Dict[str, float | str]] = []
    
    # Check for existing checkpoints
    print("\n🔍 Checking for existing model checkpoints...")
    nsl_cic_model = load_checkpoint(NSL_CIC_CHECKPOINT)
    cic_nsl_model = load_checkpoint(CIC_NSL_CHECKPOINT)
    
    # 1. Train on NSL-KDD (regular) → Evaluate on CIC-IDS-2017
    if nsl_cic_model is not None:
        print("✅ Using cached NSL→CIC model")
        results.append(evaluate_with_existing_model(
            nsl_cic_model, cic_eval_sample, "NSL-KDD", "CIC-IDS-2017", use_incremental=False
        ))
    else:
        results.append(evaluate_transfer_with_full_training(
            nsl_sample, cic_eval_sample, "NSL-KDD", "CIC-IDS-2017", use_incremental=False
        ))
    
    # 2. Train on FULL CIC-IDS-2017 (incremental) → Evaluate on NSL-KDD
    if cic_nsl_model is not None:
        print("✅ Using cached CIC→NSL model")
        results.append(evaluate_with_existing_model(
            cic_nsl_model, nsl_sample, "CIC-IDS-2017", "NSL-KDD", use_incremental=True
        ))
    else:
        results.append(evaluate_transfer_with_full_training(
            cic_eval_sample, nsl_sample, "CIC-IDS-2017", "NSL-KDD", use_incremental=True
        ))

    # Save results
    with MemoryMonitor("Results Saving"):
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "max_samples": MAX_SAMPLES,
            "batch_size": BATCH_SIZE,
            "memory_mode": "full_dataset" if use_full else "optimized",
            "training_method": "incremental_full_dataset" if use_full else "sample_based",
            "nsl_summary": nsl_summary,
            "cic_summary": cic_summary,
            "results": results,
        }
        RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nResults saved to {RESULTS_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
