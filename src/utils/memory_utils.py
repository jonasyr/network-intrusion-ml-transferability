#!/usr/bin/env python3
"""Memory utilities for adaptive dataset processing based on available system memory."""

import gc
import psutil
from typing import Tuple, Dict, Any
import numpy as np


def get_system_memory_gb() -> float:
    """Get total system memory in GB."""
    return psutil.virtual_memory().total / (1024**3)


def get_available_memory_gb() -> float:
    """Get currently available memory in GB."""
    return psutil.virtual_memory().available / (1024**3)


def should_use_full_dataset(min_memory_gb: float = 20.0) -> bool:
    """
    Determine if we should use the full dataset based on available memory.
    
    Args:
        min_memory_gb: Minimum memory in GB required for full dataset processing
        
    Returns:
        True if we have enough memory for full dataset processing
    """
    total_memory = get_system_memory_gb()
    available_memory = get_available_memory_gb()
    
    print(f"💾 System memory: {total_memory:.1f}GB total, {available_memory:.1f}GB available")
    
    # Use full dataset if we have enough total memory and reasonable available memory
    use_full = total_memory >= min_memory_gb and available_memory >= (min_memory_gb * 0.6)
    
    if use_full:
        print(f"✅ Using FULL dataset (sufficient memory: {total_memory:.1f}GB)")
    else:
        print(f"⚠️ Using OPTIMIZED dataset (limited memory: {total_memory:.1f}GB)")
    
    return use_full


def get_optimal_sample_size(
    total_samples: int, 
    features: int,
    target_memory_gb: float = 8.0
) -> int:
    """
    Calculate optimal sample size based on memory constraints.
    
    Args:
        total_samples: Total number of samples in dataset
        features: Number of features
        target_memory_gb: Target memory usage in GB
        
    Returns:
        Optimal sample size for the given memory constraints
    """
    # Rough estimate: 8 bytes per float64 feature + overhead
    bytes_per_sample = features * 8 * 3  # 3x for processing overhead
    target_bytes = target_memory_gb * (1024**3)
    
    optimal_samples = min(total_samples, int(target_bytes / bytes_per_sample))
    
    # Ensure minimum viable sample size
    min_samples = max(10000, int(total_samples * 0.01))  # At least 1% or 10k
    optimal_samples = max(optimal_samples, min_samples)
    
    # Cap at total samples to avoid issues
    optimal_samples = min(optimal_samples, total_samples)
    
    print(f"📊 Optimal sample size: {optimal_samples:,} / {total_samples:,} "
          f"({optimal_samples/total_samples*100:.1f}%)")
    
    return optimal_samples


def optimize_memory_usage() -> None:
    """Force garbage collection and memory optimization."""
    gc.collect()
    
    
def get_memory_adaptive_config() -> Dict[str, Any]:
    """Get memory-adaptive configuration for training."""
    total_memory = get_system_memory_gb()
    
    if total_memory >= 24:  # High-memory system
        return {
            "use_full_dataset": True,
            "batch_size": 10000,
            "max_sample_size": None,
            "n_jobs": -1,
            "enable_early_stopping": False
        }
    elif total_memory >= 16:  # Medium-memory system  
        return {
            "use_full_dataset": False,
            "batch_size": 5000,
            "max_sample_size": 500000,  # 500k samples
            "n_jobs": -1,
            "enable_early_stopping": True
        }
    else:  # Low-memory system
        return {
            "use_full_dataset": False,
            "batch_size": 2000,
            "max_sample_size": 200000,  # 200k samples
            "n_jobs": 2,
            "enable_early_stopping": True
        }


class MemoryMonitor:
    """Context manager for monitoring memory usage during training."""
    
    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_memory = None
        
    def __enter__(self):
        self.start_memory = get_available_memory_gb()
        print(f"🔍 Starting {self.operation_name} (Available: {self.start_memory:.1f}GB)")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_memory = get_available_memory_gb()
        memory_used = self.start_memory - end_memory
        print(f"✅ {self.operation_name} complete (Used: {memory_used:.1f}GB, "
              f"Remaining: {end_memory:.1f}GB)")