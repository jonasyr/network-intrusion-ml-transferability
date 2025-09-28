# Cross-Dataset Transferability of Machine Learning Models for Network Intrusion Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Abstract

This repository presents a comprehensive empirical investigation into the cross-dataset transferability of machine learning models for network intrusion detection. The central research question addresses: **"Inwieweit sind Machine-Learning-Modelle für Netzwerk-Anomalieerkennung zwischen verschiedenen Datensätzen übertragbar?"** (To what extent are machine learning models for network anomaly detection transferable between different datasets?)

The study systematically evaluates twelve machine learning algorithms across two fundamentally different network security datasets: the historical NSL-KDD (2009) and the contemporary CIC-IDS-2017, revealing significant generalization challenges and providing empirical evidence for optimal model selection in heterogeneous network environments.

## Research Context & Motivation

With cybersecurity damages projected to exceed $10.5 trillion annually by 2025, network intrusion detection systems (IDS) have become critical infrastructure components. Traditional signature-based approaches fail against zero-day exploits, while machine learning-based solutions promise adaptive detection capabilities. However, a critical knowledge gap exists regarding the **cross-domain generalizability** of ML models when deployed across different network environments with varying traffic characteristics, attack vectors, and feature distributions.

This research addresses three fundamental questions in network security ML:
1. **Performance Transfer**: How do baseline and advanced ML models perform when trained on one dataset and evaluated on another?
2. **Generalization Asymmetry**: Are transfer patterns symmetric between historical (NSL-KDD) and modern (CIC-IDS-2017) datasets?
3. **Practical Deployment**: Which algorithms offer the optimal balance between within-dataset performance and cross-dataset robustness?

## Key Scientific Contributions

### Novel Cross-Dataset Evaluation Framework
- **Bidirectional Transfer Analysis**: Systematic evaluation of NSL-KDD ↔ CIC-IDS-2017 transfer patterns
- **Feature Space Harmonization**: PCA-based alignment methodology for heterogeneous feature representations
- **Transfer Metrics**: Introduction of Generalization Gap, Transfer Ratio, and Relative Performance Drop metrics

### Empirical Findings
- **Average Transfer Loss**: 38.6% performance degradation in cross-dataset scenarios
- **Asymmetric Generalization**: CIC-IDS-2017 → NSL-KDD transfer (0.807 mean ratio) outperforms NSL-KDD → CIC-IDS-2017 (0.618)
- **Algorithm Robustness Ranking**: XGBoost demonstrates superior cross-dataset stability, while LightGBM achieves peak single-dataset performance (99.9% accuracy)
- **Domain Divergence Impact**: Wasserstein distance correlation with transfer degradation provides predictive insights

### Methodological Innovations
- **Statistical Feature Alignment**: Standardization-based distribution matching for cross-dataset compatibility
- **Incremental Evaluation Pipeline**: Fault-tolerant experimental framework with automatic result recovery
- **Comprehensive Efficiency Analysis**: Training/inference time profiling for real-world deployment assessment

## Repository Architecture

```
network-intrusion-ml-transferability/
│
├── 📄 Core Configuration
│   ├── README.md                     # This comprehensive documentation
│   ├── requirements.txt              # Pinned Python dependencies
│   ├── setup.py                     # Package installation configuration
│   ├── LICENSE                      # MIT license for reproducibility
│   ├── validate_environment.py      # System compatibility checker
│   └── enable_scientific_mode.py    # Reproducibility configuration
│
├── 🗂️ data/                          # Experimental datasets and outputs
│   ├── raw/                         # Original dataset files
│   │   ├── nsl-kdd/                 # NSL-KDD training/testing files
│   │   │   ├── KDDTrain+.txt        # NSL-KDD training data (125,973 samples)
│   │   │   └── KDDTest+.txt         # NSL-KDD test data (22,544 samples)
│   │   └── cic-ids-2017/           # CIC-IDS-2017 network captures
│   │       └── full_dataset/        # Complete 5-day network simulation
│   ├── models/                      # Trained model artifacts
│   │   ├── baseline/               # Random Forest, Decision Tree, k-NN models
│   │   ├── advanced/               # XGBoost, LightGBM, Neural Network models
│   │   ├── nsl_baseline/           # NSL-KDD optimized baselines
│   │   ├── nsl_advanced/           # NSL-KDD advanced models
│   │   ├── cic_baseline/           # CIC-IDS-2017 baseline models
│   │   └── cic_advanced/           # CIC-IDS-2017 advanced models
│   └── results/                    # Experimental outputs and analysis
│       ├── experiment_summary.csv   # Consolidated performance metrics
│       ├── experiment_summary.json  # Structured result metadata
│       ├── nsl_*_results.csv       # NSL-KDD intra-dataset results
│       ├── cic_*_results.csv       # CIC-IDS-2017 intra-dataset results
│       ├── *_cross_dataset_*.csv   # Bidirectional transfer analysis
│       ├── confusion_matrices/      # Classification visualization outputs
│       ├── roc_curves/             # ROC/AUC performance curves
│       ├── feature_importance/     # Model interpretability analysis
│       ├── learning_curves/        # Training convergence analysis
│       ├── precision_recall_curves/ # Precision-recall trade-offs
│       ├── scientific_analysis/    # Statistical significance tests
│       └── paper_figures/          # Publication-ready visualizations
│
├── 🔬 src/                          # Core research implementation
│   ├── __init__.py                 # Package initialization
│   ├── features/                   # Feature engineering and alignment
│   │   ├── __init__.py
│   │   └── feature_alignment.py    # Cross-dataset feature harmonization
│   ├── preprocessing/              # Data preparation pipelines
│   │   ├── __init__.py
│   │   ├── nsl_kdd_preprocessor.py # NSL-KDD specific transformations
│   │   ├── cic_ids_preprocessor.py # CIC-IDS-2017 data processing
│   │   └── data_analyzer.py        # Exploratory data analysis utilities
│   ├── models/                     # Machine learning implementations
│   │   ├── __init__.py
│   │   ├── baseline_models.py      # Traditional ML algorithms (RF, DT, k-NN)
│   │   └── advanced_models.py      # Modern algorithms (XGBoost, LightGBM, NN)
│   ├── metrics/                    # Evaluation framework
│   │   ├── __init__.py
│   │   ├── cross_validation.py     # Within-dataset validation protocols
│   │   └── cross_dataset_metrics.py # Transfer learning evaluation metrics
│   ├── evaluation/                 # Enhanced analysis tools
│   │   └── enhanced_evaluation.py  # Scientific visualization and statistics
│   └── visualization/              # Result presentation
│       ├── __init__.py
│       └── paper_figures.py        # Publication-quality figure generation
│
├── 🧪 experiments/                  # Systematic experimental pipeline
│   ├── 01_data_exploration.py      # Dataset characterization and EDA
│   ├── 02_baseline_training.py     # Traditional algorithm benchmarking
│   ├── 03_advanced_training.py     # Modern algorithm evaluation
│   ├── 04_cross_validation.py      # Within-dataset robustness validation
│   ├── 05_cross_dataset_evaluation.py # Core transfer learning analysis
│   ├── 06_harmonized_evaluation.py # Feature-aligned cross-dataset testing
│   ├── 07_generate_results_summary.py # Consolidated result aggregation
│   ├── 08_generate_paper_figures.py # Scientific visualization generation
│   ├── 09_enhance_repository.py    # Repository optimization utilities
│   └── 10_*_visualizations.py      # Advanced scientific plotting
│
├── 🧩 tests/                       # Quality assurance framework
│   ├── __init__.py
│   ├── test_feature_alignment.py   # Feature harmonization validation
│   ├── test_harmonization.py       # Cross-dataset compatibility tests
│   └── test_io.py                  # Data pipeline integrity checks
│
└── 📚 docs/                        # Scientific documentation
    ├── methodology.md              # Detailed experimental methodology
    ├── results.md                  # Comprehensive result analysis
    └── Praxisprojekt_4/           # Academic paper (German)
        ├── Praxisprojekt_4.tex    # LaTeX manuscript source
        ├── Praxisprojekt_4.pdf    # Compiled scientific paper
        └── Praxisprojekt_4.bib    # Bibliography database
│
```

## Dataset Specifications & Acquisition

### NSL-KDD Dataset (Network Security Laboratory - Knowledge Discovery and Data Mining)
**Scientific Description**: Enhanced version of the KDD Cup 99 dataset, addressing critical limitations of duplicate records and biased distributions. Represents simulated network traffic from 1998 with comprehensive attack taxonomies.

**Technical Specifications**:
- **Training Set**: 125,973 records with 41 numerical/categorical features
- **Test Set**: 22,544 records (separate holdout for unbiased evaluation)
- **Attack Categories**: 4 main classes (DoS, Probe, R2L, U2R) + 22 specific attack types
- **Class Distribution**: Heavily imbalanced (Normal: 53%, DoS: 36%, Probe: 11%)
- **Feature Types**: Mixed (continuous flow statistics, categorical protocol indicators)

**Acquisition**:
```bash
# Download from Canadian Institute for Cybersecurity
wget https://www.unb.ca/cic/datasets/nsl.html
# Place in: data/raw/nsl-kdd/KDDTrain+.txt and data/raw/nsl-kdd/KDDTest+.txt
```

### CIC-IDS-2017 Dataset (Canadian Institute for Cybersecurity Intrusion Detection System)
**Scientific Description**: Contemporary network intrusion dataset capturing realistic attack scenarios over 5 days (July 3-7, 2017) in a controlled network environment with 25 users executing normal activities.

**Technical Specifications**:
- **Total Records**: ~2.8 million labeled network flows
- **Features**: 79 bidirectional flow statistics (duration, packet counts, flag distributions)
- **Attack Taxonomy**: 14 attack families including modern threats (Heartbleed, SQL Injection, XSS)
- **Temporal Structure**: Daily captures with varied attack intensities
- **Class Imbalance**: Severe (Normal: 83%, Attack: 17% across multiple categories)

**Acquisition**:
```bash
# Download from CIC Research Portal
wget https://www.unb.ca/cic/datasets/ids-2017.html
# Extract to: data/raw/cic-ids-2017/full_dataset/
# Alternative: Use provided sample in data/raw/cic-ids-2017/cic_ids_sample_backup.csv
```

### Dataset Compatibility Matrix
| Aspect | NSL-KDD | CIC-IDS-2017 | Alignment Challenge |
|--------|---------|-------------|-------------------|
| **Temporal Coverage** | 1998 simulation | 2017 real traffic | 19-year technology gap |
| **Feature Dimensionality** | 41 features | 79 features | Semantic feature mapping required |
| **Attack Sophistication** | Legacy protocols | Modern web/application attacks | Domain knowledge adaptation |
| **Data Volume** | 148K samples | 2.8M samples | Computational scalability |
| **Label Granularity** | Binary + 4 categories | Binary + 14 families | Hierarchical class mapping |

## System Requirements & Environment Setup

### Minimum Hardware Requirements
```
CPU: 4 cores, 2.5 GHz (Intel i5 equivalent or higher)
RAM: 8 GB (16 GB recommended for full CIC-IDS-2017 dataset)
Storage: 50 GB free space (datasets + intermediate results)
GPU: Optional (CUDA-capable for XGBoost/LightGBM acceleration)
```

### Python Environment
```bash
# Tested with Python 3.8-3.11
python --version  # Ensure >=3.8

# Create isolated environment
python -m venv network_ids_env
source network_ids_env/bin/activate  # Linux/macOS
# network_ids_env\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
python setup.py develop  # Development installation
```

### Dependency Overview
```python
# Core Scientific Computing
pandas>=1.3.0          # Data manipulation and analysis
numpy>=1.21.0           # Numerical computing foundation
scikit-learn>=1.2.0     # Machine learning algorithms and metrics

# Advanced ML Algorithms  
xgboost>=1.5.0          # Gradient boosting framework
lightgbm>=3.3.0         # Microsoft's gradient boosting
imbalanced-learn>=0.10.0 # Class imbalance handling (SMOTE)

# Scientific Visualization
matplotlib>=3.6.0       # Plotting and figure generation
seaborn>=0.11.0         # Statistical data visualization

# Performance and Utilities
scipy>=1.7.0           # Scientific computing (Wasserstein distance)
joblib>=1.2.0          # Parallel processing and model persistence
psutil>=5.8.0          # System resource monitoring
pydantic>=1.10.0       # Data validation and settings management
```

## Experimental Pipeline & Methodology

### Phase 1: Data Exploration and Characterization (`01_data_exploration.py`)
**Objective**: Console-based exploration of dataset characteristics and basic statistics.

**Key Analyses**:
- **Dataset Loading Verification**: Confirms NSL-KDD and CIC-IDS-2017 data accessibility
- **Basic Dataset Information**: Shape, columns, data types (console output only)
- **Class Distribution Analysis**: Label frequency analysis (console output only)
- **Sample Data Inspection**: Head/tail data previews for validation

**Outputs**: 
- Console output only - no files generated
- Verification of successful dataset loading and basic statistics

### Phase 2: Baseline Algorithm Benchmarking (`02_baseline_training.py`)
**Objective**: Establish performance baselines using traditional machine learning algorithms.

**Algorithms Evaluated**:
```python
# Traditional Classifiers
RandomForestClassifier(n_estimators=200, max_depth=25, class_weight='balanced')
DecisionTreeClassifier(max_depth=20, class_weight='balanced') 
KNeighborsClassifier(n_neighbors=5, weights='distance')
```

**Evaluation Protocol**:
- **Stratified 5-Fold Cross-Validation**: Maintains class proportions across folds
- **Performance Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Timing Analysis**: Training and inference latency measurement

### Phase 3: Advanced Algorithm Assessment (`03_advanced_training.py`)
**Objective**: Evaluate state-of-the-art ensemble methods and deep learning approaches.

**Advanced Algorithms**:
```python
# Gradient Boosting Ensembles
XGBClassifier(n_estimators=200, max_depth=8, learning_rate=0.1, scale_pos_weight=auto)
LGBMClassifier(n_estimators=200, max_depth=10, learning_rate=0.05, class_weight='balanced')

# Neural Network Architecture
MLPClassifier(hidden_layers=(128, 64, 32), activation='relu', solver='adam')
```

**Hyperparameter Optimization**:
- **Grid Search Strategy**: Systematic parameter space exploration
- **Validation Protocol**: Nested cross-validation for unbiased performance estimation
- **Early Stopping**: Prevents overfitting in iterative algorithms

### Phase 4: Cross-Dataset Transfer Learning Analysis (`05_cross_dataset_evaluation.py`)
**Objective**: Core scientific contribution - systematic evaluation of cross-domain generalization.

**Transfer Learning Framework**:

#### Feature Space Alignment Methodology
1. **Semantic Feature Mapping**: Domain knowledge-based feature correspondence
   ```python
   # Example mappings between datasets
   NSL_KDD_to_CIC_mapping = {
       'duration': 'Flow Duration',
       'src_bytes': 'Total Fwd Packets', 
       'dst_bytes': 'Total Backward Packets'
   }
   ```

2. **Statistical Harmonization**: Distribution standardization across domains
   ```python
   # Z-score normalization with robust scaling
   scaler = StandardScaler()
   source_normalized = scaler.fit_transform(source_features)
   target_normalized = scaler.transform(target_features)  # Uses source statistics
   ```

3. **Dimensionality Reduction**: PCA-based latent space projection
   ```python
   # Shared latent representation (20 components)
   pca = PCA(n_components=20, random_state=42)
   source_latent = pca.fit_transform(source_normalized)
   target_latent = pca.transform(target_normalized)  # Same transformation
   ```

#### Transfer Evaluation Protocol
**Bidirectional Assessment**:
- **Forward Transfer**: NSL-KDD (train) → CIC-IDS-2017 (test)
- **Reverse Transfer**: CIC-IDS-2017 (train) → NSL-KDD (test)

**Novel Transfer Metrics**:
```python
# Generalization Gap: Absolute performance difference
gap = source_performance - target_performance

# Relative Performance Drop: Proportional degradation  
relative_drop = (source_performance - target_performance) / source_performance

# Transfer Ratio: Retained performance fraction
transfer_ratio = target_performance / source_performance
```

### Phase 5: Statistical Validation & Scientific Analysis (`04_cross_validation.py`, `06_harmonized_evaluation.py`)
**Objective**: Rigorous statistical assessment of results significance and reproducibility.

**Validation Protocols**:
- **Stratified K-Fold Cross-Validation** (k=5): Balanced sampling across attack categories
- **Statistical Significance Testing**: Paired t-tests for performance comparisons
- **Effect Size Analysis**: Cohen's d for practical significance assessment
- **Confidence Interval Estimation**: Bootstrap-based uncertainty quantification

## Advanced Scientific Analysis Features

### Domain Divergence Quantification
**Wasserstein Distance Calculation**: Measures distributional differences between aligned feature spaces.
```python
def compute_domain_divergence(source_features, target_features):
    """Earth Mover's Distance between feature distributions."""
    divergences = []
    for feature_idx in range(source_features.shape[1]):
        distance = wasserstein_distance(
            source_features[:, feature_idx], 
            target_features[:, feature_idx]
        )
        divergences.append(distance)
    return np.mean(divergences)
```

**Scientific Interpretation**: Higher divergence values correlate with greater transfer difficulty, providing predictive insights for cross-domain deployment.

### Feature Importance Analysis
**Model-Agnostic Interpretability**: SHAP (SHapley Additive exPlanations) values for feature attribution across algorithms and datasets.

**Cross-Dataset Feature Stability**: Identification of universally important features vs. dataset-specific discriminators.

### Learning Curve Analysis
**Training Efficiency Assessment**: Convergence behavior analysis across algorithms and dataset sizes.
```python
# Sample size impact on cross-dataset transferability
train_sizes = np.logspace(3, 6, num=10)  # 1K to 1M samples
for size in train_sizes:
    transfer_performance = evaluate_transfer(model, source_subset=size)
```

### Computational Efficiency Profiling
**Timing Analysis Framework**:
```python
@measure_time
def training_efficiency_analysis(model, X_train, y_train):
    """Comprehensive timing analysis for model training."""
    start_time = time.perf_counter()
    model.fit(X_train, y_train)
    training_time = time.perf_counter() - start_time
    
    # Memory usage monitoring
    memory_peak = psutil.Process().memory_info().rss / 1024**2  # MB
    
    return {
        'training_time': training_time,
        'memory_usage_mb': memory_peak,
        'samples_per_second': len(X_train) / training_time
    }
```

## Quick Start & Execution Guide

### Environment Validation
```bash
# Verify system compatibility and dataset availability
python validate_environment.py

# Expected output:
# ✅ Python version and dependencies check
# ✅ Dataset availability verification
# ✅ System resource assessment
```

### Manual Experimental Pipeline
```bash
# Run experiments individually in sequence
python experiments/01_data_exploration.py
python experiments/02_baseline_training.py 
python experiments/03_advanced_training.py
python experiments/04_cross_validation.py
python experiments/05_cross_dataset_evaluation.py
python experiments/06_harmonized_evaluation.py
python experiments/07_generate_results_summary.py
python experiments/08_generate_paper_figures.py


```

### Result Analysis
```bash
# Key result files are automatically generated in data/results/
# experiment_summary.csv and experiment_summary.json contain consolidated results
# Individual experiment outputs are saved to dataset-specific CSV files
```

### Key Result Files
```bash
# Primary scientific outputs (generated by experiments)
data/results/experiment_summary.csv           # Consolidated performance table
data/results/experiment_summary.json          # Detailed metrics with metadata
data/results/bidirectional_cross_dataset_analysis.csv  # Core transfer analysis

# Dataset-specific results
data/results/nsl_baseline_results.csv         # NSL-KDD baseline model results
data/results/nsl_advanced_results.csv         # NSL-KDD advanced model results
data/results/cic_baseline_results.csv         # CIC-IDS-2017 baseline results
data/results/cic_advanced_results.csv         # CIC-IDS-2017 advanced results

# Cross-dataset transfer analysis
data/results/nsl_trained_tested_on_cic.csv    # NSL → CIC transfer results
data/results/cic_trained_tested_on_nsl.csv    # CIC → NSL transfer results

# Visualization outputs
data/results/paper_figures/                   # Publication-ready plots
data/results/confusion_matrices/              # Classification matrices
data/results/roc_curves/                      # ROC analysis
data/results/feature_importance/              # Feature analysis
```

## Reproducibility & Scientific Rigor

### Random Seed Control
All stochastic processes use `RANDOM_STATE = 42` for deterministic reproducibility:
```python
# Consistent across all experiments
np.random.seed(42)
random.seed(42)
sklearn.utils.check_random_state(42)
xgboost.set_config(verbosity=0, random_state=42)
```

### Experimental Logging
```python
# Comprehensive experiment tracking
experiment_metadata = {
    'timestamp': datetime.now().isoformat(),
    'python_version': sys.version,
    'package_versions': get_package_versions(),
    'system_info': platform.uname(),
    'dataset_checksums': compute_data_hashes(),
    'random_seed': 42
}
```

### Result Validation Protocol
- **Checksum Verification**: Ensures dataset integrity across runs
- **Performance Bounds Checking**: Validates results within expected ranges  
- **Statistical Consistency**: Confirms reproducibility across multiple runs
- **Cross-Platform Compatibility**: Tested on Linux, macOS, and Windows

## Citation & Academic Use

### Primary Citation
```bibtex
@mastersthesis{weirauch2025transferability,
  title={Inwieweit sind Machine-Learning-Modelle f{\"u}r Netzwerk-Anomalieerkennung zwischen verschiedenen Datens{\"a}tzen {\"u}bertragbar?},
  author={Weirauch, Jonas},
  year={2025},
  school={IU Internationale Hochschule},
  type={Bachelor's Thesis},
  address={Germany},
  note={Cross-dataset evaluation of ML models for network intrusion detection}
}
```

### Research Data Citation
```bibtex
@dataset{weirauch2025_network_ids_data,
  title={Cross-Dataset Network Intrusion Detection: Experimental Results and Analysis},
  author={Weirauch, Jonas},
  year={2025},
  publisher={GitHub},
  url={https://github.com/jonasyr/network-intrusion-ml-transferability},
  version={1.0.0}
}
```

### Related Work References
- **NSL-KDD Dataset**: Tavallaee, M., et al. (2009). "A detailed analysis of the KDD CUP 99 data set"
- **CIC-IDS-2017 Dataset**: Sharafaldin, I., et al. (2018). "Toward Generating a New Intrusion Detection Dataset"  
- **Transfer Learning Theory**: Pan, S. J., & Yang, Q. (2009). "A survey on transfer learning"

## License & Contribution Guidelines

### Open Science License
This research is released under the **MIT License** to promote reproducible science and collaborative research development.

### Contributing to the Research
We welcome contributions that enhance the scientific rigor and applicability of this work:

- **Algorithm Extensions**: Implementation of additional ML algorithms (deep learning, ensemble methods)
- **Dataset Integration**: Support for additional network security datasets (UNSW-NB15, CSE-CIC-IDS2018)
- **Evaluation Metrics**: Novel transfer learning assessment methodologies
- **Visualization Improvements**: Enhanced scientific plotting and interpretability tools

### Development Setup
```bash
git clone https://github.com/jonasyr/network-intrusion-ml-transferability.git
cd network-intrusion-ml-transferability
pip install -e .[dev]  # Development dependencies
pre-commit install    # Code quality hooks
pytest tests/         # Run test suite
```

### Research Ethics & Data Privacy
- All datasets used are publicly available and ethically collected
- No personally identifiable information is processed or stored
- Experimental protocols comply with reproducible research standards
- Results are reported with appropriate uncertainty quantification
