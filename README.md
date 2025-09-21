# 🛡️ ML Network Anomaly Detection Research

## Eine experimentelle Analyse der Effektivität von Machine-Learning-Modellen für Anomalieerkennung im Netzwerkverkehr

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2+-orange.svg)](https://scikit-learn.org)
[![Research](https://img.shields.io/badge/Research-In%20Progress-yellow.svg)](https://github.com/jonasyr/ml-network-anomaly-detection)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Target**: 15-page scientific paper | **Timeline**: 10-12 weeks

---

## 📋 Table of Contents

- [🎯 Project Overview](#-project-overview)
- [✨ Features](#-features)
- [🚀 Quick Start](#-quick-start)
- [📊 Current Results](#-current-results)
- [🏗️ Project Structure](#️-project-structure)
- [🔬 Research Methodology](#-research-methodology)
- [📈 Progress Tracking](#-progress-tracking)
- [🛠️ Development](#️-development)
- [📚 References](#-references)

---

## 🎯 Project Overview

This research project investigates the effectiveness of various machine learning models for network anomaly detection, focusing on intrusion detection systems (IDS). The study compares traditional ML approaches with modern techniques using established datasets NSL-KDD and CIC-IDS-2017.

### 🎓 Research Question

> **"Wie effektiv sind Machine-Learning-Modelle für Anomalieerkennung im Netzwerkverkehr? Eine experimentelle Analyse mit NSL-KDD und CIC-IDS-2017"**

### 🎯 Objectives

- Compare effectiveness of different ML algorithms for network anomaly detection
- Analyze performance across different attack types and categories
- Evaluate cross-dataset generalization capabilities
- Provide comprehensive experimental analysis for academic publication

---

## ✨ Features

### ✅ **Currently Implemented**

#### 📊 **Data Analysis**

- ✅ NSL-KDD dataset integration
- ✅ Comprehensive data exploration
- ✅ Attack categorization (DoS, Probe, R2L, U2R)
- ✅ Statistical analysis pipeline
- ✅ Data quality validation

#### 🤖 **Machine Learning**

- ✅ Baseline model implementation
  - Random Forest
  - Logistic Regression
  - Decision Tree
  - K-Nearest Neighbors
  - Naive Bayes
  - SVM (Linear)
- ✅ Preprocessing pipeline
- ✅ Class balancing (SMOTE, Undersampling)
- ✅ Model evaluation framework
- ✅ Advanced model suite (XGBoost, LightGBM, Gradient Boosting, Extra Trees, MLP, Soft Voting)

#### 🔧 **Infrastructure**

- ✅ Modular code architecture
- ✅ Automated testing scripts
- ✅ Data validation pipeline
- ✅ Model persistence
- ✅ Results tracking

#### 📈 **Visualization**

- ✅ Attack distribution plots
- ✅ Feature analysis charts
- ✅ Model comparison graphics
- ✅ Interactive Jupyter notebooks

### 🔄 **In Progress**

- 🔄 CIC-IDS-2017 dataset integration
- 🔄 Advanced model implementations
- 🔄 Hyperparameter optimization
- 🔄 Cross-dataset evaluation

### 📋 **Planned**

- 📋 Deep learning models (MLP, Autoencoders)
- 📋 Ensemble methods
- 📋 Feature selection optimization
- 📋 Time-series analysis
- 📋 Real-time detection simulation
- 📋 Scientific paper publication

---

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.9+ required
python --version

# Git for version control
git --version
```

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/jonasyr/ml-network-anomaly-detection.git
cd ml-network-anomaly-detection

# 2. Set up Python environment
conda create -n anomaly-detection python=3.9
conda activate anomaly-detection

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify installation
python check_setup.py
```

### Dataset Setup

```bash
# Download NSL-KDD dataset
# Place files in data/raw/:
# - KDDTrain+.txt
# - KDDTest+.txt  
# - KDDTrain+_20Percent.txt

# Verify data
python check_data.py
```

### Quick Analysis

```bash
# Run baseline experiments
python scripts/run_baseline.py

# Run advanced experiments
python scripts/run_advanced.py

# Start Jupyter for detailed analysis
jupyter lab notebooks/01_data_exploration.ipynb

# Quick smoke test
python check_smoke.py
```

---

## 📊 Current Results

### Dataset Analysis (NSL-KDD 20% Subset)

| Metric | Value |
|--------|-------|
| **Total Records** | 25,192 |
| **Features** | 41 + 2 labels |
| **Attack Categories** | 5 (Normal, DoS, Probe, R2L, U2R) |
| **Attack Types** | 22 unique |
| **Data Quality** | ✅ Clean (no missing values) |

### Attack Distribution

```text
Normal Traffic: 53.4% (13,449 records)
DoS Attacks:    36.7% (9,234 records)  
Probe Attacks:   9.1% (2,289 records)
R2L Attacks:     0.8% (209 records)
U2R Attacks:     0.04% (11 records)
```

### Baseline Model Performance

| Model | Accuracy | F1-Score | Precision | Recall |
|-------|----------|----------|-----------|--------|
| **Random Forest** | 0.995 | 0.995 | 0.995 | 0.995 |
| **Decision Tree** | 0.993 | 0.993 | 0.993 | 0.993 |
| **Logistic Regression** | 0.945 | 0.944 | 0.946 | 0.945 |
| **K-Nearest Neighbors** | 0.967 | 0.967 | 0.968 | 0.967 |
| **Naive Bayes** | 0.821 | 0.815 | 0.835 | 0.821 |

> 📝 *Results on balanced validation set (binary classification: Normal vs Attack)*

---

## 🏗️ Project Structure

```text
ml-network-anomaly-detection/
├── 📊 data/
│   ├── raw/                    # Original datasets
│   ├── processed/              # Cleaned & preprocessed data
│   ├── models/                 # Trained model files
│   └── results/                # Analysis outputs
├── 📓 notebooks/
│   ├── 01_data_exploration.ipynb      # ✅ Data analysis
│   ├── 02_preprocessing.ipynb         # 🔄 Feature engineering
│   ├── 03_baseline_models.ipynb       # 📋 Model training
│   └── 04_evaluation.ipynb            # 📋 Results analysis
├── 🐍 src/
│   ├── data/
│   │   ├── preprocessor.py             # ✅ Data preprocessing
│   │   └── loader.py                   # ✅ Dataset loading
│   ├── models/
│   │   ├── baseline.py                 # ✅ Traditional ML models
│   │   └── advanced.py                 # 📋 Deep learning models
│   ├── evaluation/
│   │   └── metrics.py                  # ✅ Evaluation framework
│   └── nsl_kdd_analyzer.py             # ✅ Main analysis engine
├── 🧪 experiments/                     # Experiment configurations
├── 📈 reports/                         # Generated reports & papers
├── 🔧 scripts/
│   ├── run_baseline.py                 # ✅ Quick model training
│   └── run_experiments.py              # 📋 Full experiment suite
├── check_*.py                          # ✅ Validation scripts
├── requirements.txt                    # ✅ Dependencies
└── README.md                           # 📖 This file
```

**Legend:** ✅ Implemented | 🔄 In Progress | 📋 Planned

---

## 🔬 Research Methodology

### Phase 1: Foundation & Setup ✅

- [x] Literature review (initial)
- [x] Environment setup
- [x] Dataset acquisition and validation
- [x] Basic analysis pipeline

### Phase 2: Data Preprocessing ✅

- [x] NSL-KDD preprocessing pipeline
- [x] Feature analysis and selection
- [x] Class balancing strategies
- [x] Data validation framework

### Phase 3: Model Implementation 🔄

- [x] Baseline traditional ML models
- [x] Evaluation framework
- [ ] Advanced ensemble methods
- [ ] Deep learning approaches
- [ ] Hyperparameter optimization

### Phase 4: Experimentation 📋

- [ ] Cross-validation studies
- [ ] Cross-dataset evaluation
- [ ] Feature importance analysis
- [ ] Performance comparison

### Phase 5: Analysis & Documentation 📋

- [ ] Statistical significance testing
- [ ] Results interpretation
- [ ] Scientific paper writing
- [ ] Peer review preparation

---

## 📈 Progress Tracking

### ✅ Week 1-2 (Current Status)

- Complete project infrastructure
- NSL-KDD dataset integrated and analyzed
- Baseline models trained and evaluated
- Initial results documented

### 🔄 Week 3-4 (In Progress)

- CIC-IDS-2017 dataset integration
- Advanced model implementations
- Cross-dataset validation setup

### 📋 Week 5-8 (Planned)

- Comprehensive model comparison
- Feature engineering optimization
- Statistical analysis and testing
- Performance benchmarking

### 📋 Week 9-12 (Planned)

- Scientific paper writing
- Results validation and peer review
- Final documentation and submission

---

## 🛠️ Development

### Running Tests

```bash
# Environment validation
python check_setup.py

# Data integrity check
python check_data.py

# Quick functionality test
python check_smoke.py

# Full test suite
python -m pytest tests/
```

### Adding New Models

```python
# Example: Adding a new classifier
from src.models.baseline import BaselineModels

baseline = BaselineModels()
baseline.add_model('my_model', MyClassifier())
baseline.train_all(X_train, y_train)
```

---

## 📚 References

### Datasets

- **NSL-KDD**: [University of New Brunswick](https://www.unb.ca/cic/datasets/nsl.html)
- **CIC-IDS-2017**: [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/ids-2017.html)

### Key Literature

- Tavallaee, M., et al. (2009). "A detailed analysis of the KDD CUP 99 data set"
- Sharafaldin, I., et al. (2018). "Toward generating a new intrusion detection dataset and intrusion traffic characterization"

### Technical Stack

- **Python 3.9+**: Core programming language
- **Scikit-Learn**: Machine learning framework
- **Pandas/NumPy**: Data manipulation
- **Matplotlib/Seaborn**: Visualization
- **Jupyter**: Interactive analysis

---

## 🎓 Academic Research Project

Targeting 15-page scientific paper submission

**Progress**: Week 2 of 12 | **Status**: On Track ✅

---

**Last Updated**: June 28, 2025
