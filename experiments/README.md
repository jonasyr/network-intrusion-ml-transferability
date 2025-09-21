# Experimental Documentation

## Overview

This directory contains all experimental scripts for the research paper **"Machine Learning Models for Network Anomaly Detection: A Cross-Dataset Generalization Study"**.

## Experiment Sequence

The experiments should be run in the following order as later experiments depend on models trained in earlier ones:

### 1. Baseline Models (`01_baseline_models.py`)
- **Purpose**: Train fundamental ML models on NSL-KDD dataset
- **Models**: Random Forest, Logistic Regression, Decision Tree, k-NN, Naive Bayes, SVM
- **Output**: Trained models in `data/models/`, baseline results CSV
- **Duration**: ~5-10 minutes

### 2. Advanced Models (`02_advanced_models.py`)
- **Purpose**: Train ensemble and advanced ML models
- **Models**: XGBoost, LightGBM, Gradient Boosting, Extra Trees, MLP, Voting Classifier
- **Output**: Advanced models in `data/models/advanced/`
- **Duration**: ~10-15 minutes

### 3. Cross-Validation (`03_cross_validation.py`)
- **Purpose**: Statistical validation with 5-fold cross-validation
- **Analysis**: Pairwise t-tests, confidence intervals, significance testing
- **Output**: CV results in `data/results/cross_validation/`
- **Duration**: ~15-20 minutes

### 4. NSL-KDD → CIC-IDS-2017 (`04_cross_dataset_nsl_to_cic.py`)
- **Purpose**: Test generalization from NSL-KDD to CIC-IDS-2017
- **Key Finding**: 28.7% average accuracy drop
- **Output**: Cross-dataset results CSV
- **Duration**: ~5-10 minutes

### 5. CIC-IDS-2017 → NSL-KDD (`05_cross_dataset_cic_to_nsl.py`)
- **Purpose**: Test reverse generalization direction
- **Key Finding**: 14.7% average accuracy drop
- **Output**: Reverse cross-dataset results CSV
- **Duration**: ~5-10 minutes

### 6. Bidirectional Analysis (`06_bidirectional_analysis.py`)
- **Purpose**: Comprehensive analysis of both transfer directions
- **Key Finding**: Directional bias - NSL→CIC significantly harder
- **Output**: Combined analysis CSV and visualization
- **Duration**: ~2-5 minutes

### 7. Paper Figures (`07_generate_paper_figures.py`)
- **Purpose**: Generate all publication-ready figures
- **Output**: High-quality figures in `data/results/paper_figures/`
- **Duration**: ~5-10 minutes

## Running Experiments

### Individual Experiments
```bash
python experiments/01_baseline_models.py
python experiments/02_advanced_models.py
# ... etc
```

### Complete Pipeline
```bash
python run_all_experiments.py
```

## Key Research Findings

### Cross-Dataset Generalization Gap
- **NSL-KDD → CIC-IDS-2017**: 28.7% accuracy drop
- **CIC-IDS-2017 → NSL-KDD**: 14.7% accuracy drop
- **Directional Bias**: NSL-KDD models struggle with modern attacks

### Best Performing Models
1. **XGBoost**: 72.5% average generalization score
2. **LightGBM**: 65.8% average generalization score
3. **Random Forest**: 63.9% average generalization score

### Statistical Significance
- All results validated with 5-fold cross-validation
- Pairwise t-tests confirm significant differences
- 95% confidence intervals provided for all metrics

## Output Files

### Model Files
- `data/models/*.joblib`: Baseline models
- `data/models/advanced/*.joblib`: Advanced models
- `data/models/reverse_cross_dataset/*.joblib`: Cross-dataset trained models

### Result Files
- `baseline_results.csv`: Baseline model performance
- `cross_dataset_evaluation.csv`: NSL→CIC results
- `reverse_cross_dataset_evaluation.csv`: CIC→NSL results
- `bidirectional_cross_dataset_analysis.csv`: Combined analysis
- `cross_validation/cv_results_*.csv`: Statistical validation

### Visualizations
- `bidirectional_cross_dataset_analysis.png`: Main research figure
- `cv_results_boxplot.png`: Cross-validation comparison
- `paper_figures/`: Publication-ready figures

## Reproducibility

- All random seeds are fixed (random_state=42)
- Virtual environment ensures package consistency
- Complete dependency specification in requirements.txt
- Detailed logging and progress tracking

## Academic Contribution

These experiments provide the first comprehensive bidirectional cross-dataset evaluation in network intrusion detection literature, demonstrating significant generalization challenges and validating the need for domain adaptation research.