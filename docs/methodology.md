# Methodology

## Experimental Design

This study investigates the generalization capabilities of machine learning models for network intrusion detection across different datasets, specifically NSL-KDD and CIC-IDS-2017.

### Research Questions

1. How do ML models perform when trained on one network intrusion dataset and tested on another?
2. What is the magnitude of performance degradation in cross-dataset evaluation?
3. Which models show the best generalization capabilities?
4. How does feature harmonization affect cross-dataset performance?

### Datasets

#### NSL-KDD
- **Source**: Canadian Institute for Cybersecurity
- **Characteristics**: Classic intrusion detection benchmark
- **Size**: 125,973 training samples
- **Features**: 41 features including connection-level statistics
- **Classes**: Binary (benign/malicious) and multi-class attack categories

#### CIC-IDS-2017
- **Source**: Canadian Institute for Cybersecurity  
- **Characteristics**: Modern network traffic dataset
- **Size**: ~2.8M samples (full dataset), 10K sample for memory optimization
- **Features**: 78 flow-level features
- **Classes**: Binary (benign/malicious) and multi-class attack categories

### Feature Harmonization

Due to different feature spaces between datasets, we implemented a feature harmonization process:

1. **Common Feature Mapping**: Identified 13 overlapping features
2. **Schema Standardization**: Applied consistent naming and scaling
3. **Data Type Alignment**: Ensured compatible numeric types
4. **Missing Value Handling**: Median imputation for numeric, mode for categorical

### Models Evaluated

#### Baseline Models
- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine
- K-Nearest Neighbors
- Naive Bayes

#### Advanced Models
- XGBoost
- LightGBM
- Extra Trees
- Gradient Boosting
- Multi-layer Perceptron (MLP)
- Voting Classifier

### Evaluation Metrics

#### Within-Dataset Performance
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Cross-validation (3-fold stratified)

#### Cross-Dataset Performance
- **Transfer Ratio**: target_performance / source_performance
- **Generalization Gap**: source_performance - target_performance  
- **Relative Drop %**: (source_performance - target_performance) / source_performance × 100
- **Domain Divergence**: Statistical distance between feature distributions

### Experimental Pipeline

1. **Data Exploration** (`01_data_exploration.py`)
2. **Baseline Training** (`02_baseline_training.py`)
3. **Advanced Training** (`03_advanced_training.py`)
4. **Cross-Validation** (`04_cross_validation.py`)
5. **Cross-Dataset Evaluation** (`05_cross_dataset_evaluation.py`)
6. **Harmonized Evaluation** (`06_harmonized_evaluation.py`)
7. **Results Summary** (`07_generate_results_summary.py`)
8. **Paper Figures** (`08_generate_paper_figures.py`)

### Memory Optimization

Given the large size of CIC-IDS-2017, we implemented memory-adaptive strategies:

- **Adaptive Batch Processing**: Dynamic batch sizes based on available memory
- **Incremental Learning**: SGDClassifier for large-scale training
- **Streaming Data Processing**: Chunk-based file reading
- **Memory Monitoring**: Real-time memory usage tracking

### Statistical Validation

- **Cross-validation**: 3-fold stratified for robust performance estimates
- **Confidence Intervals**: Bootstrap estimation for metric uncertainty
- **Significance Testing**: t-tests for performance comparisons
- **Reproducibility**: Fixed random seeds (42) for all experiments
