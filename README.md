# Machine Learning Models for Network Anomaly Detection: A Cross-Dataset Generalization Study

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![Research: Complete](https://img.shields.io/badge/Research-Complete-green.svg)](https://github.com/jonasyr/ml-network-anomaly-detection)

## 📋 Overview

This repository contains the complete implementation and experimental framework for the research paper **"Machine Learning Models for Network Anomaly Detection: A Cross-Dataset Generalization Study"**. 

The study presents the first comprehensive **bidirectional cross-dataset evaluation** of machine learning models for network intrusion detection, revealing significant generalization challenges when models are transferred between different network datasets.

### 🔬 Key Research Contributions

- **Novel Evaluation Methodology**: First bidirectional cross-dataset evaluation protocol for network intrusion detection
- **Empirical Findings**: Quantified generalization gaps (28.7% accuracy drop NSL-KDD→CIC-IDS-2017, 14.7% drop CIC-IDS-2017→NSL-KDD)
- **Directional Bias Discovery**: NSL-KDD models struggle significantly more when applied to modern attack datasets
- **Model Comparison**: Comprehensive evaluation of 11 machine learning models across multiple datasets
- **Academic Impact**: Demonstrates limitations of single-dataset evaluation and validates need for domain adaptation

## 🏗️ Repository Structure

```
ml-network-anomaly-detection/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── LICENSE                            # MIT License
│
├── src/                               # Core source code
│   ├── nsl_kdd_analyzer.py           # Main analyzer class
│   ├── data/                         # Data preprocessing
│   │   ├── preprocessor.py           # NSL-KDD preprocessor
│   │   └── cic_ids_preprocessor.py   # CIC-IDS-2017 preprocessor
│   ├── models/                       # Model implementations
│   │   ├── baseline.py               # Baseline ML models
│   │   └── advanced.py               # Advanced ensemble models
│   ├── evaluation/                   # Evaluation frameworks
│   │   └── cross_validation.py       # Statistical validation
│   └── visualization/                # Figure generation
│       └── paper_figures.py          # Publication figures
│
├── experiments/                       # Experimental scripts
│   ├── 01_baseline_models.py         # Baseline model training
│   ├── 02_advanced_models.py         # Advanced model training
│   ├── 03_cross_validation.py        # Cross-validation analysis
│   ├── 04_cross_dataset_nsl_to_cic.py # NSL-KDD → CIC-IDS-2017
│   ├── 05_cross_dataset_cic_to_nsl.py # CIC-IDS-2017 → NSL-KDD
│   ├── 06_bidirectional_analysis.py  # Comprehensive analysis
│   └── 07_generate_paper_figures.py  # Publication figures
│
├── data/                             # Datasets and results
│   ├── raw/                          # Raw datasets
│   │   ├── KDDTrain+_20Percent.txt   # NSL-KDD training data
│   │   ├── KDDTest+.txt              # NSL-KDD test data
│   │   └── cic_ids_2017/             # CIC-IDS-2017 sample data
│   ├── models/                       # Trained models
│   │   ├── *.joblib                  # Baseline models
│   │   ├── advanced/                 # Advanced models
│   │   └── reverse_cross_dataset/    # Cross-dataset models
│   └── results/                      # Experimental results
│       ├── *.csv                     # Result tables
│       ├── *.png                     # Figures
│       ├── cross_validation/         # CV results
│       └── paper_figures/            # Publication figures
│
├── notebooks/                        # Jupyter analysis
│   └── 01_data_exploration.ipynb     # Data exploration
│
└── docs/                             # Documentation
    └── (generated documentation)
```

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- Virtual environment support
- ~2GB disk space for datasets and models

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/jonasyr/ml-network-anomaly-detection.git
   cd ml-network-anomaly-detection
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**
   ```bash
   python -c "import src.nsl_kdd_analyzer; print('✅ Installation successful')"
   ```

## 🧪 Running Experiments

The experiments are designed to be run sequentially, as later experiments depend on models trained in earlier ones.

### Core Experiments

1. **Baseline Models Training**
   ```bash
   python experiments/01_baseline_models.py
   ```
   Trains 6 baseline ML models on NSL-KDD dataset.

2. **Advanced Models Training**
   ```bash
   python experiments/02_advanced_models.py
   ```
   Trains advanced ensemble models (XGBoost, LightGBM, etc.).

3. **Cross-Validation Analysis**
   ```bash
   python experiments/03_cross_validation.py
   ```
   Performs 5-fold cross-validation with statistical significance testing.

### Cross-Dataset Evaluation (Key Contribution)

4. **NSL-KDD → CIC-IDS-2017 Transfer**
   ```bash
   python experiments/04_cross_dataset_nsl_to_cic.py
   ```
   Tests generalization from NSL-KDD to CIC-IDS-2017.

5. **CIC-IDS-2017 → NSL-KDD Transfer**
   ```bash
   python experiments/05_cross_dataset_cic_to_nsl.py
   ```
   Tests reverse generalization direction.

6. **Bidirectional Analysis**
   ```bash
   python experiments/06_bidirectional_analysis.py
   ```
   Comprehensive analysis of both transfer directions.

7. **Generate Publication Figures**
   ```bash
   python experiments/07_generate_paper_figures.py
   ```
   Creates all figures used in the research paper.

### Complete Pipeline

To run all experiments in sequence:
```bash
for script in experiments/*.py; do python "$script"; done
```

## 📊 Key Results

### Model Performance Summary

| Model | NSL-KDD Accuracy | CIC-IDS-2017 Accuracy | Avg. Generalization |
|-------|------------------|------------------------|-------------------|
| **XGBoost** | 79.1% | 49.9% | **72.5%** |
| **LightGBM** | 79.0% | 50.5% | 65.8% |
| **Random Forest** | 77.6% | 49.2% | 63.9% |

### Cross-Dataset Generalization Gaps

- **NSL-KDD → CIC-IDS-2017**: 28.7% average accuracy drop
- **CIC-IDS-2017 → NSL-KDD**: 14.7% average accuracy drop
- **Directional Bias**: NSL-KDD models transfer poorly to modern attacks

### Statistical Significance

All results include 95% confidence intervals and pairwise t-tests across 5-fold cross-validation.

## 📈 Generated Outputs

### Result Files
- `bidirectional_cross_dataset_analysis.csv`: Complete cross-dataset results
- `cross_validation_results.csv`: Statistical validation results
- `baseline_results.csv`: Baseline model performance

### Visualizations
- `bidirectional_cross_dataset_analysis.png`: Main research figure
- `cv_results_boxplot.png`: Cross-validation comparison
- `paper_figures/`: All publication-ready figures

### Trained Models
- 11 trained models in `data/models/`
- Cross-dataset specialized models in `data/models/reverse_cross_dataset/`

## 🔬 Research Methodology

### Datasets
- **NSL-KDD**: Standard benchmark dataset (41 features, 5 attack classes)
- **CIC-IDS-2017**: Modern intrusion dataset (77 features, realistic attacks)

### Models Evaluated
- **Baseline**: Logistic Regression, Decision Tree, Random Forest, k-NN, Naive Bayes, SVM
- **Advanced**: XGBoost, LightGBM, Gradient Boosting, Extra Trees, MLP, Voting Classifier

### Evaluation Protocol
1. **Within-dataset**: Traditional evaluation on same dataset
2. **Cross-dataset**: Train on one dataset, test on another
3. **Bidirectional**: Both transfer directions evaluated
4. **Statistical**: 5-fold CV with significance testing

## 🎓 Academic Usage

This repository serves as the **"Anhang" (Appendix/Source Code)** for the research paper. All experiments are fully reproducible and results can be regenerated.

### Citation
```bibtex
@article{author2025network,
  title={Machine Learning Models for Network Anomaly Detection: A Cross-Dataset Generalization Study},
  author={[Author Name]},
  journal={[Journal Name]},
  year={2025},
  note={Source code available at: https://github.com/jonasyr/ml-network-anomaly-detection}
}
```

### Reproducibility
- All random seeds are fixed for reproducibility
- Virtual environment ensures consistent package versions
- Complete experimental pipeline with clear dependencies

## �️ Technical Details

### Dependencies
- **Core**: scikit-learn, pandas, numpy
- **Advanced ML**: xgboost, lightgbm
- **Visualization**: matplotlib, seaborn
- **Processing**: imbalanced-learn (SMOTE)

### Performance Requirements
- **Memory**: ~4GB RAM for full experiments
- **Storage**: ~2GB for datasets and models
- **Time**: ~30-60 minutes for complete pipeline

### Python Version
Developed and tested on Python 3.12. Earlier versions may work but are not guaranteed.

## 📝 Documentation

### Code Documentation
- Comprehensive docstrings in all modules
- Type hints for better code understanding
- Inline comments explaining complex algorithms

### Experimental Documentation
- Each experiment script includes detailed headers
- Result files include metadata and timestamps
- Clear naming conventions throughout

## 🤝 Contributing

This is an academic research repository. For questions or discussions about the methodology:

1. Check the paper for detailed methodology
2. Review the experimental scripts for implementation details
3. Examine result files for comprehensive output

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## � Troubleshooting

### Common Issues

1. **Dataset not found**: Ensure NSL-KDD datasets are in `data/raw/`
2. **Package conflicts**: Use the provided `requirements.txt` in a fresh virtual environment
3. **Memory errors**: Reduce dataset size or use smaller model parameters
4. **CUDA warnings**: XGBoost/LightGBM may show GPU warnings; models will fall back to CPU

### Performance Optimization

- Use `n_jobs=-1` for parallel processing
- Consider reducing `n_estimators` for faster training during development
- Monitor memory usage during cross-validation experiments

## 📞 Contact

For academic inquiries related to this research:
- **Repository**: https://github.com/jonasyr/ml-network-anomaly-detection
- **Issues**: Use GitHub Issues for technical problems
- **Academic Discussion**: [Contact information from paper]

---

**Last Updated**: September 2025  
**Research Period**: September 2025  
**Paper Submission**: Academic Conference/Journal TBD

Targeting 15-page scientific paper submission

**Progress**: Week 2 of 12 | **Status**: On Track ✅

---

**Last Updated**: June 28, 2025
