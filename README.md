# Machine Learning Models for Network Anomaly Detection: A Cross-Dataset Generalization Study

## Abstract
This research investigates the generalization capabilities of machine learning models across different network intrusion detection datasets (NSL-KDD and CIC-IDS-2017), revealing significant performance degradation when models are applied across domains.

## Key Findings
- Average performance drop of 38.6% when transferring models between datasets (NSL-KDD → CIC-IDS-2017)
- Mean transfer ratio across models: 0.807 (best: XGBoost 0.618)
- LightGBM achieves best single-dataset performance (99.9% accuracy)
- Significant transfer asymmetry: CIC → NSL shows better generalization than NSL → CIC
- Cross-validation reveals high within-dataset stability (F1: 0.999 ± 0.0001)

## Repository Structure
```
ml-network-anomaly-detection/
├── README.md
├── requirements.txt
├── setup.py
├── LICENSE
├── .gitignore
│
├── data/
│   ├── raw/
│   │   ├── nsl-kdd/
│   │   └── cic-ids-2017/
│   ├── processed/
│   ├── models/
│   └── results/
│
├── src/
│   ├── __init__.py
│   ├── features/
│   │   ├── __init__.py
│   │   └── feature_alignment.py
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── nsl_kdd_preprocessor.py
│   │   ├── cic_ids_preprocessor.py
│   │   └── data_analyzer.py
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── cross_validation.py
│   │   └── cross_dataset_metrics.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baseline_models.py
│   │   └── advanced_models.py
│   └── visualization/
│       ├── __init__.py
│       └── paper_figures.py
│
├── experiments/
│   ├── 01_data_exploration.py
│   ├── 02_baseline_training.py
│   ├── 03_advanced_training.py
│   ├── 04_cross_validation.py
│   ├── 05_cross_dataset_evaluation.py
│   ├── 06_harmonized_evaluation.py
│   ├── 07_generate_results_summary.py
│   └── 08_generate_paper_figures.py
│
├── tests/
│   ├── __init__.py
│   └── test_feature_alignment.py
│
└── docs/
    ├── methodology.md
    ├── results.md
    └── figures/
```

## Data Setup
### Required Datasets

1. **NSL-KDD Dataset**
   - Download from: https://www.unb.ca/cic/datasets/nsl.html
   - Place files in: `data/raw/nsl-kdd/`
   - Required files: `KDDTrain+.txt`, `KDDTest+.txt`

2. **CIC-IDS-2017 Dataset**
   - Download from: https://www.unb.ca/cic/datasets/ids-2017.html
   - Place files in: `data/raw/cic-ids-2017/full_dataset/`
   - Or use sample file in: `data/raw/cic-ids-2017/cic_ids_sample_backup.csv`

### System Requirements
- Python 3.8+
- 16GB+ RAM recommended (8GB minimum with memory optimization)
- CUDA support optional for XGBoost/LightGBM acceleration

## Installation
```bash
git clone https://github.com/jonasyr/ml-network-anomaly-detection
cd ml-network-anomaly-detection
pip install -r requirements.txt
python setup.py install
```

## Quick Start
```bash
# Verify environment and data
python3 validate_environment.py

# Run complete experimental pipeline
python3 run_all_experiments.py

# Or run individual experiments
python3 experiments/01_data_exploration.py
python3 experiments/02_baseline_training.py
# ... etc

# Validate results
python3 validate_results.py
```

## Reproducibility
All experiments use fixed random seeds (42) for reproducibility. Results are saved in `data/results/` with timestamps.

## Citation
If you use this repository in your research, please cite it as:

```bibtex
@misc{ml_network_anomaly_detection_2025,
  title={Machine Learning Models for Network Anomaly Detection: A Cross-Dataset Generalization Study},
  author={Jonas Weirauch},
  year={2025},
  url={https://github.com/jonasyr/ml-network-anomaly-detection},
  note={Cross-dataset evaluation of ML models for network intrusion detection}
}
```
