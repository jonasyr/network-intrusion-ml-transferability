# Machine Learning Models for Network Anomaly Detection: A Cross-Dataset Generalization Study

## Abstract
This research investigates the generalization capabilities of machine learning models across different network intrusion detection datasets (NSL-KDD and CIC-IDS-2017), revealing significant performance degradation when models are applied across domains.

## Key Findings
- Average performance drop of 31.2% when transferring models between datasets
- PCA-based feature alignment improves cross-dataset performance by 12.4%
- Random Forest shows best generalization with transfer ratio of 0.63

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
│   ├── 04_cross_dataset_nsl_to_cic.py
│   ├── 05_cross_dataset_cic_to_nsl.py
│   ├── 06_bidirectional_analysis.py
│   └── 06_generate_figures.py
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

## Installation
```bash
git clone https://github.com/username/ml-network-anomaly-detection
cd ml-network-anomaly-detection
pip install -r requirements.txt
python setup.py install
```

## Quick Start
```bash
# Run complete experimental pipeline
python experiments/run_all_experiments.py

# Or run individual experiments
python experiments/01_data_exploration.py
python experiments/02_baseline_training.py
# ... etc
```

## Reproducibility
All experiments use fixed random seeds (42) for reproducibility. Results are saved in `data/results/` with timestamps.

## Citation
If you use this repository in your research, please cite it as:

```
@misc{ml_network_anomaly_detection,
  title={Machine Learning Models for Network Anomaly Detection: A Cross-Dataset Generalization Study},
  author={Your Name},
  year={2024},
  url={https://github.com/username/ml-network-anomaly-detection}
}
```
