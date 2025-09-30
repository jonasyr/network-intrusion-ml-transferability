# Machine Learning für Netzwerk-Anomalieerkennung

> Empirische Untersuchung zur Cross-Dataset-Übertragbarkeit von ML-Modellen zwischen NSL-KDD und CIC-IDS-2017

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Überblick

Diese Arbeit untersucht systematisch die **Cross-Dataset-Transferabilität** von Machine Learning-Modellen für die Netzwerk-Intrusion-Detection. Kernfrage: *Inwieweit sind ML-Modelle für Netzwerk-Anomalieerkennung zwischen verschiedenen Datensätzen übertragbar?*

**Zentrale Ergebnisse:**
- 38.6% durchschnittlicher Leistungsverlust bei Cross-Dataset-Transfer
- XGBoost zeigt beste Cross-Dataset-Stabilität
- CIC-IDS-2017 → NSL-KDD Transfer erfolgreicher als umgekehrt

## Schnellstart

```bash
# Environment Setup
python -m venv network_ids_env
source network_ids_env/bin/activate
pip install -r requirements.txt

# Experimente ausführen
python validate_environment.py
python experiments/01_data_exploration.py
python experiments/05_cross_dataset_evaluation.py
```

## Projektstruktur

```
├── data/
│   ├── raw/                    # NSL-KDD & CIC-IDS-2017 Datasets
│   ├── models/                 # Trainierte Modelle
│   └── results/                # Experimentelle Ergebnisse
├── experiments/                # Experimentelle Pipeline (01-10)
├── src/                        # Core Implementation
└── docs/                       # Methodologie & Ergebnisse
```

## Ergebnisse

Die wichtigsten Resultate finden sich in:
- `data/results/experiment_summary.csv` - Konsolidierte Leistungsmetriken
- `data/results/bidirectional_cross_dataset_analysis.csv` - Transfer-Analyse
- `data/results/paper_figures/` - Publikationsreife Visualisierungen

## Reproduzierbarkeit

Alle Experimente verwenden `RANDOM_STATE = 42` für deterministische Ergebnisse. Getestet auf Python 3.8-3.11, Linux/macOS/Windows.
