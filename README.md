# Fraud Detection Synthetic Data Project

A Python-based fraud detection prototype for telecommunications that generates synthetic customer data with injected fraud patterns and uses **ensemble machine learning** and clustering-based detection algorithms to identify suspicious records.

## Features

- **Synthetic Data Generation**: Creates realistic customer PII using Faker
- **Fraud Pattern Injection**: Injects various fraud patterns (near duplicates, typos, shared IBANs, synthetic identities, and more)
- **Multi-Modal Detection**: Multiple detector types including Isolation Forest, LOF, DBSCAN, and Graph-based
- **Ensemble Methods**: Combine detectors with configurable fusion strategies (weighted average, voting, stacking)
- **Explainability**: SHAP-based explanations for flagged records
- **Comprehensive Evaluation**: Precision, Recall, F1-Score metrics with confusion matrix and stratified reporting

## Project Structure

```
fraud-detection-synthetic/
├── pyproject.toml              # Project configuration and dependencies
├── README.md                   # This file
├── config/
│   └── default.yaml            # Default configuration
├── src/
│   ├── __init__.py
│   ├── config.py               # Configuration management
│   ├── generate_dataset.py     # Main data generation script
│   ├── detect_fraud.py         # Fraud detection pipeline
│   ├── evaluate.py             # Full pipeline with evaluation
│   ├── data/
│   │   ├── __init__.py
│   │   ├── generator.py        # Synthetic customer data generator
│   │   ├── fraud_injector.py   # Fraud pattern injection
│   │   └── fraud_patterns.py   # Extended fraud patterns
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py             # BaseDetector abstract class
│   │   ├── features.py         # Feature engineering (36 features)
│   │   ├── preprocessing.py    # Data cleaning
│   │   ├── clustering.py       # Legacy DBSCAN clustering
│   │   ├── ensemble.py         # Ensemble detector with fusion
│   │   ├── explainer.py        # SHAP-based explainability
│   │   └── detectors/
│   │       ├── __init__.py
│   │       ├── isolation_forest.py  # Isolation Forest detector
│   │       ├── lof.py               # Local Outlier Factor detector
│   │       ├── dbscan.py            # DBSCAN with sparse matrices
│   │       └── graph.py             # Graph-based community detection
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py          # Evaluation metrics
│   └── utils/
│       ├── __init__.py
│       └── logging.py          # Structured logging
├── data/                       # Output directory (created at runtime)
│   ├── customer_dataset.csv    # Generated dataset
│   ├── detected_fraud.csv      # Detection results
│   ├── explanations.json       # SHAP explanations
│   └── evaluation_report.txt   # Performance report
└── test_features.py            # Feature test script
```

## Installation

### Prerequisites

- Python 3.10+
- uv package manager

### Setup

```bash
# Initialize with uv
uv sync

# Or install dependencies manually
uv pip install faker pandas numpy scikit-learn jellyfish scipy pyyaml

# Optional dependencies for full functionality
uv pip install shap networkx structlog pydantic
```

## Usage

### Quick Start - Full Pipeline

Run the complete pipeline (generate → detect → evaluate):

```bash
uv run python -m src.evaluate
```

### Parameter + Seed Sweep (Stability / Tuning)

Run a small grid over multiple random seeds and parameter settings and export a CSV summary:

```bash
uv run python -m src.sweep \
  --seeds 1 2 3 4 5 \
  --num-records 500 --fraud-ratio 0.02 \
  --detector-sets dbscan,isolation_forest dbscan,isolation_forest,graph \
  --eps 0.3 0.35 0.4 \
  --thresholds 0.6 0.7 0.8 \
  --fusion-strategies weighted_avg voting \
  --output data/sweeps/sweep_results.csv
```

This writes a sortable CSV with precision/recall/F1 + confusion-matrix counts per run.

### Individual Components

#### 1. Generate Synthetic Dataset

```bash
uv run python -m src.generate_dataset --num-records 1000 --fraud-ratio 0.15
```

Options:
- `-n, --num-records`: Number of legitimate records (default: 1000)
- `-f, --fraud-ratio`: Ratio of fraudulent records (default: 0.15)
- `-s, --seed`: Random seed for reproducibility (default: 42)
- `-o, --output`: Output CSV path (default: data/customer_dataset.csv)
- `-l, --locale`: Faker locale (default: de_DE)

#### 2. Run Fraud Detection

```bash
# Basic detection with ensemble (default)
uv run python -m src.detect_fraud --input data/customer_dataset.csv

# Single detector
uv run python -m src.detect_fraud --input data/customer_dataset.csv --detectors isolation_forest

# Multiple detectors with custom fusion
uv run python -m src.detect_fraud --input data/customer_dataset.csv \
    --detectors dbscan isolation_forest lof \
    --fusion-strategy voting

# With SHAP explanations
uv run python -m src.detect_fraud --input data/customer_dataset.csv --explain
```

Options:
- `-i, --input`: Input CSV file path
- `-o, --output`: Output CSV file path
- `-e, --eps`: DBSCAN epsilon (default: 0.35)
- `-m, --min-samples`: Minimum cluster size (default: 2)
- `--no-blocking`: Disable blocking optimization
- `-d, --distance-metric`: jaro_winkler, levenshtein, or damerau
- `--detectors`: Detector(s) to use: dbscan, isolation_forest, lof, graph
- `--fusion-strategy`: Ensemble fusion: max, weighted_avg, voting, stacking
- `--threshold`: Detection threshold (default: 0.5)
- `--explain`: Generate SHAP explanations for flagged records
- `--config`: Path to YAML configuration file

#### 3. Run Full Pipeline with Evaluation

```bash
uv run python -m src.evaluate --num-records 500 --fraud-ratio 0.2
```

## Detectors

### Available Detectors

| Detector | Description | Best For |
|----------|-------------|----------|
| `dbscan` | Density-based clustering with string similarity | Near duplicates, shared attributes |
| `isolation_forest` | Anomaly detection via random forests | Outliers, synthetic identities |
| `lof` | Local Outlier Factor | Local anomalies, unusual patterns |
| `graph` | Graph-based community detection | Ring fraud, coordinated attacks |

### Ensemble Fusion Strategies

| Strategy | Description |
|----------|-------------|
| `max` | Take maximum score from all detectors |
| `weighted_avg` | Weighted average of detector scores |
| `voting` | Majority voting (>50% detectors flag = fraud) |
| `stacking` | Meta-learner trained on detector outputs |

## Fraud Types Detected

| Type | Description |
|------|-------------|
| `near_duplicate` | Same address, different name (identity theft indicator) |
| `typo_variant` | Slight variations in name/email (multiple account attempts) |
| `shared_iban` | Multiple accounts sharing the same IBAN |
| `synthetic_identity` | Mix of real and fabricated data |
| `device_sharing` | Multiple accounts from same device fingerprint |
| `velocity_fraud` | Rapid account creation patterns |
| `ring_fraud` | Coordinated fraud ring activity |
| `data_harvesting` | Suspicious data collection patterns |
| `birthday_paradox` | Statistically unlikely DOB patterns |

## Feature Engineering

The system extracts 36 features across 8 categories:

1. **String Length Features**: Length of name, address, email fields
2. **Complexity Features**: Digit count, special character count
3. **Email Features**: Domain extraction, suspicious patterns
4. **Phonetic Features**: Soundex, Metaphone, NYSIIS encodings
5. **Cross-Field Features**: Name-email consistency, address patterns
6. **Behavioral Features**: Account creation patterns
7. **Entropy Features**: Information entropy of string fields
8. **Temporal Features**: Time-based patterns (when available)

## Explainability

The system provides SHAP-based explanations for flagged records:

```bash
uv run python -m src.detect_fraud --input data/customer_dataset.csv --explain
```

Explanations are saved to `data/explanations.json` and include:
- Top contributing features for each flagged record
- Per-detector scores and weights (in ensemble mode)
- Feature contribution breakdown

Example explanation:
```json
{
  "index": 42,
  "score": 0.85,
  "is_fraud": true,
  "reason": "isolation_forest_anomaly (top: shared_email_domain_count, len_surname, surname_entropy)",
  "detector_results": {
    "IsolationForest": {"score": 0.92, "weight": 0.4},
    "DBSCAN": {"score": 0.78, "weight": 0.6}
  }
}
```

## Configuration

### YAML Configuration

Create a custom config file:

```yaml
# config/custom.yaml
dbscan:
  eps: 0.35
  min_samples: 2
  use_blocking: true

isolation_forest:
  contamination: 0.1
  n_estimators: 100
  random_state: 42

lof:
  n_neighbors: 20
  contamination: 0.1

graph:
  similarity_threshold: 0.7
  min_community_size: 3

ensemble:
  strategy: weighted_avg
  threshold: 0.5
  weights:
    dbscan: 0.3
    isolation_forest: 0.4
    lof: 0.2
    graph: 0.1
```

Use it:
```bash
uv run python -m src.detect_fraud --config config/custom.yaml --input data/customer_dataset.csv
```

### Tuning Detection Sensitivity

| Parameter | Effect |
|-----------|--------|
| `eps ↓` | More strict matching (fewer clusters, higher precision) |
| `eps ↑` | Looser matching (more clusters, higher recall) |
| `contamination ↓` | Fewer anomalies flagged (higher precision) |
| `contamination ↑` | More anomalies flagged (higher recall) |
| `threshold ↓` | More records flagged (higher recall) |
| `threshold ↑` | Fewer records flagged (higher precision) |

### Recommended Settings

| Scenario | Detectors | Fusion | Threshold |
|----------|-----------|--------|-----------|
| High Precision | isolation_forest, dbscan | voting | 0.6 |
| Balanced | dbscan, isolation_forest, lof | weighted_avg | 0.5 |
| High Recall | all detectors | max | 0.4 |

## Example Output

```
============================================================
FRAUD DETECTION EVALUATION REPORT
============================================================

Dataset Statistics:
  Total Records:        115
  Actual Frauds:        15
  Detected as Fraud:    69

Confusion Matrix:
                    Predicted
                    Neg     Pos
  Actual  Neg       46      54
          Pos       0       15

Performance Metrics:
  Precision:  0.2174 (21.7%)
  Recall:     1.0000 (100.0%)
  F1-Score:   0.3571 (35.7%)
  Accuracy:   0.5304 (53.0%)

Detection Rate by Fraud Type:
----------------------------------------
  near_duplicate: 5/5 (100.0%)
  shared_iban: 5/5 (100.0%)
  synthetic_identity: 3/3 (100.0%)
  typo_variant: 2/2 (100.0%)
============================================================
```

## Testing

Run the feature test suite:

```bash
# Run all feature tests
uv run python test_features.py

# Test individual detectors
uv run python -c "
from src.models import IsolationForestDetector, DBSCANDetector, EnsembleDetector
print('Imports successful!')
"
```

## API Usage

```python
import pandas as pd
from src.models import (
    IsolationForestDetector,
    DBSCANDetector,
    LocalOutlierFactorDetector,
    EnsembleDetector,
    FusionStrategy
)
from src.models.explainer import FraudExplainer

# Load data
df = pd.read_csv('data/customer_dataset.csv')

# Create ensemble detector
ensemble = EnsembleDetector(
    detectors=[
        (IsolationForestDetector(contamination=0.1), 0.4),
        (DBSCANDetector(eps=0.35), 0.6),
    ],
    strategy=FusionStrategy.WEIGHTED_AVG,
    threshold=0.5
)

# Fit and predict
ensemble.fit(df)
results = ensemble.predict(df)

# Get explanations
flagged_idx = df.index[results['is_fraud']].tolist()
explainer = FraudExplainer(ensemble)
for idx in flagged_idx[:5]:
    explanation = explainer.explain_ensemble(df, idx)
    print(f"Record {idx}: {explanation}")
```

## License

This project is for educational and prototyping purposes.

## References

- [Scikit-Learn Clustering Documentation](https://scikit-learn.org/stable/modules/clustering.html)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [NetworkX Documentation](https://networkx.org/)
- [Faker Documentation](https://faker.readthedocs.io/en/master/)
- [Jellyfish String Similarity](https://github.com/jamesturk/jellyfish)
