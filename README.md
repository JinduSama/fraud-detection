# Fraud Detection Synthetic Data Project

A Python-based fraud detection prototype that generates synthetic customer data with injected fraud patterns and uses **ensemble machine learning** to identify suspicious records.

## Table of Contents

- [Quick Start](#quick-start)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Production Scoring](#production-scoring)
- [Configuration](#configuration)
- [Detectors & Strategies](#detectors)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [API Usage](#api-usage)

---

## Quick Start

```bash
# 1. Install dependencies
uv sync

# 2. Run the complete pipeline (generate → detect → evaluate)
uv run python -m src.evaluate

# 3. View results
cat data/evaluation_report.txt
```

That's it! The pipeline will generate synthetic data, run fraud detection, and output performance metrics.

---

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
├── pyproject.toml              # Project config & dependencies
├── README.md                   # This file
├── config/
│   └── default.yaml            # Default configuration (paths, detector params)
├── src/
│   ├── config.py               # Pydantic v2 configuration with validation
│   ├── generate_dataset.py     # Synthetic data generation
│   ├── detect_fraud.py         # Fraud detection pipeline
│   ├── evaluate.py             # Full pipeline with evaluation
│   ├── sweep.py                # Parameter sweep for tuning
│   ├── train_production.py     # Train production models (monthly)
│   ├── data/
│   │   ├── generator.py        # Synthetic customer data generator
│   │   ├── fraud_injector.py   # Fraud pattern injection
│   │   └── fraud_patterns.py   # Fraud type definitions (FraudType enum)
│   ├── models/
│   │   ├── base.py             # BaseDetector abstract class
│   │   ├── features.py         # Feature engineering (36 features)
│   │   ├── preprocessing.py    # Data cleaning
│   │   ├── clustering.py       # DBSCAN clustering with string similarity
│   │   ├── ensemble.py         # Ensemble detector with fusion strategies
│   │   ├── explainer.py        # SHAP-based explainability
│   │   └── detectors/
│   │       ├── isolation_forest.py  # + save_model/load_model
│   │       ├── lof.py               # + save_model/load_model
│   │       ├── dbscan.py
│   │       └── graph.py
│   ├── scoring/                # NEW: Production real-time scoring
│   │   ├── intrinsic_features.py   # Features without comparison
│   │   ├── similarity_index.py     # FAISS-based ANN search
│   │   └── realtime.py             # RealTimeScorer class
│   ├── evaluation/
│   │   └── metrics.py          # Precision, Recall, F1, confusion matrix
│   └── utils/
│       ├── text.py             # Shared text normalization & string distances
│       ├── address.py          # Address normalization utilities
│       └── logging.py          # Structured logging
├── models/                     # Saved production models
│   └── production/             # Output from train_production.py
├── tests/                      # Test suite (76 tests)
│   ├── conftest.py             # Shared pytest fixtures
│   ├── test_config.py          # Configuration validation tests
│   ├── test_utils.py           # Utility function tests
│   ├── test_scoring/           # Real-time scoring tests
│   │   └── test_intrinsic_features.py
│   ├── test_evaluation/
│   │   └── test_metrics.py     # Metrics calculation tests
│   └── test_integration/
│       └── test_pipeline.py    # End-to-end detector tests
└── data/                       # Output directory (created at runtime)
    ├── customer_dataset.csv
    ├── detected_fraud.csv
    ├── explanations.json
    └── evaluation_report.txt
```

## Installation

**Prerequisites:** Python 3.10+ and [uv](https://docs.astral.sh/uv/) package manager

```bash
# Clone and install
git clone <repository-url>
cd agents
uv sync

# For production real-time scoring (FAISS)
uv pip install faiss-cpu
```

This installs all dependencies including Pydantic v2 for configuration validation.

## Usage

### Full Pipeline

```bash
# Run complete pipeline: generate data → detect fraud → evaluate
uv run python -m src.evaluate

# With custom parameters
uv run python -m src.evaluate --num-records 500 --fraud-ratio 0.2
```

### Step-by-Step Pipeline

| Step | Command | Output |
|------|---------|--------|
| 1. Generate Data | `uv run python -m src.generate_dataset` | `data/customer_dataset.csv` |
| 2. Detect Fraud | `uv run python -m src.detect_fraud` | `data/detected_fraud.csv` |
| 3. Evaluate | `uv run python -m src.evaluate` | `data/evaluation_report.txt` |

### Parameter Sweep (Tuning)

Run a small grid over multiple random seeds and parameter settings and export a CSV summary:

```bash
uv run python -m src.sweep \
  --seeds 1 2 3 4 5 \
  --num-records-list 200 500 1000 \
  --fraud-ratios 0.02 0.05 0.10 \
  --detector-sets dbscan,isolation_forest dbscan,isolation_forest,graph \
  --eps 0.3 0.35 0.4 \
  --thresholds 0.6 0.7 0.8 \
  --fusion-strategies weighted_avg voting \
  --output data/sweeps/sweep_results.csv
```

This writes a sortable CSV with precision/recall/F1 + confusion-matrix counts per run.

You can also write an aggregated file (mean/std across seeds per parameter combo):

```bash
uv run python -m src.sweep \
  --seeds 1 2 3 4 5 \
  --num-records-list 200 500 \
  --fraud-ratios 0.02 0.10 \
  --detector-sets dbscan,isolation_forest dbscan,isolation_forest,graph \
  --eps 0.3 0.35 0.4 \
  --thresholds 0.6 0.7 0.8 \
  --fusion-strategies weighted_avg voting \
  --aggregate \
  --output data/sweeps/sweep_results.csv
```

If you want broad exploration with a run cap, you can shuffle the grid so `--max-runs` samples the space:

```bash
uv run python -m src.sweep \
  --seeds 1 2 3 4 5 \
  --num-records-list 200 500 \
  --fraud-ratios 0.02 0.10 \
  --detector-sets dbscan,isolation_forest dbscan,isolation_forest,graph \
  --eps 0.3 0.35 0.4 \
  --thresholds 0.6 0.7 0.8 \
  --fusion-strategies weighted_avg voting \
  --shuffle --shuffle-seed 0 \
  --max-runs 200 \
  --aggregate \
  --output data/sweeps/sweep_results.csv
```

### Generate Synthetic Dataset

```bash
uv run python -m src.generate_dataset --num-records 1000 --fraud-ratio 0.15
```

| Option | Description | Default |
|--------|-------------|---------|
| `-n, --num-records` | Number of legitimate records | 1000 |
| `-f, --fraud-ratio` | Ratio of fraudulent records | 0.15 |
| `-s, --seed` | Random seed for reproducibility | 42 |
| `-o, --output` | Output CSV path | `data/customer_dataset.csv` |
| `-l, --locale` | Faker locale | `de_DE` |

### Run Fraud Detection

```bash
# Basic detection with ensemble (default)
uv run python -m src.detect_fraud --input data/customer_dataset.csv

# Single detector
uv run python -m src.detect_fraud --detectors isolation_forest

# Multiple detectors with custom fusion
uv run python -m src.detect_fraud --detectors dbscan isolation_forest lof --fusion-strategy voting

# With SHAP explanations
uv run python -m src.detect_fraud --explain
```

**Key Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--detectors` | Detectors to use: `dbscan`, `isolation_forest`, `lof`, `graph` | all |
| `--fusion-strategy` | Ensemble fusion: `max`, `weighted_avg`, `voting`, `stacking` | `weighted_avg` |
| `--threshold` | Detection threshold (0.0-1.0) | 0.5 |
| `--eps` | DBSCAN epsilon | 0.35 |
| `--explain` | Generate SHAP explanations | false |
| `--config` | Path to YAML config file | `config/default.yaml` |

---

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

---

## Production Scoring

For real-world use with your own data, we provide a **two-tier scoring system** optimized for production:

| Tier | What | Speed | Use Case |
|------|------|-------|----------|
| **Tier 1: Intrinsic** | Features from record itself (IBAN validation, email patterns, entropy) | <10ms | Real-time scoring |
| **Tier 2: Similarity** | FAISS-based search against historical records | <50ms | Duplicate/ring detection |

### Training (Monthly)

Train the production scorer on historical data:

```bash
# Install production dependencies
uv pip install faiss-cpu

# Train on your historical data
uv run python -m src.train_production --input data/historical_customers.csv

# Or generate synthetic data for testing
uv run python -m src.train_production --generate --num-records 5000
```

Models are saved to `models/production/` by default.

### Real-Time Scoring

```python
from src.scoring import RealTimeScorer, AlertLevel

# Load trained models
scorer = RealTimeScorer.load("models/production/")

# Score a new application
new_application = {
    "customer_id": "NEW-001",
    "surname": "Mueller",
    "first_name": "Hans",
    "email": "hans.mueller@gmail.com",
    "iban": "DE89370400440532013000",
    "street": "Hauptstrasse",
    "postal_code": "10115",
    "city": "Berlin",
    "date_of_birth": "1985-06-15",
}

result = scorer.score(new_application)

print(f"Alert Level: {result.alert_level.value}")  # HIGH, MEDIUM, or LOW
print(f"Score: {result.combined_score:.2f}")
print(f"Flags: {result.flags}")
print(f"Similar Records: {len(result.similar_records)}")

if result.alert_level == AlertLevel.HIGH:
    send_to_review_queue(new_application)
```

### Intrinsic Features (No Comparison Needed)

These features are extracted instantly from a single record:

| Feature | Fraud Signal |
|---------|--------------|
| `iban_valid` | Invalid checksum = fake bank account |
| `iban_country_matches_address` | Mismatch = identity mixing |
| `email_is_disposable` | Tempmail.com etc. = throwaway |
| `email_entropy` | High randomness = generated |
| `name_has_digits` | "John123" = data quality issue |
| `keyboard_pattern_score` | "asdfgh" = lazy fraud |
| `postal_code_valid` | Invalid format = fabricated |

### Input Schema for Real Data

| Column | Type | Required | Notes |
|--------|------|----------|-------|
| `customer_id` | string | ✅ | Unique identifier |
| `surname` | string | ✅ | Used for phonetic blocking |
| `first_name` | string | ✅ | |
| `email` | string | ✅ | Domain used for blocking |
| `iban` | string | ✅ | Validated, used for exact matching |
| `street` | string | ⚠️ Preferred | Structured address |
| `house_number` | string | ⚠️ Preferred | |
| `postal_code` | string | ⚠️ Preferred | |
| `city` | string | ⚠️ Preferred | |
| `date_of_birth` | date | ⚠️ Preferred | Important identity variable |
| `address` | string | Fallback | Used if structured parts missing |
| `is_fraud` | boolean | For validation | Ground truth (not needed for scoring) |

---

## Configuration

Configuration is managed via YAML files with **Pydantic v2 validation**. Invalid settings raise clear errors at startup.

### Configuration File

```yaml
# config/default.yaml
paths:
  data_dir: "data"
  detected_fraud: "data/detected_fraud.csv"
  customer_dataset: "data/customer_dataset.csv"
  evaluation_report: "data/evaluation_report.txt"

dbscan:
  eps: 0.35                    # 0.0-1.0, validated
  min_samples: 2
  distance_metric: jaro_winkler  # jaro_winkler | levenshtein | damerau
  use_blocking: true

isolation_forest:
  contamination: "0.1"         # "auto" or float 0.0-0.5
  n_estimators: 100
  random_state: 42

lof:
  n_neighbors: 20
  contamination: 0.1

graph:
  enabled: true
  similarity_threshold: 0.7
  min_community_size: 3

ensemble:
  strategy: weighted_avg        # max | weighted_avg | voting | stacking
  threshold: 0.5                # 0.0-1.0, validated
  weights:
    dbscan: 0.3
    isolation_forest: 0.4
    lof: 0.2
    graph: 0.1                  # negative weights raise error
```

### Environment Variable Overrides

Override any setting via environment variables:

```bash
# Format: FRAUD_DETECTION__<SECTION>__<KEY>
export FRAUD_DETECTION__DBSCAN__EPS=0.4
export FRAUD_DETECTION__ENSEMBLE__THRESHOLD=0.6
uv run python -m src.detect_fraud
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

The project includes a comprehensive test suite with 76 tests:

```bash
# Run all tests
uv run pytest tests/ -v

# Run specific test categories
uv run pytest tests/test_config.py -v      # Configuration validation
uv run pytest tests/test_utils.py -v        # Utility functions
uv run pytest tests/test_scoring/ -v        # Real-time scoring
uv run pytest tests/test_evaluation/ -v     # Metrics calculation
uv run pytest tests/test_integration/ -v    # End-to-end pipeline

# Run with coverage
uv run pytest tests/ --cov=src --cov-report=html
```

### Test Structure

| Test File | Coverage |
|-----------|----------|
| `test_config.py` | Pydantic validation, env overrides, path config |
| `test_utils.py` | Text normalization, string distances, address utilities |
| `test_scoring/` | Intrinsic features, IBAN validation, email patterns |
| `test_evaluation/test_metrics.py` | Precision, recall, F1, confusion matrix |
| `test_integration/test_pipeline.py` | All detectors, ensemble, feature extraction |

---

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
