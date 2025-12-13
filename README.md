# Fraud Detection Synthetic Data Project

A Python-based fraud detection prototype for telecommunications that generates synthetic customer data with injected fraud patterns and uses clustering-based detection algorithms to identify suspicious records.

## Features

- **Synthetic Data Generation**: Creates realistic customer PII using Faker
- **Fraud Pattern Injection**: Injects various fraud patterns (near duplicates, typos, shared IBANs, synthetic identities)
- **Clustering-Based Detection**: Uses DBSCAN with custom string similarity metrics
- **Comprehensive Evaluation**: Precision, Recall, F1-Score metrics with detailed reporting

## Project Structure

```
fraud-detection-synthetic/
 pyproject.toml              # Project configuration and dependencies
 README.md                   # This file
 src/
    __init__.py
    generate_dataset.py     # Main data generation script
    detect_fraud.py         # Fraud detection pipeline
    evaluate.py             # Full pipeline with evaluation
    data/
       __init__.py
       generator.py        # Synthetic customer data generator
       fraud_injector.py   # Fraud pattern injection
¦    models/
       __init__.py
       preprocessing.py    # Data cleaning and feature engineering
       clustering.py       # DBSCAN clustering with string similarity
    evaluation/
        __init__.py
        metrics.py          # Precision, Recall, F1 metrics
 data/                       # Output directory (created at runtime)
     customer_dataset.csv    # Generated dataset
     detected_fraud.csv      # Detection results
     evaluation_report.txt   # Performance report
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
uv pip install faker pandas numpy scikit-learn jellyfish
```

## Usage

### Quick Start - Full Pipeline

Run the complete pipeline (generate -> detect -> evaluate):

```bash
uv run python -m src.evaluate
```

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
uv run python -m src.detect_fraud --input data/customer_dataset.csv
```

Options:
- `-i, --input`: Input CSV file path
- `-o, --output`: Output CSV file path
- `-e, --eps`: DBSCAN epsilon (default: 0.35)
- `-m, --min-samples`: Minimum cluster size (default: 2)
- `--no-blocking`: Disable blocking optimization
- `-d, --distance-metric`: jaro_winkler, levenshtein, or damerau

#### 3. Run Full Pipeline with Evaluation

```bash
uv run python -m src.evaluate --num-records 500 --fraud-ratio 0.2
```

## Fraud Types Detected

| Type | Description |
|------|-------------|
| `near_duplicate` | Same address, different name (identity theft indicator) |
| `typo_variant` | Slight variations in name/email (multiple account attempts) |
| `shared_iban` | Multiple accounts sharing the same IBAN |
| `synthetic_identity` | Mix of real and fabricated data |

## Algorithm Details

### Distance Metrics

The system uses string similarity metrics to compare records:

- **Jaro-Winkler**: Optimal for typos and minor variations
- **Levenshtein**: Edit distance normalized by length
- **Damerau-Levenshtein**: Includes transposition detection

### Clustering

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) groups similar records together based on a composite distance metric that weights multiple fields:

- Surname: 25%
- First Name: 20%
- Address: 20%
- IBAN: 20%
- Email: 15%

### Blocking

To handle large datasets efficiently, blocking groups records by shared characteristics before comparison, reducing O(n²) complexity.

## Example Output

```
============================================================
FRAUD DETECTION EVALUATION REPORT
============================================================

Dataset Statistics:
  Total Records:        575
  Actual Frauds:        75
  Detected as Fraud:    82

Confusion Matrix:
                    Predicted
                    Neg     Pos
  Actual  Neg       493     7
          Pos       13      62

Performance Metrics:
  Precision:  0.7561 (75.6%)
  Recall:     0.8267 (82.7%)
  F1-Score:   0.7898 (79.0%)
  Accuracy:   0.9652 (96.5%)

Detection Rate by Fraud Type:
----------------------------------------
  near_duplicate: 18/20 (90.0%)
  shared_iban: 15/18 (83.3%)
  typo_variant: 16/22 (72.7%)
  synthetic_identity: 13/15 (86.7%)
============================================================
```

## Testing

Run the test suite:

```bash
# Test data generation
uv run python -c "from src.data.generator import CustomerDataGenerator; g = CustomerDataGenerator(seed=42); print(g.generate_records(5))"

# Test preprocessing
uv run python -c "from src.models.preprocessing import DataPreprocessor; print(DataPreprocessor.normalize_text('Müller-Schmidt'))"

# Test metrics
uv run python -m src.evaluation.metrics
```

## Configuration

### Tuning Detection Sensitivity

| Parameter | Effect |
|-----------|--------|
| `eps ` | More strict matching (fewer clusters, higher precision) |
| `eps ` | Looser matching (more clusters, higher recall) |
| `min_samples ` | Requires larger groups to be flagged |
| Blocking disabled | More thorough but slower detection |

### Recommended Settings

| Scenario | eps | min_samples | Blocking |
|----------|-----|-------------|----------|
| High Precision | 0.25 | 2 | Yes |
| Balanced | 0.35 | 2 | Yes |
| High Recall | 0.45 | 2 | No |

## License

This project is for educational and prototyping purposes.

## References

- [Scikit-Learn Clustering Documentation](https://scikit-learn.org/stable/modules/clustering.html)
- [Faker Documentation](https://faker.readthedocs.io/en/master/)
- [Jellyfish String Similarity](https://github.com/jamesturk/jellyfish)
