# Plan: Scalable Modular Fraud Detection System (Optimized)

## Goals
1.  **Modular Architecture**: Define a standard `BaseDetector` interface.
2.  **Scalability**: Refactor DBSCAN to use sparse matrices + approximate nearest neighbors.
3.  **Multi-Modal Detection**: Add `IsolationForest` and `LocalOutlierFactor` detectors.
4.  **Ensemble**: Combine detectors with configurable fusion strategies.
5.  **[NEW] Explainability**: Add SHAP-based explanations for flagged records.
6.  **[NEW] Incremental Learning**: Support batch updates without full retraining.

---

## Step-by-Step Implementation

### 1. Define Detector Interface
*   **File**: `src/models/base.py`
*   **Action**: Create abstract base class `BaseDetector`.
*   **Contract**:
    *   `fit(df: pd.DataFrame) -> self`
    *   `predict(df: pd.DataFrame) -> pd.DataFrame` (columns: `score`, `is_fraud`, `reason`)
    *   **[NEW]** `predict_proba(df: pd.DataFrame) -> np.ndarray` (calibrated probabilities)
    *   **[NEW]** `explain(df: pd.DataFrame, idx: int) -> dict` (feature contributions)

### 2. Feature Engineering (Enhanced)
*   **File**: `src/models/features.py`
*   **Action**: Create `FeatureExtractor` class with rich features.
*   **Features**:
    | Category | Features |
    |----------|----------|
    | String Lengths | `len(surname)`, `len(first_name)`, `len(address)` |
    | Complexity | digit_count, special_char_count, uppercase_ratio |
    | Email | domain_length, local_part_length, has_numbers_in_local |
    | Phonetic | Soundex code, Metaphone encoding, NYSIIS |
    | **[NEW] Cross-Field** | `name_in_email` (bool), `surname_email_similarity`, `address_contains_name` |
    | **[NEW] Behavioral** | `iban_country_matches_nationality`, `email_domain_geo_match` |
    | **[NEW] Entropy** | `email_entropy`, `name_entropy` (randomness detection) |
    | **[NEW] Temporal** | `dob_is_weekend`, `dob_round_number` (Jan 1st, 2000 etc.) |
    | **[NEW] Network** | `shared_iban_count`, `shared_address_count`, `shared_email_domain_count` |

### 3. Implement Isolation Forest Detector
*   **File**: `src/models/detectors/isolation_forest.py`
*   **Logic**:
    *   Use `FeatureExtractor` to prepare data.
    *   Train `sklearn.ensemble.IsolationForest` with `contamination='auto'`.
    *   **[NEW]** Use `CalibratedClassifierCV` for proper probability calibration.
    *   Map decision function to 0-1 probability-like score.

### 4. **[NEW]** Implement Local Outlier Factor Detector
*   **File**: `src/models/detectors/lof.py`
*   **Why**: LOF detects local density deviations—useful for subtle fraud patterns that IF misses.
*   **Logic**:
    *   Use same `FeatureExtractor`.
    *   `sklearn.neighbors.LocalOutlierFactor(novelty=True)` for predict support.

### 5. Refactor DBSCAN Detector (Scalable)
*   **File**: `src/models/detectors/dbscan.py`
*   **Critical Improvements**:
    *   Replace `np.ones((n, n))` dense matrix with `scipy.sparse.lil_matrix`.
    *   **[NEW]** Use `pynndescent` or `hnswlib` for approximate nearest neighbors on large datasets.
    *   **[NEW]** Add adaptive epsilon selection using k-distance graph elbow method.
    *   Update blocking logic to populate only necessary entries.

```python
# Example sparse matrix approach
from scipy.sparse import lil_matrix

def compute_distance_matrix_sparse(df, block_indices, distance_fn):
    n = len(df)
    sparse_dist = lil_matrix((n, n), dtype=np.float32)
    
    for idx_i, idx_j in block_indices:
        dist = distance_fn(df.iloc[idx_i], df.iloc[idx_j])
        if dist < eps_threshold:  # Only store nearby pairs
            sparse_dist[idx_i, idx_j] = dist
            sparse_dist[idx_j, idx_i] = dist
    
    return sparse_dist.tocsr()
```

### 6. **[NEW]** Implement Graph-Based Detector
*   **File**: `src/models/detectors/graph.py`
*   **Why**: Fraud often forms networks (rings). Graph analysis finds connected components.
*   **Logic**:
    *   Build similarity graph using blocking.
    *   Use `networkx` or `igraph` for community detection.
    *   Flag records in dense subgraphs or with high betweenness centrality.

### 7. Implement Ensemble Detector (Enhanced)
*   **File**: `src/models/ensemble.py`
*   **Logic**:
    *   Accept list of initialized detectors with weights.
    *   **Fusion Strategies**:
        *   `max`: Flag if any detector flags (high recall)
        *   `weighted_avg`: Weighted average of scores
        *   **[NEW]** `voting`: Majority voting with threshold
        *   **[NEW]** `stacking`: Train meta-classifier on detector outputs
    *   **[NEW]** Add `threshold_optimizer` using ground truth to find optimal cutoff.

```python
class EnsembleDetector(BaseDetector):
    FUSION_STRATEGIES = ['max', 'weighted_avg', 'voting', 'stacking']
    
    def __init__(self, detectors: list[tuple[BaseDetector, float]], 
                 strategy: str = 'weighted_avg'):
        self.detectors = detectors
        self.strategy = strategy
        self.meta_classifier = None  # For stacking
    
    def optimize_threshold(self, df: pd.DataFrame, 
                          y_true: pd.Series,
                          metric: str = 'f1') -> float:
        """Find optimal threshold using precision-recall curve."""
        from sklearn.metrics import precision_recall_curve
        scores = self.predict(df)['score']
        precisions, recalls, thresholds = precision_recall_curve(y_true, scores)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        return thresholds[np.argmax(f1_scores)]
```

### 8. **[NEW]** Add Explainability Module
*   **File**: `src/models/explainer.py`
*   **Why**: Fraud analysts need to understand *why* a record was flagged.
*   **Logic**:
    *   Use `shap.TreeExplainer` for Isolation Forest.
    *   For DBSCAN, show nearest cluster members and field-level similarity breakdown.

```python
class FraudExplainer:
    def explain_isolation_forest(self, model, X, idx):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X.iloc[[idx]])
        return dict(zip(X.columns, shap_values[0]))
    
    def explain_dbscan(self, detector, df, idx):
        cluster_id = detector.labels_[idx]
        cluster_members = df[detector.labels_ == cluster_id]
        return {
            'cluster_size': len(cluster_members),
            'field_similarities': detector.get_field_contributions(df, idx)
        }
```

### 9. Update Pipeline & Evaluation (Enhanced)
*   **File**: `src/detect_fraud.py`
    *   Use `EnsembleDetector` as main entry point.
    *   Add CLI args: `--detectors`, `--fusion-strategy`, `--threshold`.
    *   **[NEW]** Add `--explain` flag to generate explanations for flagged records.
*   **File**: `src/evaluate.py`
    *   Track metrics per detector + ensemble.
    *   **[NEW]** Add confusion matrix visualization.
    *   **[NEW]** Add precision-recall curve plotting.
    *   **[NEW]** Add stratified evaluation by fraud type.

### 10. **[NEW]** Configuration Management
*   **File**: `src/config.py` + `config/default.yaml`
*   **Why**: Hyperparameters should be externalized, not hardcoded.
*   **Logic**:
    *   Use `pydantic` or `dataclasses` for typed config.
    *   Support YAML/JSON config files.
    *   Environment variable overrides.

```yaml
# config/default.yaml
detectors:
  dbscan:
    enabled: true
    eps: 0.35
    min_samples: 2
    distance_metric: jaro_winkler
  isolation_forest:
    enabled: true
    contamination: auto
    n_estimators: 100
  lof:
    enabled: false
    n_neighbors: 20

ensemble:
  strategy: weighted_avg
  threshold: 0.5
  weights:
    dbscan: 0.4
    isolation_forest: 0.4
    lof: 0.2
```

### 11. **[NEW]** Add Comprehensive Logging & Monitoring
*   **File**: `src/utils/logging.py`
*   **Why**: Production systems need observability.
*   **Logic**:
    *   Structured logging with `structlog`.
    *   Log detection latency, memory usage.
    *   Optional metrics export (Prometheus/StatsD).

### 12. **[NEW]** Fraud Pattern Library
*   **File**: `src/data/fraud_patterns.py`
*   **Why**: Expand injected fraud patterns for better testing.
*   **New Fraud Types**:
    | Pattern | Description |
    |---------|-------------|
    | `DEVICE_SHARING` | Multiple accounts from same "device fingerprint" |
    | `VELOCITY_FRAUD` | Many accounts created in short time window |
    | `RING_FRAUD` | Circular references between accounts |
    | `DATA_HARVESTING` | Sequential/patterned IBANs or emails |
    | `BIRTHDAY_PARADOX` | Unlikely DOB distributions (e.g., all Jan 1) |

---

## Updated Dependencies
```toml
# Add to pyproject.toml
dependencies = [
    # ...existing...
    "scipy>=1.12.0",         # Sparse matrices
    "shap>=0.44.0",          # Explainability
    "pynndescent>=0.5.0",    # Approximate NN (optional)
    "networkx>=3.2",         # Graph analysis (optional)
    "pydantic>=2.5.0",       # Config validation
    "structlog>=24.0.0",     # Structured logging
]
```

---

## Performance Benchmarks (New Section)
| Dataset Size | Current (dense) | Optimized (sparse+blocking) | Target |
|--------------|-----------------|----------------------------|--------|
| 1,000        | ~2s             | ~0.5s                      | <1s    |
| 10,000       | ~200s           | ~5s                        | <10s   |
| 100,000      | OOM             | ~60s                       | <120s  |

---

## Testing (Enhanced)
| Test ID | Description |
|---------|-------------|
| TEST-001 | Verify synthetic data contains no null values |
| TEST-002 | Verify injected fraud patterns are present |
| TEST-003 | Unit test distance functions |
| TEST-004 | Integration test full pipeline |
| **TEST-005** | **[NEW]** Benchmark memory usage at 50K records |
| **TEST-006** | **[NEW]** Test each fraud type detection in isolation |
| **TEST-007** | **[NEW]** Test ensemble produces better F1 than any single detector |
| **TEST-008** | **[NEW]** Test config loading from YAML |

---

## Risks & Mitigations (Updated)
| Risk | Impact | Mitigation |
|------|--------|------------|
| $O(N^2)$ complexity | High | Sparse matrices + blocking + ANN |
| **[NEW]** Model drift | Medium | Periodic retraining, monitoring |
| **[NEW]** Adversarial fraud | High | Ensemble diversity, human review |
| **[NEW]** False positive fatigue | Medium | Explainability, threshold tuning |