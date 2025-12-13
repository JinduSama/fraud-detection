"""
Configuration Management Module.

Provides typed configuration for fraud detection system with
support for YAML files and environment variable overrides.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

try:
    from pydantic import BaseModel, Field
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False


@dataclass
class DBSCANConfig:
    """Configuration for DBSCAN detector."""
    enabled: bool = True
    eps: float = 0.35
    min_samples: int = 2
    distance_metric: str = "jaro_winkler"
    use_sparse: bool = True


@dataclass
class IsolationForestConfig:
    """Configuration for Isolation Forest detector."""
    enabled: bool = True
    contamination: str = "auto"
    n_estimators: int = 100
    max_samples: str = "auto"
    random_state: Optional[int] = 42


@dataclass
class LOFConfig:
    """Configuration for Local Outlier Factor detector."""
    enabled: bool = False
    n_neighbors: int = 20
    contamination: str = "auto"
    metric: str = "minkowski"


@dataclass
class GraphConfig:
    """Configuration for Graph-based detector."""
    enabled: bool = False
    similarity_threshold: float = 0.7
    min_community_size: int = 3
    use_betweenness: bool = True
    betweenness_threshold: float = 0.1


@dataclass
class DetectorConfigs:
    """Container for all detector configurations."""
    dbscan: DBSCANConfig = field(default_factory=DBSCANConfig)
    isolation_forest: IsolationForestConfig = field(default_factory=IsolationForestConfig)
    lof: LOFConfig = field(default_factory=LOFConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)


@dataclass
class EnsembleConfig:
    """Configuration for ensemble detector."""
    strategy: str = "weighted_avg"
    threshold: float = 0.5
    voting_threshold: float = 0.5
    weights: dict = field(default_factory=lambda: {
        "dbscan": 0.4,
        "isolation_forest": 0.4,
        "lof": 0.1,
        "graph": 0.1
    })


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    level: str = "INFO"
    format: str = "json"
    log_file: Optional[str] = None
    log_metrics: bool = True


@dataclass
class DataConfig:
    """Configuration for data generation and processing."""
    num_records: int = 500
    fraud_ratio: float = 0.15
    seed: int = 42
    locale: str = "de_DE"
    input_path: str = "data/customer_dataset.csv"
    output_path: str = "data/detected_fraud.csv"


@dataclass
class FraudDetectionConfig:
    """Main configuration for the fraud detection system."""
    detectors: DetectorConfigs = field(default_factory=DetectorConfigs)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    data: DataConfig = field(default_factory=DataConfig)


def _deep_update(base_dict: dict, update_dict: dict) -> dict:
    """Recursively update a dictionary."""
    result = base_dict.copy()
    for key, value in update_dict.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


def _dict_to_config(data: dict) -> FraudDetectionConfig:
    """Convert a dictionary to configuration objects."""
    config = FraudDetectionConfig()
    
    if "detectors" in data:
        detectors = data["detectors"]
        if "dbscan" in detectors:
            for key, value in detectors["dbscan"].items():
                setattr(config.detectors.dbscan, key, value)
        if "isolation_forest" in detectors:
            for key, value in detectors["isolation_forest"].items():
                setattr(config.detectors.isolation_forest, key, value)
        if "lof" in detectors:
            for key, value in detectors["lof"].items():
                setattr(config.detectors.lof, key, value)
        if "graph" in detectors:
            for key, value in detectors["graph"].items():
                setattr(config.detectors.graph, key, value)
    
    if "ensemble" in data:
        for key, value in data["ensemble"].items():
            setattr(config.ensemble, key, value)
    
    if "logging" in data:
        for key, value in data["logging"].items():
            setattr(config.logging, key, value)
    
    if "data" in data:
        for key, value in data["data"].items():
            setattr(config.data, key, value)
    
    return config


def load_config(config_path: Optional[str] = None) -> FraudDetectionConfig:
    """
    Load configuration from YAML file with environment variable overrides.
    
    Args:
        config_path: Path to YAML config file. If None, uses default config.
        
    Returns:
        FraudDetectionConfig object.
    """
    config = FraudDetectionConfig()
    
    # Try to load from YAML file
    if config_path and HAS_YAML:
        path = Path(config_path)
        if path.exists():
            with open(path, "r") as f:
                yaml_data = yaml.safe_load(f)
                if yaml_data:
                    config = _dict_to_config(yaml_data)
    
    # Apply environment variable overrides
    config = _apply_env_overrides(config)
    
    return config


def _apply_env_overrides(config: FraudDetectionConfig) -> FraudDetectionConfig:
    """Apply environment variable overrides to configuration."""
    
    # DBSCAN overrides
    if os.getenv("FRAUD_DBSCAN_EPS"):
        config.detectors.dbscan.eps = float(os.environ["FRAUD_DBSCAN_EPS"])
    if os.getenv("FRAUD_DBSCAN_MIN_SAMPLES"):
        config.detectors.dbscan.min_samples = int(os.environ["FRAUD_DBSCAN_MIN_SAMPLES"])
    if os.getenv("FRAUD_DBSCAN_ENABLED"):
        config.detectors.dbscan.enabled = os.environ["FRAUD_DBSCAN_ENABLED"].lower() == "true"
    
    # Isolation Forest overrides
    if os.getenv("FRAUD_IF_CONTAMINATION"):
        config.detectors.isolation_forest.contamination = os.environ["FRAUD_IF_CONTAMINATION"]
    if os.getenv("FRAUD_IF_ESTIMATORS"):
        config.detectors.isolation_forest.n_estimators = int(os.environ["FRAUD_IF_ESTIMATORS"])
    if os.getenv("FRAUD_IF_ENABLED"):
        config.detectors.isolation_forest.enabled = os.environ["FRAUD_IF_ENABLED"].lower() == "true"
    
    # LOF overrides
    if os.getenv("FRAUD_LOF_NEIGHBORS"):
        config.detectors.lof.n_neighbors = int(os.environ["FRAUD_LOF_NEIGHBORS"])
    if os.getenv("FRAUD_LOF_ENABLED"):
        config.detectors.lof.enabled = os.environ["FRAUD_LOF_ENABLED"].lower() == "true"
    
    # Graph overrides
    if os.getenv("FRAUD_GRAPH_ENABLED"):
        config.detectors.graph.enabled = os.environ["FRAUD_GRAPH_ENABLED"].lower() == "true"
    
    # Ensemble overrides
    if os.getenv("FRAUD_ENSEMBLE_STRATEGY"):
        config.ensemble.strategy = os.environ["FRAUD_ENSEMBLE_STRATEGY"]
    if os.getenv("FRAUD_ENSEMBLE_THRESHOLD"):
        config.ensemble.threshold = float(os.environ["FRAUD_ENSEMBLE_THRESHOLD"])
    
    # Logging overrides
    if os.getenv("FRAUD_LOG_LEVEL"):
        config.logging.level = os.environ["FRAUD_LOG_LEVEL"]
    if os.getenv("FRAUD_LOG_FILE"):
        config.logging.log_file = os.environ["FRAUD_LOG_FILE"]
    
    # Data overrides
    if os.getenv("FRAUD_DATA_SEED"):
        config.data.seed = int(os.environ["FRAUD_DATA_SEED"])
    if os.getenv("FRAUD_DATA_RECORDS"):
        config.data.num_records = int(os.environ["FRAUD_DATA_RECORDS"])
    
    return config


def save_config(config: FraudDetectionConfig, config_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration to save.
        config_path: Path to save YAML file.
    """
    if not HAS_YAML:
        raise ImportError("PyYAML is required to save config. Install with: pip install pyyaml")
    
    # Convert to dictionary
    config_dict = {
        "detectors": {
            "dbscan": {
                "enabled": config.detectors.dbscan.enabled,
                "eps": config.detectors.dbscan.eps,
                "min_samples": config.detectors.dbscan.min_samples,
                "distance_metric": config.detectors.dbscan.distance_metric,
                "use_sparse": config.detectors.dbscan.use_sparse,
            },
            "isolation_forest": {
                "enabled": config.detectors.isolation_forest.enabled,
                "contamination": config.detectors.isolation_forest.contamination,
                "n_estimators": config.detectors.isolation_forest.n_estimators,
                "max_samples": config.detectors.isolation_forest.max_samples,
                "random_state": config.detectors.isolation_forest.random_state,
            },
            "lof": {
                "enabled": config.detectors.lof.enabled,
                "n_neighbors": config.detectors.lof.n_neighbors,
                "contamination": config.detectors.lof.contamination,
                "metric": config.detectors.lof.metric,
            },
            "graph": {
                "enabled": config.detectors.graph.enabled,
                "similarity_threshold": config.detectors.graph.similarity_threshold,
                "min_community_size": config.detectors.graph.min_community_size,
                "use_betweenness": config.detectors.graph.use_betweenness,
                "betweenness_threshold": config.detectors.graph.betweenness_threshold,
            },
        },
        "ensemble": {
            "strategy": config.ensemble.strategy,
            "threshold": config.ensemble.threshold,
            "voting_threshold": config.ensemble.voting_threshold,
            "weights": config.ensemble.weights,
        },
        "logging": {
            "level": config.logging.level,
            "format": config.logging.format,
            "log_file": config.logging.log_file,
            "log_metrics": config.logging.log_metrics,
        },
        "data": {
            "num_records": config.data.num_records,
            "fraud_ratio": config.data.fraud_ratio,
            "seed": config.data.seed,
            "locale": config.data.locale,
            "input_path": config.data.input_path,
            "output_path": config.data.output_path,
        },
    }
    
    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)


def get_default_config() -> FraudDetectionConfig:
    """Get default configuration."""
    return FraudDetectionConfig()


if __name__ == "__main__":
    # Test configuration
    config = get_default_config()
    print("Default configuration:")
    print(f"  DBSCAN eps: {config.detectors.dbscan.eps}")
    print(f"  IF contamination: {config.detectors.isolation_forest.contamination}")
    print(f"  Ensemble strategy: {config.ensemble.strategy}")
    print(f"  Data seed: {config.data.seed}")
