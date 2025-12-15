"""Configuration Management Module.

Provides typed configuration for fraud detection system with
support for YAML files, environment variable overrides, and validation.

Uses Pydantic v2 for robust configuration validation.
"""

import os
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class PathsConfig(BaseModel):
    """Configuration for file paths."""

    data_dir: str = Field(default="data", description="Base directory for data files")
    detected_fraud: str = Field(
        default="detected_fraud.csv", description="Output file for detected fraud"
    )
    customer_dataset: str = Field(
        default="customer_dataset.csv", description="Input customer dataset"
    )
    evaluation_report: str = Field(
        default="evaluation_report.txt", description="Evaluation report file"
    )
    explanations: str = Field(
        default="explanations.json", description="SHAP explanations output"
    )
    plots_dir: str = Field(default="plots", description="Directory for plot outputs")

    def get_path(self, file_key: str) -> Path:
        """Get full path for a data file.

        Args:
            file_key: One of 'detected_fraud', 'customer_dataset',
                     'evaluation_report', 'explanations'.

        Returns:
            Full path combining data_dir and the file name.
        """
        file_map = {
            "detected_fraud": self.detected_fraud,
            "customer_dataset": self.customer_dataset,
            "evaluation_report": self.evaluation_report,
            "explanations": self.explanations,
        }
        return Path(self.data_dir) / file_map.get(file_key, file_key)


class DBSCANConfig(BaseModel):
    """Configuration for DBSCAN detector."""

    enabled: bool = Field(default=True, description="Enable DBSCAN detector")
    eps: float = Field(
        default=0.35, ge=0.0, le=1.0, description="Max distance for clustering"
    )
    min_samples: int = Field(
        default=2, ge=1, description="Minimum samples to form cluster"
    )
    distance_metric: Literal["jaro_winkler", "levenshtein", "damerau"] = Field(
        default="jaro_winkler", description="String distance metric"
    )
    use_sparse: bool = Field(default=True, description="Use sparse distance matrix")


class IsolationForestConfig(BaseModel):
    """Configuration for Isolation Forest detector."""

    enabled: bool = Field(default=True, description="Enable Isolation Forest detector")
    contamination: str = Field(
        default="auto",
        description="Expected proportion of outliers, or 'auto'",
    )
    n_estimators: int = Field(
        default=100, ge=1, description="Number of base estimators"
    )
    max_samples: str = Field(default="auto", description="Samples per estimator")
    random_state: Optional[int] = Field(
        default=42, description="Random seed for reproducibility"
    )

    @field_validator("contamination")
    @classmethod
    def validate_contamination(cls, v: str) -> str:
        """Validate contamination is 'auto' or a valid float string."""
        if v == "auto":
            return v
        try:
            val = float(v)
            if not 0.0 < val <= 0.5:
                raise ValueError("contamination must be in (0.0, 0.5]")
        except ValueError:
            raise ValueError("contamination must be 'auto' or a float in (0.0, 0.5]")
        return v


class LOFConfig(BaseModel):
    """Configuration for Local Outlier Factor detector."""

    enabled: bool = Field(default=False, description="Enable LOF detector")
    n_neighbors: int = Field(default=20, ge=1, description="Number of neighbors for LOF")
    contamination: str = Field(
        default="auto", description="Expected proportion of outliers"
    )
    metric: str = Field(default="minkowski", description="Distance metric")


class GraphConfig(BaseModel):
    """Configuration for Graph-based detector."""

    enabled: bool = Field(default=False, description="Enable Graph detector")
    similarity_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Edge creation threshold"
    )
    min_community_size: int = Field(
        default=3, ge=2, description="Minimum community size"
    )
    use_betweenness: bool = Field(
        default=True, description="Use betweenness centrality"
    )
    betweenness_threshold: float = Field(
        default=0.1, ge=0.0, le=1.0, description="Betweenness threshold for flagging"
    )


class DetectorConfigs(BaseModel):
    """Container for all detector configurations."""

    dbscan: DBSCANConfig = Field(default_factory=DBSCANConfig)
    isolation_forest: IsolationForestConfig = Field(
        default_factory=IsolationForestConfig
    )
    lof: LOFConfig = Field(default_factory=LOFConfig)
    graph: GraphConfig = Field(default_factory=GraphConfig)


class EnsembleConfig(BaseModel):
    """Configuration for ensemble detector."""

    strategy: Literal["max", "weighted_avg", "voting", "stacking"] = Field(
        default="weighted_avg", description="Fusion strategy"
    )
    threshold: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Decision threshold"
    )
    voting_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Voting threshold for voting strategy"
    )
    weights: dict[str, float] = Field(
        default_factory=lambda: {
            "dbscan": 0.4,
            "isolation_forest": 0.4,
            "lof": 0.1,
            "graph": 0.1,
        },
        description="Detector weights for weighted averaging",
    )

    @field_validator("weights")
    @classmethod
    def validate_weights(cls, v: dict[str, float]) -> dict[str, float]:
        """Validate detector weights are non-negative."""
        for name, weight in v.items():
            if weight < 0:
                raise ValueError(f"Weight for {name} must be non-negative")
        return v


class LoggingConfig(BaseModel):
    """Configuration for logging."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Log level"
    )
    format: Literal["json", "text"] = Field(default="json", description="Log format")
    log_file: Optional[str] = Field(default=None, description="Log file path")
    log_metrics: bool = Field(default=True, description="Log performance metrics")


class DataConfig(BaseModel):
    """Configuration for data generation and processing."""

    num_records: int = Field(
        default=500, ge=1, description="Number of records to generate"
    )
    fraud_ratio: float = Field(
        default=0.15, ge=0.0, le=1.0, description="Proportion of fraudulent records"
    )
    seed: int = Field(default=42, description="Random seed")
    locale: str = Field(default="de_DE", description="Faker locale for data generation")
    input_path: str = Field(
        default="data/customer_dataset.csv", description="Input data path"
    )
    output_path: str = Field(
        default="data/detected_fraud.csv", description="Output data path"
    )


class FraudDetectionConfig(BaseModel):
    """Main configuration for the fraud detection system."""

    paths: PathsConfig = Field(default_factory=PathsConfig)
    detectors: DetectorConfigs = Field(default_factory=DetectorConfigs)
    ensemble: EnsembleConfig = Field(default_factory=EnsembleConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    data: DataConfig = Field(default_factory=DataConfig)

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def sync_paths_with_data(self) -> "FraudDetectionConfig":
        """Ensure data.input_path and data.output_path are consistent with paths."""
        # Sync input/output paths from paths config
        self.data.input_path = str(self.paths.get_path("customer_dataset"))
        self.data.output_path = str(self.paths.get_path("detected_fraud"))
        return self


def _apply_env_overrides(config: FraudDetectionConfig) -> FraudDetectionConfig:
    """Apply environment variable overrides to configuration.

    Supports the following environment variables:
    - FRAUD_DBSCAN_EPS, FRAUD_DBSCAN_MIN_SAMPLES, FRAUD_DBSCAN_ENABLED
    - FRAUD_IF_CONTAMINATION, FRAUD_IF_ESTIMATORS, FRAUD_IF_ENABLED
    - FRAUD_LOF_NEIGHBORS, FRAUD_LOF_ENABLED
    - FRAUD_GRAPH_ENABLED
    - FRAUD_ENSEMBLE_STRATEGY, FRAUD_ENSEMBLE_THRESHOLD
    - FRAUD_LOG_LEVEL, FRAUD_LOG_FILE
    - FRAUD_DATA_SEED, FRAUD_DATA_RECORDS
    """
    # Create a mutable copy of the config
    config_dict = config.model_dump()

    # DBSCAN overrides
    if os.getenv("FRAUD_DBSCAN_EPS"):
        config_dict["detectors"]["dbscan"]["eps"] = float(os.environ["FRAUD_DBSCAN_EPS"])
    if os.getenv("FRAUD_DBSCAN_MIN_SAMPLES"):
        config_dict["detectors"]["dbscan"]["min_samples"] = int(
            os.environ["FRAUD_DBSCAN_MIN_SAMPLES"]
        )
    if os.getenv("FRAUD_DBSCAN_ENABLED"):
        config_dict["detectors"]["dbscan"]["enabled"] = (
            os.environ["FRAUD_DBSCAN_ENABLED"].lower() == "true"
        )

    # Isolation Forest overrides
    if os.getenv("FRAUD_IF_CONTAMINATION"):
        config_dict["detectors"]["isolation_forest"]["contamination"] = os.environ[
            "FRAUD_IF_CONTAMINATION"
        ]
    if os.getenv("FRAUD_IF_ESTIMATORS"):
        config_dict["detectors"]["isolation_forest"]["n_estimators"] = int(
            os.environ["FRAUD_IF_ESTIMATORS"]
        )
    if os.getenv("FRAUD_IF_ENABLED"):
        config_dict["detectors"]["isolation_forest"]["enabled"] = (
            os.environ["FRAUD_IF_ENABLED"].lower() == "true"
        )

    # LOF overrides
    if os.getenv("FRAUD_LOF_NEIGHBORS"):
        config_dict["detectors"]["lof"]["n_neighbors"] = int(
            os.environ["FRAUD_LOF_NEIGHBORS"]
        )
    if os.getenv("FRAUD_LOF_ENABLED"):
        config_dict["detectors"]["lof"]["enabled"] = (
            os.environ["FRAUD_LOF_ENABLED"].lower() == "true"
        )

    # Graph overrides
    if os.getenv("FRAUD_GRAPH_ENABLED"):
        config_dict["detectors"]["graph"]["enabled"] = (
            os.environ["FRAUD_GRAPH_ENABLED"].lower() == "true"
        )

    # Ensemble overrides
    if os.getenv("FRAUD_ENSEMBLE_STRATEGY"):
        config_dict["ensemble"]["strategy"] = os.environ["FRAUD_ENSEMBLE_STRATEGY"]
    if os.getenv("FRAUD_ENSEMBLE_THRESHOLD"):
        config_dict["ensemble"]["threshold"] = float(
            os.environ["FRAUD_ENSEMBLE_THRESHOLD"]
        )

    # Logging overrides
    if os.getenv("FRAUD_LOG_LEVEL"):
        config_dict["logging"]["level"] = os.environ["FRAUD_LOG_LEVEL"]
    if os.getenv("FRAUD_LOG_FILE"):
        config_dict["logging"]["log_file"] = os.environ["FRAUD_LOG_FILE"]

    # Data overrides
    if os.getenv("FRAUD_DATA_SEED"):
        config_dict["data"]["seed"] = int(os.environ["FRAUD_DATA_SEED"])
    if os.getenv("FRAUD_DATA_RECORDS"):
        config_dict["data"]["num_records"] = int(os.environ["FRAUD_DATA_RECORDS"])

    return FraudDetectionConfig.model_validate(config_dict)


def load_config(config_path: Optional[str] = None) -> FraudDetectionConfig:
    """Load configuration from YAML file with environment variable overrides.

    Args:
        config_path: Path to YAML config file. If None, tries default locations:
                    1. config/default.yaml
                    2. Default FraudDetectionConfig values

    Returns:
        Validated FraudDetectionConfig object.

    Raises:
        ValueError: If configuration is invalid.
    """
    config_dict: dict = {}

    # Determine config file path
    if config_path:
        path = Path(config_path)
    else:
        # Try default locations
        path = Path("config/default.yaml")

    # Load from YAML if exists
    if path.exists():
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f) or {}

    # Create config from dict (Pydantic handles validation)
    try:
        config = FraudDetectionConfig.model_validate(config_dict)
    except Exception as e:
        raise ValueError(f"Invalid configuration: {e}") from e

    # Apply environment variable overrides
    config = _apply_env_overrides(config)

    return config


def save_config(config: FraudDetectionConfig, config_path: str) -> None:
    """Save configuration to YAML file.

    Args:
        config: Configuration to save.
        config_path: Path to save YAML file.
    """
    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    config_dict = config.model_dump()

    with open(path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)


def get_default_config() -> FraudDetectionConfig:
    """Get default configuration.

    Returns:
        FraudDetectionConfig with default values.
    """
    return FraudDetectionConfig()


if __name__ == "__main__":
    # Test configuration
    config = get_default_config()
    print("Default configuration:")
    print(f"  DBSCAN eps: {config.detectors.dbscan.eps}")
    print(f"  IF contamination: {config.detectors.isolation_forest.contamination}")
    print(f"  Ensemble strategy: {config.ensemble.strategy}")
    print(f"  Ensemble threshold: {config.ensemble.threshold}")
    print(f"  Data seed: {config.data.seed}")
    print(f"  Paths data_dir: {config.paths.data_dir}")

    # Test validation
    print("\nTesting validation...")
    try:
        config.detectors.dbscan.eps = 1.5  # Should fail
    except Exception as e:
        print(f"  Validation caught invalid eps: {type(e).__name__}")

    print("\nConfiguration test complete.")
