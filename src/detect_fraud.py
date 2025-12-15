"""Fraud Detection Pipeline Script.

Main script that takes CSV input, runs the ensemble detection algorithm,
and outputs flagged records with explanations.

Enhanced with ensemble detection, explainability, and configuration support.
"""

import argparse
import time
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import load_config, FraudDetectionConfig
from src.models.ensemble import EnsembleDetector, FusionStrategy
from src.models.detectors import (
    IsolationForestDetector,
    LocalOutlierFactorDetector,
    DBSCANDetector,
    GraphDetector,
)
from src.models.explainer import FraudExplainer
from src.models.clustering import FraudDetector  # Legacy support
from src.utils.logging import get_logger, FraudLogger


def create_ensemble_from_config(
    config: FraudDetectionConfig
) -> EnsembleDetector:
    """
    Create an ensemble detector from configuration.
    
    Args:
        config: Fraud detection configuration.
        
    Returns:
        Configured EnsembleDetector.
    """
    detectors = []
    weights = config.ensemble.weights
    
    if config.detectors.dbscan.enabled:
        dbscan = DBSCANDetector(
            eps=config.detectors.dbscan.eps,
            min_samples=config.detectors.dbscan.min_samples,
            distance_metric=config.detectors.dbscan.distance_metric,
            use_sparse=config.detectors.dbscan.use_sparse,
        )
        detectors.append((dbscan, weights.get("dbscan", 0.4)))
    
    if config.detectors.isolation_forest.enabled:
        iso_forest = IsolationForestDetector(
            contamination=config.detectors.isolation_forest.contamination,
            n_estimators=config.detectors.isolation_forest.n_estimators,
            random_state=config.detectors.isolation_forest.random_state,
        )
        detectors.append((iso_forest, weights.get("isolation_forest", 0.4)))
    
    if config.detectors.lof.enabled:
        lof = LocalOutlierFactorDetector(
            n_neighbors=config.detectors.lof.n_neighbors,
            contamination=config.detectors.lof.contamination,
        )
        detectors.append((lof, weights.get("lof", 0.1)))
    
    if config.detectors.graph.enabled:
        graph = GraphDetector(
            similarity_threshold=config.detectors.graph.similarity_threshold,
            min_community_size=config.detectors.graph.min_community_size,
            use_betweenness=config.detectors.graph.use_betweenness,
        )
        detectors.append((graph, weights.get("graph", 0.1)))
    
    ensemble = EnsembleDetector(
        detectors=detectors,
        strategy=config.ensemble.strategy,
        voting_threshold=config.ensemble.voting_threshold,
    )
    ensemble.set_threshold(config.ensemble.threshold)
    
    return ensemble


def detect_fraud(
    input_path: str,
    output_path: str = "data/detected_fraud.csv",
    eps: float = 0.35,
    min_samples: int = 2,
    use_blocking: bool = True,
    distance_metric: str = "jaro_winkler",
    config_path: Optional[str] = None,
    detectors: Optional[list[str]] = None,
    fusion_strategy: str = "weighted_avg",
    threshold: float = 0.5,
    explain: bool = False,
    logger: Optional[FraudLogger] = None,
) -> pd.DataFrame:
    """
    Run fraud detection pipeline on customer dataset.
    
    Args:
        input_path: Path to input CSV with customer data.
        output_path: Path for output CSV with detection results.
        eps: DBSCAN epsilon parameter (similarity threshold).
        min_samples: Minimum cluster size.
        use_blocking: Use blocking to speed up computation.
        distance_metric: Distance metric to use.
        config_path: Path to YAML configuration file.
        detectors: List of detectors to use (dbscan, isolation_forest, lof, graph).
        fusion_strategy: Ensemble fusion strategy.
        threshold: Decision threshold for fraud classification.
        explain: Whether to generate explanations for flagged records.
        logger: Logger instance.
        
    Returns:
        DataFrame with fraud detection results.
    """
    log = logger or get_logger(format="text")
    start_time = time.time()
    
    print(f"=" * 60)
    print("FRAUD DETECTION PIPELINE")
    print(f"=" * 60)
    
    # Load configuration
    config = load_config(config_path)
    
    # Override config with CLI args if provided
    if detectors:
        config.detectors.dbscan.enabled = "dbscan" in detectors
        config.detectors.isolation_forest.enabled = "isolation_forest" in detectors
        config.detectors.lof.enabled = "lof" in detectors
        config.detectors.graph.enabled = "graph" in detectors
    
    config.ensemble.strategy = fusion_strategy
    config.ensemble.threshold = threshold
    config.detectors.dbscan.eps = eps
    config.detectors.dbscan.min_samples = min_samples
    config.detectors.dbscan.distance_metric = distance_metric
    
    # Load data
    print(f"\n[1/5] Loading data from {input_path}...")
    log.start_timer("data_loading")
    df = pd.read_csv(input_path)
    log.stop_timer("data_loading")
    print(f"       Loaded {len(df)} records")
    
    # Use ensemble detector
    print(f"\n[2/5] Initializing ensemble detector...")
    enabled_detectors = []
    if config.detectors.dbscan.enabled:
        enabled_detectors.append("DBSCAN")
    if config.detectors.isolation_forest.enabled:
        enabled_detectors.append("IsolationForest")
    if config.detectors.lof.enabled:
        enabled_detectors.append("LOF")
    if config.detectors.graph.enabled:
        enabled_detectors.append("Graph")
    
    print(f"      - Detectors: {', '.join(enabled_detectors)}")
    print(f"      - Fusion strategy: {fusion_strategy}")
    print(f"      - Threshold: {threshold}")
    
    ensemble = create_ensemble_from_config(config)
    
    # Fit and predict
    print(f"\n[3/5] Running fraud detection...")
    log.start_timer("detection")
    ensemble.fit(df)
    result_df = ensemble.predict(df)
    detection_time = log.stop_timer("detection")
    
    # Add results to original DataFrame
    df_result = df.copy()
    df_result["detected_fraud"] = result_df["is_fraud"]
    df_result["fraud_score"] = result_df["score"]
    df_result["detection_reason"] = result_df["reason"]
    
    # Add linked records information
    df_result["linked_records"] = ""
    df_result["cluster_id"] = -1
    
    # Extract cluster/community memberships from detectors
    for detector, _ in ensemble.detectors:
        if hasattr(detector, 'clusters') and detector.clusters:
            for cluster in detector.clusters:
                for idx in cluster.record_indices:
                    if idx < len(df_result):
                        # Get other customer IDs in this cluster
                        other_ids = [
                            df.iloc[i]["customer_id"] if "customer_id" in df.columns else f"row_{i}"
                            for i in cluster.record_indices if i != idx
                        ]
                        if other_ids:
                            current_linked = df_result.at[idx, "linked_records"]
                            new_linked = "; ".join(other_ids[:5])  # Limit to 5
                            if current_linked:
                                df_result.at[idx, "linked_records"] = f"{current_linked}; {new_linked}"
                            else:
                                df_result.at[idx, "linked_records"] = new_linked
                        df_result.at[idx, "cluster_id"] = cluster.cluster_id
        
        # Also check for graph communities
        if hasattr(detector, 'communities') and detector.communities:
            for comm_id, indices in enumerate(detector.communities):
                for idx in indices:
                    if idx < len(df_result):
                        other_ids = [
                            df.iloc[i]["customer_id"] if "customer_id" in df.columns else f"row_{i}"
                            for i in indices if i != idx
                        ]
                        if other_ids:
                            current_linked = df_result.at[idx, "linked_records"]
                            new_linked = "; ".join(other_ids[:5])
                            if current_linked:
                                df_result.at[idx, "linked_records"] = f"{current_linked}; {new_linked}"
                            else:
                                df_result.at[idx, "linked_records"] = new_linked
    
    # Also add shared IBAN links for any flagged records
    if "iban" in df.columns:
        iban_groups = df.groupby("iban").groups
        for iban, indices in iban_groups.items():
            if len(indices) > 1:
                for idx in indices:
                    if idx < len(df_result) and df_result.at[idx, "detected_fraud"]:
                        other_ids = [
                            df.iloc[i]["customer_id"] if "customer_id" in df.columns else f"row_{i}"
                            for i in indices if i != idx
                        ]
                        current_linked = str(df_result.at[idx, "linked_records"])
                        # Avoid duplicates
                        for oid in other_ids:
                            if oid not in current_linked:
                                if current_linked:
                                    current_linked = f"{current_linked}; {oid}"
                                else:
                                    current_linked = oid
                        df_result.at[idx, "linked_records"] = current_linked
    
    # Generate explanations if requested
    if explain:
        print(f"\n[4/5] Generating explanations...")
        explainer = FraudExplainer(ensemble)
        flagged_indices = df_result[df_result["detected_fraud"]].index.tolist()
        
        explanations = []
        for idx in flagged_indices[:100]:  # Limit to 100 for performance
            exp = ensemble.explain(df, idx)
            explanations.append(exp)
        
        # Save explanations
        explanation_path = Path(output_path).parent / "explanations.json"
        import json
        with open(explanation_path, "w") as f:
            json.dump(explanations, f, indent=2, default=str)
        print(f"       Saved {len(explanations)} explanations to {explanation_path}")
    else:
        print(f"\n[4/5] Skipping explanations (use --explain to enable)")
    
    num_clusters = sum(1 for d, _ in ensemble.detectors 
                      if hasattr(d, 'clusters') and d.clusters)
    
    # Save results
    print(f"\n[5/5] Saving results to {output_path}...")
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_result.to_csv(output_file, index=False)
    
    # Summary
    num_flagged = df_result["detected_fraud"].sum()
    total_time = time.time() - start_time
    
    print(f"\n{'=' * 60}")
    print("DETECTION SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total records:     {len(df)}")
    print(f"Records flagged:   {num_flagged} ({num_flagged/len(df):.1%})")
    print(f"Detection time:    {detection_time:.2f}s")
    print(f"Total time:        {total_time:.2f}s")
    print(f"Output file:       {output_file.absolute()}")
    print(f"{'=' * 60}")
    
    log.log_detection_result(
        num_records=len(df),
        num_flagged=int(num_flagged),
        num_clusters=num_clusters,
        duration=total_time
    )
    
    return df_result


def main():
    """Main entry point for fraud detection."""
    parser = argparse.ArgumentParser(
        description="Detect fraud patterns in customer dataset"
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        default="data/customer_dataset.csv",
        help="Input CSV file path (default: data/customer_dataset.csv)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="data/detected_fraud.csv",
        help="Output CSV file path (default: data/detected_fraud.csv)"
    )
    parser.add_argument(
        "-e", "--eps",
        type=float,
        default=0.35,
        help="DBSCAN epsilon - max distance for clustering (default: 0.35)"
    )
    parser.add_argument(
        "-m", "--min-samples",
        type=int,
        default=2,
        help="Minimum samples to form a cluster (default: 2)"
    )
    parser.add_argument(
        "--no-blocking",
        action="store_true",
        help="Disable blocking (slower but more thorough)"
    )
    parser.add_argument(
        "-d", "--distance-metric",
        type=str,
        default="jaro_winkler",
        choices=["jaro_winkler", "levenshtein", "damerau"],
        help="Distance metric to use (default: jaro_winkler)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--detectors",
        type=str,
        nargs="+",
        choices=["dbscan", "isolation_forest", "lof", "graph"],
        help="Detectors to use (default: from config)"
    )
    parser.add_argument(
        "--fusion-strategy",
        type=str,
        default="weighted_avg",
        choices=["max", "weighted_avg", "voting", "stacking"],
        help="Ensemble fusion strategy (default: weighted_avg)"
    )
    parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for fraud (default: 0.5)"
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Generate explanations for flagged records"
    )
    
    args = parser.parse_args()
    
    detect_fraud(
        input_path=args.input,
        output_path=args.output,
        eps=args.eps,
        min_samples=args.min_samples,
        use_blocking=not args.no_blocking,
        distance_metric=args.distance_metric,
        config_path=args.config,
        detectors=args.detectors,
        fusion_strategy=args.fusion_strategy,
        threshold=args.threshold,
        explain=args.explain,
    )


if __name__ == "__main__":
    main()
