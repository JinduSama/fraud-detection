"""
Fraud Detection Pipeline Script.

Main script that takes CSV input, runs the clustering algorithm,
and outputs clusters of suspicious records.

TASK-007: Develop pipeline that takes CSV input and outputs suspicious clusters.
"""

import argparse
from pathlib import Path

import pandas as pd

from src.models.clustering import FraudDetector


def detect_fraud(
    input_path: str,
    output_path: str = "data/detected_fraud.csv",
    eps: float = 0.35,
    min_samples: int = 2,
    use_blocking: bool = True,
    distance_metric: str = "jaro_winkler"
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
        
    Returns:
        DataFrame with fraud detection results.
    """
    print(f"=" * 60)
    print("FRAUD DETECTION PIPELINE")
    print(f"=" * 60)
    
    # Load data
    print(f"\n[1/4] Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"       Loaded {len(df)} records")
    
    # Initialize detector
    print(f"\n[2/4] Initializing fraud detector...")
    print(f"      - Distance metric: {distance_metric}")
    print(f"      - DBSCAN eps: {eps}")
    print(f"      - Min samples: {min_samples}")
    print(f"      - Use blocking: {use_blocking}")
    
    detector = FraudDetector(
        eps=eps,
        min_samples=min_samples,
        distance_metric=distance_metric
    )
    
    # Run detection
    print(f"\n[3/4] Running fraud detection...")
    clusters = detector.detect_clusters(df, use_blocking=use_blocking)
    
    # Get results
    result_df = detector.get_flagged_records(df)
    
    # Save results
    print(f"\n[4/4] Saving results to {output_path}...")
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_file, index=False)
    
    # Summary
    num_flagged = result_df["detected_fraud"].sum()
    print(f"\n{'=' * 60}")
    print("DETECTION SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total records:     {len(df)}")
    print(f"Clusters found:    {len(clusters)}")
    print(f"Records flagged:   {num_flagged} ({num_flagged/len(df):.1%})")
    print(f"Output file:       {output_file.absolute()}")
    print(f"{'=' * 60}")
    
    # Print cluster details
    if clusters:
        print("\nCluster Details:")
        print("-" * 60)
        for cluster in clusters:
            print(f"\n  Cluster {cluster.cluster_id}:")
            print(f"    Records:    {len(cluster.record_indices)}")
            print(f"    IDs:        {cluster.customer_ids[:5]}{'...' if len(cluster.customer_ids) > 5 else ''}")
            print(f"    Similarity: {cluster.similarity_score:.2%}")
            print(f"    Reason:     {cluster.detection_reason}")
    
    print()
    return result_df


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
    
    args = parser.parse_args()
    
    detect_fraud(
        input_path=args.input,
        output_path=args.output,
        eps=args.eps,
        min_samples=args.min_samples,
        use_blocking=not args.no_blocking,
        distance_metric=args.distance_metric
    )


if __name__ == "__main__":
    main()
