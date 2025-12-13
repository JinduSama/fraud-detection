"""
Full Pipeline Evaluation Script.

Runs the complete pipeline: Generate -> Detect -> Evaluate
and produces a comprehensive performance report.

TASK-009: Create reporting script for full pipeline execution.
"""

import argparse
from pathlib import Path

import pandas as pd

from src.generate_dataset import generate_dataset
from src.detect_fraud import detect_fraud
from src.evaluation.metrics import FraudMetrics


def run_full_pipeline(
    num_records: int = 500,
    fraud_ratio: float = 0.15,
    seed: int = 42,
    eps: float = 0.35,
    min_samples: int = 2,
    use_blocking: bool = True,
    output_dir: str = "data",
    locale: str = "de_DE"
) -> dict:
    """
    Run the full fraud detection pipeline with evaluation.
    
    Pipeline stages:
    1. Generate synthetic dataset with fraud patterns
    2. Run fraud detection algorithm
    3. Evaluate performance against ground truth
    
    Args:
        num_records: Number of legitimate records to generate.
        fraud_ratio: Ratio of fraudulent records.
        seed: Random seed for reproducibility.
        eps: DBSCAN epsilon parameter.
        min_samples: Minimum cluster size.
        use_blocking: Use blocking for performance.
        output_dir: Directory for output files.
        locale: Faker locale for data generation.
        
    Returns:
        Dictionary with all results and metrics.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("   FRAUD DETECTION FULL PIPELINE EXECUTION")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Records:      {num_records}")
    print(f"  Fraud ratio:  {fraud_ratio:.0%}")
    print(f"  Seed:         {seed}")
    print(f"  DBSCAN eps:   {eps}")
    print(f"  Min samples:  {min_samples}")
    print(f"  Blocking:     {use_blocking}")
    print(f"  Locale:       {locale}")
    print()
    
    # Stage 1: Generate Dataset
    print("\n" + "-" * 70)
    print("STAGE 1: DATA GENERATION")
    print("-" * 70)
    
    dataset_path = str(output_path / "customer_dataset.csv")
    df_generated = generate_dataset(
        num_legitimate=num_records,
        fraud_ratio=fraud_ratio,
        seed=seed,
        output_path=dataset_path,
        locale=locale
    )
    
    # Stage 2: Fraud Detection
    print("\n" + "-" * 70)
    print("STAGE 2: FRAUD DETECTION")
    print("-" * 70)
    
    detection_path = str(output_path / "detected_fraud.csv")
    df_detected = detect_fraud(
        input_path=dataset_path,
        output_path=detection_path,
        eps=eps,
        min_samples=min_samples,
        use_blocking=use_blocking
    )
    
    # Stage 3: Evaluation
    print("\n" + "-" * 70)
    print("STAGE 3: EVALUATION")
    print("-" * 70)
    
    metrics = FraudMetrics()
    report = metrics.get_detailed_report(df_detected)
    print(report)
    
    # Save evaluation report
    report_path = output_path / "evaluation_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to: {report_path.absolute()}")
    
    # Return all results
    results = {
        "dataset": df_generated,
        "detection_results": df_detected,
        "metrics": metrics._results,
        "report": report,
        "files": {
            "dataset": dataset_path,
            "detection": detection_path,
            "report": str(report_path)
        }
    }
    
    print("\n" + "=" * 70)
    print("   PIPELINE COMPLETE")
    print("=" * 70 + "\n")
    
    return results


def main():
    """Main entry point for full pipeline execution."""
    parser = argparse.ArgumentParser(
        description="Run full fraud detection pipeline with evaluation"
    )
    parser.add_argument(
        "-n", "--num-records",
        type=int,
        default=500,
        help="Number of legitimate records (default: 500)"
    )
    parser.add_argument(
        "-f", "--fraud-ratio",
        type=float,
        default=0.15,
        help="Fraud ratio (default: 0.15)"
    )
    parser.add_argument(
        "-s", "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "-e", "--eps",
        type=float,
        default=0.35,
        help="DBSCAN epsilon (default: 0.35)"
    )
    parser.add_argument(
        "-m", "--min-samples",
        type=int,
        default=2,
        help="Minimum cluster size (default: 2)"
    )
    parser.add_argument(
        "--no-blocking",
        action="store_true",
        help="Disable blocking"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default="data",
        help="Output directory (default: data)"
    )
    parser.add_argument(
        "-l", "--locale",
        type=str,
        default="de_DE",
        help="Faker locale (default: de_DE)"
    )
    
    args = parser.parse_args()
    
    run_full_pipeline(
        num_records=args.num_records,
        fraud_ratio=args.fraud_ratio,
        seed=args.seed,
        eps=args.eps,
        min_samples=args.min_samples,
        use_blocking=not args.no_blocking,
        output_dir=args.output_dir,
        locale=args.locale
    )


if __name__ == "__main__":
    main()
