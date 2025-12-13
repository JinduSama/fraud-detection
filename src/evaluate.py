"""
Full Pipeline Evaluation Script.

Runs the complete pipeline: Generate -> Detect -> Evaluate
and produces a comprehensive performance report with visualization.

TASK-009: Create reporting script for full pipeline execution.
Enhanced with confusion matrix, PR curves, and stratified evaluation.
"""

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.generate_dataset import generate_dataset
from src.detect_fraud import detect_fraud
from src.evaluation.metrics import FraudMetrics
from src.config import load_config
from src.utils.logging import get_logger


def calculate_stratified_metrics(
    df: pd.DataFrame,
    fraud_type_col: str = "fraud_type"
) -> dict:
    """
    Calculate metrics stratified by fraud type.
    
    Args:
        df: DataFrame with detection results.
        fraud_type_col: Column containing fraud type labels.
        
    Returns:
        Dictionary with metrics per fraud type.
    """
    if fraud_type_col not in df.columns:
        return {}
    
    stratified_results = {}
    metrics = FraudMetrics()
    
    # Get unique fraud types
    fraud_types = df[df["is_fraud"] == True][fraud_type_col].unique()
    
    for fraud_type in fraud_types:
        if pd.isna(fraud_type):
            continue
        
        # Create mask for this fraud type
        mask = df[fraud_type_col] == fraud_type
        subset = df[mask]
        
        if len(subset) == 0:
            continue
        
        # Calculate metrics
        ground_truth = subset["is_fraud"].values
        predictions = subset["detected_fraud"].values
        
        tp = ((ground_truth == True) & (predictions == True)).sum()
        fn = ((ground_truth == True) & (predictions == False)).sum()
        
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        stratified_results[str(fraud_type)] = {
            "count": int(len(subset)),
            "detected": int(tp),
            "missed": int(fn),
            "recall": float(recall)
        }
    
    return stratified_results


def generate_confusion_matrix(
    df: pd.DataFrame
) -> tuple[np.ndarray, dict]:
    """
    Generate confusion matrix from results.
    
    Args:
        df: DataFrame with detection results.
        
    Returns:
        Tuple of (matrix, labels).
    """
    ground_truth = df["is_fraud"].fillna(False).values
    predictions = df["detected_fraud"].fillna(False).values
    
    tp = ((ground_truth == True) & (predictions == True)).sum()
    fp = ((ground_truth == False) & (predictions == True)).sum()
    fn = ((ground_truth == True) & (predictions == False)).sum()
    tn = ((ground_truth == False) & (predictions == False)).sum()
    
    matrix = np.array([[tn, fp], [fn, tp]])
    
    return matrix, {"TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn)}


def format_confusion_matrix_report(matrix: np.ndarray, labels: dict) -> str:
    """Format confusion matrix as a string report."""
    lines = [
        "",
        "CONFUSION MATRIX",
        "-" * 40,
        "                   Predicted",
        "                 Neg      Pos",
        f"Actual Neg    {matrix[0, 0]:6d}   {matrix[0, 1]:6d}",
        f"Actual Pos    {matrix[1, 0]:6d}   {matrix[1, 1]:6d}",
        "-" * 40,
        f"True Positives:  {labels['TP']}",
        f"False Positives: {labels['FP']}",
        f"True Negatives:  {labels['TN']}",
        f"False Negatives: {labels['FN']}",
    ]
    return "\n".join(lines)


def calculate_precision_recall_curve(
    df: pd.DataFrame,
    num_thresholds: int = 20
) -> dict:
    """
    Calculate precision-recall curve data.
    
    Args:
        df: DataFrame with detection results.
        num_thresholds: Number of threshold points.
        
    Returns:
        Dictionary with curve data.
    """
    if "fraud_score" not in df.columns:
        return {}
    
    ground_truth = df["is_fraud"].fillna(False).values
    scores = df["fraud_score"].fillna(0).values
    
    thresholds = np.linspace(0, 1, num_thresholds)
    precisions = []
    recalls = []
    
    for thresh in thresholds:
        predictions = scores >= thresh
        
        tp = ((ground_truth == True) & (predictions == True)).sum()
        fp = ((ground_truth == False) & (predictions == True)).sum()
        fn = ((ground_truth == True) & (predictions == False)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        precisions.append(precision)
        recalls.append(recall)
    
    return {
        "thresholds": thresholds.tolist(),
        "precisions": precisions,
        "recalls": recalls
    }


def format_stratified_report(stratified_results: dict) -> str:
    """Format stratified metrics as a string report."""
    if not stratified_results:
        return "\nNo fraud type information available for stratified analysis."
    
    lines = [
        "",
        "STRATIFIED EVALUATION BY FRAUD TYPE",
        "-" * 60,
        f"{'Fraud Type':<25} {'Count':>8} {'Detected':>10} {'Missed':>8} {'Recall':>10}",
        "-" * 60,
    ]
    
    for fraud_type, metrics in stratified_results.items():
        lines.append(
            f"{fraud_type:<25} {metrics['count']:>8} {metrics['detected']:>10} "
            f"{metrics['missed']:>8} {metrics['recall']:>9.1%}"
        )
    
    return "\n".join(lines)


def format_pr_curve_report(pr_data: dict) -> str:
    """Format precision-recall curve as ASCII."""
    if not pr_data:
        return "\nNo score data available for PR curve."
    
    lines = [
        "",
        "PRECISION-RECALL CURVE (Sampled)",
        "-" * 50,
        f"{'Threshold':>10} {'Precision':>12} {'Recall':>12}",
        "-" * 50,
    ]
    
    for i in range(0, len(pr_data["thresholds"]), 4):  # Sample every 4th point
        lines.append(
            f"{pr_data['thresholds'][i]:>10.2f} "
            f"{pr_data['precisions'][i]:>11.1%} "
            f"{pr_data['recalls'][i]:>11.1%}"
        )
    
    return "\n".join(lines)


def save_inspection_files(df: pd.DataFrame, output_dir: Path) -> dict:
    """
    Save detailed inspection files for manual review.
    
    Creates separate CSV files for:
    - All flagged records
    - True positives (correctly caught frauds)
    - False positives (incorrectly flagged legitimate records)
    - False negatives (missed frauds)
    - All actual fraud records with detection status
    
    Args:
        df: DataFrame with detection results.
        output_dir: Directory for output files.
        
    Returns:
        Dictionary with paths to saved files.
    """
    saved_files = {}
    
    # Ensure columns exist
    has_ground_truth = "is_fraud" in df.columns
    has_predictions = "detected_fraud" in df.columns
    
    if not has_predictions:
        return saved_files
    
    # 1. All flagged records (for investigation)
    flagged = df[df["detected_fraud"] == True].copy()
    if len(flagged) > 0:
        flagged_path = output_dir / "flagged_records.csv"
        flagged.to_csv(flagged_path, index=False)
        saved_files["flagged_records"] = str(flagged_path)
    
    if has_ground_truth:
        ground_truth = df["is_fraud"].fillna(False)
        predictions = df["detected_fraud"].fillna(False)
        
        # 2. True positives (correctly caught frauds)
        tp_mask = (ground_truth == True) & (predictions == True)
        true_positives = df[tp_mask].copy()
        if len(true_positives) > 0:
            tp_path = output_dir / "true_positives.csv"
            true_positives.to_csv(tp_path, index=False)
            saved_files["true_positives"] = str(tp_path)
        
        # 3. False positives (incorrectly flagged legitimate records)
        fp_mask = (ground_truth == False) & (predictions == True)
        false_positives = df[fp_mask].copy()
        if len(false_positives) > 0:
            fp_path = output_dir / "false_positives.csv"
            false_positives.to_csv(fp_path, index=False)
            saved_files["false_positives"] = str(fp_path)
        
        # 4. False negatives (missed frauds)
        fn_mask = (ground_truth == True) & (predictions == False)
        false_negatives = df[fn_mask].copy()
        if len(false_negatives) > 0:
            fn_path = output_dir / "false_negatives.csv"
            false_negatives.to_csv(fn_path, index=False)
            saved_files["false_negatives"] = str(fn_path)
        
        # 5. All actual frauds with detection status
        actual_frauds = df[ground_truth == True].copy()
        if len(actual_frauds) > 0:
            # Add a column showing if caught or missed
            actual_frauds["detection_status"] = actual_frauds["detected_fraud"].apply(
                lambda x: "CAUGHT" if x else "MISSED"
            )
            frauds_path = output_dir / "actual_frauds_summary.csv"
            actual_frauds.to_csv(frauds_path, index=False)
            saved_files["actual_frauds"] = str(frauds_path)
    
    return saved_files


def run_full_pipeline(
    num_records: int = 500,
    fraud_ratio: float = 0.15,
    seed: int = 42,
    eps: float = 0.35,
    min_samples: int = 2,
    use_blocking: bool = True,
    output_dir: str = "data",
    locale: str = "de_DE",
    config_path: Optional[str] = None,
    detectors: Optional[list[str]] = None,
    fusion_strategy: str = "weighted_avg",
    threshold: float = 0.5,
    use_ensemble: bool = True,
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
        config_path: Path to configuration file.
        detectors: List of detectors to use.
        fusion_strategy: Ensemble fusion strategy.
        threshold: Decision threshold.
        use_ensemble: Use ensemble detector.
        
    Returns:
        Dictionary with all results and metrics.
    """
    log = get_logger(format="text")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("   FRAUD DETECTION FULL PIPELINE EXECUTION")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Records:         {num_records}")
    print(f"  Fraud ratio:     {fraud_ratio:.0%}")
    print(f"  Seed:            {seed}")
    print(f"  DBSCAN eps:      {eps}")
    print(f"  Min samples:     {min_samples}")
    print(f"  Blocking:        {use_blocking}")
    print(f"  Locale:          {locale}")
    print(f"  Ensemble mode:   {use_ensemble}")
    print(f"  Fusion strategy: {fusion_strategy}")
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
        use_blocking=use_blocking,
        config_path=config_path,
        detectors=detectors,
        fusion_strategy=fusion_strategy,
        threshold=threshold,
        use_ensemble=use_ensemble,
        logger=log,
    )
    
    # Stage 3: Evaluation
    print("\n" + "-" * 70)
    print("STAGE 3: EVALUATION")
    print("-" * 70)
    
    metrics = FraudMetrics()
    report = metrics.get_detailed_report(df_detected)
    print(report)
    
    # Additional analysis: Confusion Matrix
    conf_matrix, conf_labels = generate_confusion_matrix(df_detected)
    conf_report = format_confusion_matrix_report(conf_matrix, conf_labels)
    print(conf_report)
    
    # Stratified evaluation
    stratified = calculate_stratified_metrics(df_detected)
    strat_report = format_stratified_report(stratified)
    print(strat_report)
    
    # Precision-Recall curve
    pr_data = calculate_precision_recall_curve(df_detected)
    pr_report = format_pr_curve_report(pr_data)
    print(pr_report)
    
    # Save inspection files
    print("\n" + "-" * 70)
    print("STAGE 4: SAVING INSPECTION FILES")
    print("-" * 70)
    
    inspection_files = save_inspection_files(df_detected, output_path)
    
    print("\nInspection files saved:")
    for file_type, file_path in inspection_files.items():
        count = len(pd.read_csv(file_path))
        print(f"  - {file_type}: {file_path} ({count} records)")
    
    # Combine all reports
    full_report = "\n".join([
        "=" * 70,
        "FRAUD DETECTION EVALUATION REPORT",
        "=" * 70,
        f"\nDataset: {dataset_path}",
        f"Records: {len(df_detected)}",
        f"Fraud ratio: {fraud_ratio:.0%}",
        "",
        report,
        conf_report,
        strat_report,
        pr_report,
    ])
    
    # Save evaluation report
    report_path = output_path / "evaluation_report.txt"
    with open(report_path, "w") as f:
        f.write(full_report)
    print(f"\n\nReport saved to: {report_path.absolute()}")
    
    # Log final metrics
    if metrics._results:
        log.log_evaluation_result(
            precision=metrics._results.precision,
            recall=metrics._results.recall,
            f1_score=metrics._results.f1_score
        )
    
    # Return all results
    results = {
        "dataset": df_generated,
        "detection_results": df_detected,
        "metrics": metrics._results,
        "confusion_matrix": conf_labels,
        "stratified_metrics": stratified,
        "pr_curve": pr_data,
        "report": full_report,
        "files": {
            "dataset": dataset_path,
            "detection": detection_path,
            "report": str(report_path),
            **inspection_files,
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
        help="Detectors to use"
    )
    parser.add_argument(
        "--fusion-strategy",
        type=str,
        default="weighted_avg",
        choices=["max", "weighted_avg", "voting", "stacking"],
        help="Ensemble fusion strategy"
    )
    parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=0.7,
        help="Decision threshold"
    )
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Use legacy single-detector mode"
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
        locale=args.locale,
        config_path=args.config,
        detectors=args.detectors,
        fusion_strategy=args.fusion_strategy,
        threshold=args.threshold,
        use_ensemble=not args.legacy,
    )


if __name__ == "__main__":
    main()
