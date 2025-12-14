"""Parameter + seed sweep for fraud detection pipeline.

Runs multiple synthetic dataset generations (different seeds) and evaluates
multiple detector/ensemble parameter settings, producing a summary CSV.

Usage example:
  uv run python -m src.sweep \
    --seeds 1 2 3 4 5 \
    --num-records 500 --fraud-ratio 0.02 \
    --detector-sets dbscan,isolation_forest dbscan,isolation_forest,graph \
    --eps 0.3 0.35 0.4 \
    --thresholds 0.6 0.7 0.8 \
    --fusion-strategies weighted_avg voting \
    --output data/sweep_results.csv

Notes:
- This script keeps data generation in-memory (no CSV I/O per run).
- It uses the same detectors/ensemble implementation as the main pipeline.
"""

from __future__ import annotations

import argparse
import itertools
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import load_config
from src.data.generator import CustomerDataGenerator
from src.data.fraud_injector import FraudInjector, FraudType
from src.detect_fraud import create_ensemble_from_config
from src.evaluation.metrics import FraudMetrics


def _parse_detector_set(spec: str) -> list[str]:
    spec = spec.strip()
    if not spec:
        return []
    return [s.strip() for s in spec.split(",") if s.strip()]


def _iter_seeds(args: argparse.Namespace) -> list[int]:
    if args.seeds:
        return list(args.seeds)
    if args.seed_start is not None and args.seed_end is not None:
        if args.seed_end < args.seed_start:
            raise ValueError("--seed-end must be >= --seed-start")
        return list(range(args.seed_start, args.seed_end + 1))
    return [args.seed]


def _parse_contamination(value: str) -> str | float:
    v = value.strip()
    if v.lower() == "auto":
        return "auto"
    return float(v)


def _generate_dataset_df(
    *,
    num_legitimate: int,
    fraud_ratio: float,
    seed: int,
    locale: str,
    fraud_types: Optional[list[FraudType]] = None,
) -> pd.DataFrame:
    """Generate a full dataset as a DataFrame without writing to disk."""
    customer_generator = CustomerDataGenerator(seed=seed, locale=locale)
    fraud_injector = FraudInjector(seed=seed)

    legitimate_records = customer_generator.generate_records(num_legitimate)
    fraud_records = fraud_injector.inject_fraud_patterns(
        legitimate_records,
        fraud_ratio=fraud_ratio,
        fraud_types=fraud_types
        or [
            FraudType.NEAR_DUPLICATE,
            FraudType.TYPO_VARIANT,
            FraudType.SHARED_IBAN,
            FraudType.SYNTHETIC_IDENTITY,
        ],
    )

    all_records = legitimate_records + fraud_records
    df = customer_generator.to_dataframe(all_records)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


def _evaluate_run(
    *,
    seed: int,
    df: pd.DataFrame,
    config_path: Optional[str],
    detectors: list[str],
    eps: float,
    min_samples: int,
    distance_metric: str,
    fusion_strategy: str,
    threshold: float,
    if_contamination: Optional[str | float] = None,
    graph_similarity_threshold: Optional[float] = None,
) -> dict:
    config = load_config(config_path)

    # Enable/disable detectors
    config.detectors.dbscan.enabled = "dbscan" in detectors
    config.detectors.isolation_forest.enabled = "isolation_forest" in detectors
    config.detectors.lof.enabled = "lof" in detectors
    config.detectors.graph.enabled = "graph" in detectors

    # Apply sweep params
    config.ensemble.strategy = fusion_strategy
    config.ensemble.threshold = threshold

    config.detectors.dbscan.eps = eps
    config.detectors.dbscan.min_samples = min_samples
    config.detectors.dbscan.distance_metric = distance_metric

    if if_contamination is not None:
        config.detectors.isolation_forest.contamination = if_contamination  # type: ignore[assignment]

    if graph_similarity_threshold is not None:
        config.detectors.graph.similarity_threshold = graph_similarity_threshold

    ensemble = create_ensemble_from_config(config)

    ensemble.fit(df)
    pred = ensemble.predict(df)

    df_eval = df.copy()
    df_eval["detected_fraud"] = pred["is_fraud"].astype(bool)
    df_eval["fraud_score"] = pred["score"].astype(float)

    metrics = FraudMetrics().evaluate(df_eval)

    return {
        "seed": seed,
        "detectors": ",".join(detectors),
        "eps": eps,
        "min_samples": min_samples,
        "distance_metric": distance_metric,
        "fusion_strategy": fusion_strategy,
        "threshold": threshold,
        "if_contamination": if_contamination if if_contamination is not None else config.detectors.isolation_forest.contamination,
        "graph_similarity_threshold": graph_similarity_threshold
        if graph_similarity_threshold is not None
        else config.detectors.graph.similarity_threshold,
        "records": int(len(df_eval)),
        "actual_frauds": int(df_eval["is_fraud"].fillna(False).astype(bool).sum()),
        "flagged": int(df_eval["detected_fraud"].sum()),
        **asdict(metrics),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep seeds + parameters and export evaluation summary CSV."
    )

    # Data generation
    parser.add_argument("--num-records", type=int, default=500, help="Legitimate records per run")
    parser.add_argument("--fraud-ratio", type=float, default=0.02, help="Fraud ratio vs legitimate")
    parser.add_argument("--locale", type=str, default="de_DE", help="Faker locale")

    # Seeds
    parser.add_argument("--seed", type=int, default=42, help="Single seed (if no --seeds/--seed-start)")
    parser.add_argument("--seeds", type=int, nargs="+", help="Explicit list of seeds")
    parser.add_argument("--seed-start", type=int, default=None, help="Seed range start (inclusive)")
    parser.add_argument("--seed-end", type=int, default=None, help="Seed range end (inclusive)")

    # Detector selection
    parser.add_argument(
        "--detectors",
        type=str,
        nargs="+",
        default=None,
        choices=["dbscan", "isolation_forest", "lof", "graph"],
        help="Detector list applied to all runs (ignored if --detector-sets provided)",
    )
    parser.add_argument(
        "--detector-sets",
        type=str,
        nargs="+",
        default=None,
        help="One or more comma-separated detector sets, e.g. 'dbscan,isolation_forest'",
    )

    # Sweep parameters
    parser.add_argument("--eps", type=float, nargs="+", default=[0.35])
    parser.add_argument("--min-samples", type=int, nargs="+", default=[2])
    parser.add_argument(
        "--distance-metrics",
        type=str,
        nargs="+",
        default=["jaro_winkler"],
        choices=["jaro_winkler", "levenshtein", "damerau"],
    )
    parser.add_argument(
        "--fusion-strategies",
        type=str,
        nargs="+",
        default=["weighted_avg"],
        choices=["max", "weighted_avg", "voting", "stacking"],
    )
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.7])

    # Optional extra knobs
    parser.add_argument(
        "--if-contaminations",
        type=str,
        nargs="+",
        default=None,
        help="IsolationForest contamination values, e.g. auto 0.01 0.02",
    )
    parser.add_argument(
        "--graph-thresholds",
        type=float,
        nargs="+",
        default=None,
        help="Graph similarity thresholds, e.g. 0.6 0.7 0.8",
    )

    # Other
    parser.add_argument("--config", type=str, default=None, help="Path to base YAML config")
    parser.add_argument(
        "--optimize",
        type=str,
        default="f1",
        choices=["f1", "precision", "recall"],
        help="Sort results by this metric (desc)",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Optional cap for total runs (useful for quick iteration)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path (default: data/sweeps/sweep_<timestamp>.csv)",
    )

    args = parser.parse_args()

    seeds = _iter_seeds(args)

    if args.detector_sets:
        detector_sets = [_parse_detector_set(s) for s in args.detector_sets]
    else:
        detector_sets = [args.detectors or ["dbscan", "isolation_forest"]]

    if_contaminations: list[Optional[str | float]]
    if args.if_contaminations is None:
        if_contaminations = [None]
    else:
        if_contaminations = [_parse_contamination(v) for v in args.if_contaminations]

    graph_thresholds: list[Optional[float]]
    if args.graph_thresholds is None:
        graph_thresholds = [None]
    else:
        graph_thresholds = list(args.graph_thresholds)

    if args.output:
        out_path = Path(args.output)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = Path("data") / "sweeps" / f"sweep_{ts}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    grid = itertools.product(
        seeds,
        detector_sets,
        args.eps,
        args.min_samples,
        args.distance_metrics,
        args.fusion_strategies,
        args.thresholds,
        if_contaminations,
        graph_thresholds,
    )

    results: list[dict] = []
    total = 0

    for (
        seed,
        detectors,
        eps,
        min_samples,
        distance_metric,
        fusion_strategy,
        threshold,
        if_contamination,
        graph_threshold,
    ) in grid:
        total += 1
        if args.max_runs is not None and len(results) >= args.max_runs:
            break

        df = _generate_dataset_df(
            num_legitimate=args.num_records,
            fraud_ratio=args.fraud_ratio,
            seed=seed,
            locale=args.locale,
        )

        row = _evaluate_run(
            seed=seed,
            df=df,
            config_path=args.config,
            detectors=detectors,
            eps=eps,
            min_samples=min_samples,
            distance_metric=distance_metric,
            fusion_strategy=fusion_strategy,
            threshold=threshold,
            if_contamination=if_contamination,
            graph_similarity_threshold=graph_threshold,
        )
        results.append(row)

        if len(results) % 5 == 0:
            print(f"Completed {len(results)} runs...")

    if not results:
        raise SystemExit("No runs executed (check your sweep arguments)")

    df_out = pd.DataFrame(results)
    sort_key = {
        "f1": "f1_score",
        "precision": "precision",
        "recall": "recall",
    }[args.optimize]

    df_out = df_out.sort_values(
        by=[sort_key, "recall", "precision"],
        ascending=[False, False, False],
    )

    df_out.to_csv(out_path, index=False)

    print("\nTop 10 configurations (sorted):")
    cols = [
        "seed",
        "detectors",
        "eps",
        "min_samples",
        "distance_metric",
        "fusion_strategy",
        "threshold",
        "precision",
        "recall",
        "f1_score",
        "false_positives",
        "false_negatives",
        "flagged",
        "actual_frauds",
    ]
    print(df_out[cols].head(10).to_string(index=False))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
