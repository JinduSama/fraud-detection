"""Train production models for real-time fraud scoring.

This script trains the real-time scorer on historical data and saves
the models for production use. Run this monthly or when retraining is needed.

Usage:
    # Train on historical data
    uv run python -m src.train_production --input data/customer_dataset.csv

    # Train with custom output directory
    uv run python -m src.train_production --input data/historical.csv --output models/v2/

    # Train on synthetic data for testing
    uv run python -m src.train_production --generate --num-records 5000
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train production models for real-time fraud scoring.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train on existing data
    uv run python -m src.train_production --input data/customer_dataset.csv
    
    # Generate synthetic data and train
    uv run python -m src.train_production --generate --num-records 5000
    
    # Specify custom output directory
    uv run python -m src.train_production --input data/historical.csv --output models/v2/
        """,
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "-i", "--input",
        type=str,
        help="Path to CSV file with historical customer records.",
    )
    input_group.add_argument(
        "--generate",
        action="store_true",
        help="Generate synthetic data for training (for testing purposes).",
    )
    
    # Generation options
    parser.add_argument(
        "-n", "--num-records",
        type=int,
        default=1000,
        help="Number of records to generate (with --generate). Default: 1000",
    )
    parser.add_argument(
        "-f", "--fraud-ratio",
        type=float,
        default=0.15,
        help="Fraud ratio for synthetic data (with --generate). Default: 0.15",
    )
    parser.add_argument(
        "-s", "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility. Default: 42",
    )
    
    # Output options
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="models/production/",
        help="Directory to save trained models. Default: models/production/",
    )
    
    # Scorer configuration
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.1,
        help="Isolation Forest contamination parameter. Default: 0.1",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Number of Isolation Forest estimators. Default: 100",
    )
    parser.add_argument(
        "--high-threshold",
        type=float,
        default=0.8,
        help="Threshold for HIGH alert level. Default: 0.8",
    )
    parser.add_argument(
        "--medium-threshold",
        type=float,
        default=0.5,
        help="Threshold for MEDIUM alert level. Default: 0.5",
    )
    
    return parser.parse_args()


def load_data(input_path: str) -> pd.DataFrame:
    """Load data from CSV file."""
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"  Loaded {len(df)} records with {len(df.columns)} columns")
    return df


def generate_data(num_records: int, fraud_ratio: float, seed: int) -> pd.DataFrame:
    """Generate synthetic data for training."""
    from src.data.generator import CustomerDataGenerator
    from src.data.fraud_injector import FraudInjector, FraudType
    
    print(f"Generating {num_records} synthetic records with {fraud_ratio:.0%} fraud ratio...")
    
    # Generate legitimate records
    generator = CustomerDataGenerator(locale="de_DE", seed=seed)
    legitimate_records = generator.generate_records(num_records)
    
    # Inject fraud patterns
    injector = FraudInjector(seed=seed)
    fraud_records = injector.inject_fraud_patterns(
        legitimate_records,
        fraud_ratio=fraud_ratio,
        fraud_types=[
            FraudType.NEAR_DUPLICATE,
            FraudType.TYPO_VARIANT,
            FraudType.SHARED_IBAN,
            FraudType.SYNTHETIC_IDENTITY,
        ]
    )
    
    # Combine and convert to DataFrame
    all_records = legitimate_records + fraud_records
    df = generator.to_dataframe(all_records)
    
    # Shuffle
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    print(f"  Generated {len(df)} records ({df['is_fraud'].sum()} fraudulent)")
    return df


def train_scorer(
    df: pd.DataFrame,
    output_dir: str,
    contamination: float,
    n_estimators: int,
    high_threshold: float,
    medium_threshold: float,
    seed: int,
) -> None:
    """Train and save the real-time scorer."""
    from src.scoring import RealTimeScorer
    from src.scoring.realtime import ScorerConfig
    
    output_path = Path(output_dir)
    
    # Configure scorer
    config = ScorerConfig(
        high_threshold=high_threshold,
        medium_threshold=medium_threshold,
        contamination=contamination,
        n_estimators=n_estimators,
        random_state=seed,
    )
    
    # Create and train scorer
    print("\nTraining real-time scorer...")
    scorer = RealTimeScorer(config=config)
    
    start = time.time()
    scorer.train(df, customer_id_col="customer_id")
    training_time = time.time() - start
    
    # Save models
    print(f"\nSaving models to {output_path}...")
    scorer.save(output_path)
    
    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Training time:     {training_time:.1f}s")
    print(f"  Records trained:   {len(df)}")
    print(f"  Index size:        {len(scorer.similarity_index)}")
    print(f"  Model directory:   {output_path.absolute()}")
    print(f"\n  Configuration:")
    print(f"    Contamination:   {contamination}")
    print(f"    N-estimators:    {n_estimators}")
    print(f"    High threshold:  {high_threshold}")
    print(f"    Medium threshold: {medium_threshold}")
    print("=" * 60)


def validate_scorer(df: pd.DataFrame, output_dir: str) -> None:
    """Validate the trained scorer by scoring a few records."""
    from src.scoring import RealTimeScorer
    
    print("\nValidating trained scorer...")
    scorer = RealTimeScorer.load(output_dir)
    
    # Score a sample of records
    sample_size = min(5, len(df))
    sample = df.sample(sample_size, random_state=42)
    
    print(f"\nScoring {sample_size} sample records:")
    print("-" * 60)
    
    for _, row in sample.iterrows():
        result = scorer.score(row)
        customer_id = row.get("customer_id", "unknown")
        is_fraud = row.get("is_fraud", "unknown")
        
        print(f"  {customer_id}:")
        print(f"    Alert Level:    {result.alert_level.value}")
        print(f"    Combined Score: {result.combined_score:.3f}")
        print(f"    Intrinsic:      {result.intrinsic_score:.3f}")
        print(f"    Similarity:     {result.similarity_score:.3f}")
        print(f"    Actual Fraud:   {is_fraud}")
        print(f"    Flags:          {', '.join(result.flags) if result.flags else 'none'}")
        print(f"    Time:           {result.scoring_time_ms:.1f}ms")
        print()


def main() -> int:
    """Main entry point."""
    args = parse_args()
    
    try:
        # Load or generate data
        if args.generate:
            df = generate_data(args.num_records, args.fraud_ratio, args.seed)
        else:
            df = load_data(args.input)
        
        # Validate required columns
        required_cols = ["customer_id", "surname", "first_name", "email", "iban"]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            print(f"ERROR: Missing required columns: {missing}")
            print(f"Required columns: {required_cols}")
            return 1
        
        # Train scorer
        train_scorer(
            df=df,
            output_dir=args.output,
            contamination=args.contamination,
            n_estimators=args.n_estimators,
            high_threshold=args.high_threshold,
            medium_threshold=args.medium_threshold,
            seed=args.seed,
        )
        
        # Validate
        validate_scorer(df, args.output)
        
        print("\nDone! Models are ready for production use.")
        print(f"\nTo use in production:")
        print(f"    from src.scoring import RealTimeScorer")
        print(f"    scorer = RealTimeScorer.load('{args.output}')")
        print(f"    result = scorer.score(new_application)")
        
        return 0
        
    except ImportError as e:
        print(f"ERROR: Missing dependency: {e}")
        print("Install with: uv pip install faiss-cpu")
        return 1
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
