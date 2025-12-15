"""Dataset Generation Script.

Main script that combines legitimate and fraudulent data generation,
labels the ground truth, and saves to CSV.
"""

import argparse
from pathlib import Path

import pandas as pd

from src.data.generator import CustomerDataGenerator
from src.data.fraud_injector import FraudInjector, FraudType


def generate_dataset(
    num_legitimate: int = 1000,
    fraud_ratio: float = 0.15,
    seed: int = 42,
    output_path: str = "data/customer_dataset.csv",
    locale: str = "de_DE"
) -> pd.DataFrame:
    """
    Generate a complete dataset with legitimate and fraudulent records.
    
    Args:
        num_legitimate: Number of legitimate customer records to generate.
        fraud_ratio: Ratio of fraudulent records (0.15 = 15% of legitimate count).
        seed: Random seed for reproducibility.
        output_path: Path to save the output CSV file.
        locale: Faker locale for region-specific data.
        
    Returns:
        DataFrame containing all records with ground truth labels.
    """
    print(f"=" * 60)
    print("FRAUD DETECTION SYNTHETIC DATASET GENERATOR")
    print(f"=" * 60)
    
    # Initialize generators
    print(f"\n[1/4] Initializing generators (seed={seed}, locale={locale})...")
    customer_generator = CustomerDataGenerator(seed=seed, locale=locale)
    fraud_injector = FraudInjector(seed=seed)
    
    # Generate legitimate records
    print(f"\n[2/4] Generating {num_legitimate} legitimate customer records...")
    legitimate_records = customer_generator.generate_records(num_legitimate)
    print(f"       Generated {len(legitimate_records)} legitimate records")
    
    # Inject fraud patterns
    print(f"\n[3/4] Injecting fraud patterns (ratio={fraud_ratio:.0%})...")
    fraud_records = fraud_injector.inject_fraud_patterns(
        legitimate_records,
        fraud_ratio=fraud_ratio,
        fraud_types=[
            FraudType.NEAR_DUPLICATE,
            FraudType.TYPO_VARIANT,
            FraudType.SHARED_IBAN,
            FraudType.SYNTHETIC_IDENTITY,
        ]
    )
    print(f"       Generated {len(fraud_records)} fraudulent records")
    
    # Count fraud types
    fraud_type_counts = {}
    for record in fraud_records:
        ft = record.fraud_type
        fraud_type_counts[ft] = fraud_type_counts.get(ft, 0) + 1
    
    print("\n      Fraud breakdown:")
    for ft, count in sorted(fraud_type_counts.items()):
        print(f"        - {ft}: {count}")
    
    # Combine all records
    all_records = legitimate_records + fraud_records
    
    # Convert to DataFrame
    df = customer_generator.to_dataframe(all_records)
    
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # Save to CSV
    print(f"\n[4/4] Saving dataset to {output_path}...")
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"       Saved {len(df)} total records")
    
    # Summary statistics
    print(f"\n{'=' * 60}")
    print("DATASET SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total records:      {len(df)}")
    print(f"Legitimate records: {len(legitimate_records)} ({len(legitimate_records)/len(df):.1%})")
    print(f"Fraudulent records: {len(fraud_records)} ({len(fraud_records)/len(df):.1%})")
    print(f"Output file:        {output_file.absolute()}")
    print(f"{'=' * 60}\n")
    
    return df


def main():
    """Main entry point for dataset generation."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic customer dataset with fraud patterns"
    )
    parser.add_argument(
        "-n", "--num-records",
        type=int,
        default=1000,
        help="Number of legitimate records to generate (default: 1000)"
    )
    parser.add_argument(
        "-f", "--fraud-ratio",
        type=float,
        default=0.15,
        help="Ratio of fraudulent records (default: 0.15)"
    )
    parser.add_argument(
        "-s", "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="data/customer_dataset.csv",
        help="Output CSV file path (default: data/customer_dataset.csv)"
    )
    parser.add_argument(
        "-l", "--locale",
        type=str,
        default="de_DE",
        help="Faker locale for data generation (default: de_DE)"
    )
    
    args = parser.parse_args()
    
    generate_dataset(
        num_legitimate=args.num_records,
        fraud_ratio=args.fraud_ratio,
        seed=args.seed,
        output_path=args.output,
        locale=args.locale
    )


if __name__ == "__main__":
    main()
