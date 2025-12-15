"""Show linked records for flagged fraud cases."""

import argparse
from pathlib import Path

import pandas as pd

from src.config import load_config


def show_links(config_path: str = None) -> None:
    """Show linked records for flagged fraud cases.
    
    Args:
        config_path: Optional path to config file.
    """
    config = load_config(config_path)
    data_dir = Path(config.paths.data_dir)
    
    true_positives_path = data_dir / "true_positives.csv"
    flagged_path = data_dir / "flagged_records.csv"
    
    if not true_positives_path.exists():
        print(f"File not found: {true_positives_path}")
        print("Run evaluation first to generate true_positives.csv")
        return
    
    df = pd.read_csv(true_positives_path)
    print('=== TRUE POSITIVES - FRAUD RELATIONSHIPS ===')
    print()
    for _, row in df.iterrows():
        print(f"Customer: {row['customer_id']}")
        print(f"  Name: {row['surname']}, {row['first_name']}")
        print(f"  Fraud Type: {row['fraud_type']}")
        print(f"  Score: {row['fraud_score']:.2f}")
        print(f"  Reason: {row['detection_reason']}")
        linked = str(row.get('linked_records', ''))
        if linked and linked != 'nan':
            # Show first 3 linked
            linked_list = [l.strip() for l in linked.split(';')][:3]
            print(f"  Linked to: {', '.join(linked_list)}")
        print()

    print()
    print('=== SHARED IBAN GROUPS ===')
    
    if not flagged_path.exists():
        print(f"File not found: {flagged_path}")
        return
    
    all_flagged = pd.read_csv(flagged_path)

    # Group by IBAN to show shared relationships
    iban_groups = all_flagged.groupby('iban')
    for iban, group in iban_groups:
        if len(group) > 1:
            print(f"\nIBAN: {iban}")
            for _, row in group.iterrows():
                fraud_indicator = " [FRAUD]" if row.get('is_fraud', False) else ""
                print(f"  - {row['customer_id']}: {row['surname']}, {row['first_name']}{fraud_indicator}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Show linked fraud records")
    parser.add_argument("--config", type=str, help="Path to config file")
    args = parser.parse_args()
    
    show_links(args.config)
