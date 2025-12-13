"""Show linked records for flagged fraud cases."""

import pandas as pd

df = pd.read_csv('data/true_positives.csv')
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
all_flagged = pd.read_csv('data/flagged_records.csv')

# Group by IBAN to show shared relationships
iban_groups = all_flagged.groupby('iban')
for iban, group in iban_groups:
    if len(group) > 1:
        print(f"\nIBAN: {iban}")
        for _, row in group.iterrows():
            fraud_indicator = " [FRAUD]" if row.get('is_fraud', False) else ""
            print(f"  - {row['customer_id']}: {row['surname']}, {row['first_name']}{fraud_indicator}")
