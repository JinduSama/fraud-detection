"""
Generate linked cases report showing related records side-by-side.

This script creates a CSV where linked/related fraud cases are grouped together
so you can easily compare their data and understand the similarities.
"""

import pandas as pd
from pathlib import Path


def generate_linked_cases_report(
    detected_path: str = "data/detected_fraud.csv",
    output_path: str = "data/linked_cases_report.csv"
) -> pd.DataFrame:
    """
    Generate a report showing linked fraud cases grouped together.
    
    Creates groups based on:
    - Shared IBAN
    - Shared address
    - Same cluster_id
    - Explicit linked_records references
    
    Args:
        detected_path: Path to detection results CSV
        output_path: Path for output report CSV
        
    Returns:
        DataFrame with grouped linked cases
    """
    df = pd.read_csv(detected_path)
    
    # Only look at flagged records
    flagged = df[df["detected_fraud"] == True].copy()
    
    if len(flagged) == 0:
        print("No flagged records found.")
        return pd.DataFrame()
    
    # Create a union-find structure to group related records
    parent = {idx: idx for idx in flagged.index}
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    # Group by shared IBAN
    if "iban" in flagged.columns:
        for iban, group in flagged.groupby("iban"):
            indices = group.index.tolist()
            for i in range(1, len(indices)):
                union(indices[0], indices[i])
    
    # Group by shared address (normalized)
    if all(c in flagged.columns for c in ["strasse", "hausnummer", "plz", "stadt"]):
        flagged["address_norm"] = (
            flagged["strasse"].fillna("").astype(str).str.lower().str.strip() + "|" +
            flagged["hausnummer"].fillna("").astype(str).str.lower().str.strip() + "|" +
            flagged["plz"].fillna("").astype(str).str.lower().str.strip() + "|" +
            flagged["stadt"].fillna("").astype(str).str.lower().str.strip()
        )
    elif "address" in flagged.columns:
        flagged["address_norm"] = flagged["address"].fillna("").astype(str).str.lower().str.strip()
    else:
        flagged["address_norm"] = ""

    if "address_norm" in flagged.columns:
        for addr, group in flagged.groupby("address_norm"):
            indices = group.index.tolist()
            for i in range(1, len(indices)):
                union(indices[0], indices[i])
        flagged = flagged.drop(columns=["address_norm"])
    
    # Group by cluster_id if available
    if "cluster_id" in flagged.columns:
        for cluster_id, group in flagged.groupby("cluster_id"):
            if cluster_id >= 0:  # -1 means no cluster
                indices = group.index.tolist()
                for i in range(1, len(indices)):
                    union(indices[0], indices[i])
    
    # Assign link group IDs
    group_map = {}
    group_counter = 1
    
    for idx in flagged.index:
        root = find(idx)
        if root not in group_map:
            group_map[root] = group_counter
            group_counter += 1
        flagged.at[idx, "link_group"] = group_map[root]
    
    flagged["link_group"] = flagged["link_group"].astype(int)
    
    # Sort by link_group so related records appear together
    flagged = flagged.sort_values(["link_group", "customer_id"])
    
    # Add group size
    group_sizes = flagged.groupby("link_group").size()
    flagged["group_size"] = flagged["link_group"].map(group_sizes)
    
    # Filter to only show groups with more than 1 member (actual links)
    linked_only = flagged[flagged["group_size"] > 1].copy()
    
    # Add comparison columns to highlight similarities
    comparison_cols = []
    
    # For each group, identify what they share
    def get_shared_attributes(group_df):
        shared = []
        
        # Check IBAN
        if "iban" in group_df.columns:
            if group_df["iban"].nunique() < len(group_df):
                shared.append("IBAN")
        
        # Check address
        if "address" in group_df.columns:
            addrs = group_df["address"].str.lower().str.strip()
            if addrs.nunique() < len(group_df):
                shared.append("ADDRESS")
        
        # Check email domain
        if "email" in group_df.columns:
            domains = group_df["email"].str.extract(r"@(.+)$")[0]
            if domains.nunique() < len(group_df):
                shared.append("EMAIL_DOMAIN")
        
        # Check surname similarity
        if "surname" in group_df.columns:
            surnames = group_df["surname"].str.lower()
            if surnames.nunique() < len(group_df):
                shared.append("SURNAME")
        
        # Check DOB
        if "date_of_birth" in group_df.columns:
            if group_df["date_of_birth"].nunique() < len(group_df):
                shared.append("DOB")
        
        return ", ".join(shared) if shared else "SIMILARITY_PATTERN"
    
    # Add shared_attributes column per group
    for link_group in linked_only["link_group"].unique():
        mask = linked_only["link_group"] == link_group
        group_df = linked_only[mask]
        shared = get_shared_attributes(group_df)
        linked_only.loc[mask, "shared_attributes"] = shared
    
    # Reorder columns for better readability
    priority_cols = [
        "link_group", "group_size", "shared_attributes",
        "customer_id", "surname", "first_name", 
        "address", "iban", "email", "date_of_birth",
        "is_fraud", "fraud_type", "detected_fraud", "fraud_score", "detection_reason"
    ]
    
    # Keep only columns that exist
    final_cols = [c for c in priority_cols if c in linked_only.columns]
    # Add any remaining columns
    remaining = [c for c in linked_only.columns if c not in final_cols]
    final_cols.extend(remaining)
    
    linked_only = linked_only[final_cols]
    
    # Save to CSV
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    linked_only.to_csv(output_file, index=False)
    
    # Print summary
    print("=" * 70)
    print("LINKED CASES REPORT")
    print("=" * 70)
    print(f"\nTotal flagged records: {len(flagged)}")
    print(f"Records with links: {len(linked_only)}")
    print(f"Number of link groups: {linked_only['link_group'].nunique()}")
    print(f"\nOutput saved to: {output_file.absolute()}")
    
    # Print group summary
    print("\n" + "-" * 70)
    print("LINK GROUPS SUMMARY")
    print("-" * 70)
    
    for link_group in sorted(linked_only["link_group"].unique()):
        group = linked_only[linked_only["link_group"] == link_group]
        shared = group["shared_attributes"].iloc[0]
        fraud_count = group["is_fraud"].sum()
        
        print(f"\nGroup {link_group} ({len(group)} records, {fraud_count} actual frauds)")
        print(f"  Shared: {shared}")
        print(f"  Members:")
        
        for _, row in group.iterrows():
            fraud_marker = " [FRAUD]" if row.get("is_fraud", False) else ""
            print(f"    - {row['customer_id']}: {row['surname']}, {row['first_name']}{fraud_marker}")
            
            # Show key fields that might be shared
            if "IBAN" in shared:
                print(f"      IBAN: {row['iban']}")
            if "ADDRESS" in shared:
                addr = str(row['address'])[:50] + "..." if len(str(row['address'])) > 50 else row['address']
                print(f"      Address: {addr}")
    
    print("\n" + "=" * 70)
    
    return linked_only


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate linked cases report")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--input", type=str, help="Override input file path")
    parser.add_argument("--output", type=str, help="Override output file path")
    args = parser.parse_args()
    
    cfg = load_config(args.config) if args.config else None
    generate_linked_cases_report(
        config=cfg,
        detected_path=args.input,
        output_path=args.output,
    )
