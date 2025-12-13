"""
Generate a focused linked pairs report showing exactly which records share what.

This creates multiple views:
1. Shared IBAN pairs
2. Shared Address pairs  
3. Similar Name pairs (for typo variants)
"""

import pandas as pd
from pathlib import Path
import jellyfish


def generate_focused_links_report(
    detected_path: str = "data/detected_fraud.csv",
    output_dir: str = "data"
) -> dict:
    """
    Generate focused reports showing exactly which records are linked by what.
    """
    df = pd.read_csv(detected_path)
    output_path = Path(output_dir)
    
    reports = {}
    
    # =========================================================================
    # 1. SHARED IBAN REPORT
    # =========================================================================
    print("=" * 70)
    print("1. SHARED IBAN PAIRS")
    print("=" * 70)
    
    iban_groups = []
    if "iban" in df.columns:
        for iban, group in df.groupby("iban"):
            if len(group) > 1:
                for _, row in group.iterrows():
                    iban_groups.append({
                        "shared_iban": iban,
                        "group_size": len(group),
                        "customer_id": row["customer_id"],
                        "surname": row.get("surname", ""),
                        "first_name": row.get("first_name", ""),
                        "address": row.get("address", ""),
                        "email": row.get("email", ""),
                        "date_of_birth": row.get("date_of_birth", ""),
                        "is_fraud": row.get("is_fraud", False),
                        "fraud_type": row.get("fraud_type", ""),
                        "detected_fraud": row.get("detected_fraud", False),
                        "fraud_score": row.get("fraud_score", 0),
                    })
    
    if iban_groups:
        iban_df = pd.DataFrame(iban_groups)
        iban_df = iban_df.sort_values(["shared_iban", "customer_id"])
        iban_file = output_path / "linked_by_iban.csv"
        iban_df.to_csv(iban_file, index=False)
        reports["iban"] = iban_file
        
        print(f"\nFound {iban_df['shared_iban'].nunique()} shared IBAN groups")
        print(f"Saved to: {iban_file}")
        
        # Print details
        for iban in iban_df["shared_iban"].unique():
            group = iban_df[iban_df["shared_iban"] == iban]
            fraud_in_group = group["is_fraud"].any()
            fraud_marker = " ⚠️ FRAUD DETECTED" if fraud_in_group else ""
            print(f"\n  IBAN: {iban}{fraud_marker}")
            for _, row in group.iterrows():
                fraud_tag = " [FRAUD]" if row["is_fraud"] else ""
                print(f"    → {row['customer_id']}: {row['surname']}, {row['first_name']}{fraud_tag}")
    
    # =========================================================================
    # 2. SHARED ADDRESS REPORT
    # =========================================================================
    print("\n" + "=" * 70)
    print("2. SHARED ADDRESS PAIRS")
    print("=" * 70)
    
    address_groups = []
    if "address" in df.columns:
        # Normalize addresses
        df["address_norm"] = df["address"].str.lower().str.strip()
        
        for addr, group in df.groupby("address_norm"):
            if len(group) > 1 and pd.notna(addr) and addr != "":
                for _, row in group.iterrows():
                    address_groups.append({
                        "shared_address": row["address"],  # Original address
                        "group_size": len(group),
                        "customer_id": row["customer_id"],
                        "surname": row.get("surname", ""),
                        "first_name": row.get("first_name", ""),
                        "iban": row.get("iban", ""),
                        "email": row.get("email", ""),
                        "date_of_birth": row.get("date_of_birth", ""),
                        "is_fraud": row.get("is_fraud", False),
                        "fraud_type": row.get("fraud_type", ""),
                        "detected_fraud": row.get("detected_fraud", False),
                        "fraud_score": row.get("fraud_score", 0),
                    })
        df = df.drop(columns=["address_norm"])
    
    if address_groups:
        addr_df = pd.DataFrame(address_groups)
        addr_df = addr_df.sort_values(["shared_address", "customer_id"])
        addr_file = output_path / "linked_by_address.csv"
        addr_df.to_csv(addr_file, index=False)
        reports["address"] = addr_file
        
        print(f"\nFound {addr_df['shared_address'].nunique()} shared address groups")
        print(f"Saved to: {addr_file}")
        
        for addr in addr_df["shared_address"].unique()[:10]:  # Limit output
            group = addr_df[addr_df["shared_address"] == addr]
            fraud_in_group = group["is_fraud"].any()
            fraud_marker = " ⚠️ FRAUD" if fraud_in_group else ""
            addr_short = addr[:50] + "..." if len(str(addr)) > 50 else addr
            print(f"\n  Address: {addr_short}{fraud_marker}")
            for _, row in group.iterrows():
                fraud_tag = " [FRAUD]" if row["is_fraud"] else ""
                print(f"    → {row['customer_id']}: {row['surname']}, {row['first_name']}{fraud_tag}")
    
    # =========================================================================
    # 3. SIMILAR NAMES (TYPO VARIANTS)
    # =========================================================================
    print("\n" + "=" * 70)
    print("3. SIMILAR NAME PAIRS (Potential Typo Variants)")
    print("=" * 70)
    
    name_pairs = []
    if "surname" in df.columns and "first_name" in df.columns:
        # Compare all pairs with high similarity but not exact match
        for i in range(len(df)):
            for j in range(i + 1, len(df)):
                row_i = df.iloc[i]
                row_j = df.iloc[j]
                
                name_i = f"{row_i['first_name']} {row_i['surname']}".lower()
                name_j = f"{row_j['first_name']} {row_j['surname']}".lower()
                
                if name_i == name_j:
                    continue  # Skip exact matches
                
                similarity = jellyfish.jaro_winkler_similarity(name_i, name_j)
                
                if similarity >= 0.85:  # High similarity threshold
                    name_pairs.append({
                        "similarity_score": round(similarity, 3),
                        "customer_id_1": row_i["customer_id"],
                        "name_1": f"{row_i['first_name']} {row_i['surname']}",
                        "is_fraud_1": row_i.get("is_fraud", False),
                        "fraud_type_1": row_i.get("fraud_type", ""),
                        "customer_id_2": row_j["customer_id"],
                        "name_2": f"{row_j['first_name']} {row_j['surname']}",
                        "is_fraud_2": row_j.get("is_fraud", False),
                        "fraud_type_2": row_j.get("fraud_type", ""),
                        "address_1": row_i.get("address", ""),
                        "address_2": row_j.get("address", ""),
                        "iban_1": row_i.get("iban", ""),
                        "iban_2": row_j.get("iban", ""),
                    })
    
    if name_pairs:
        names_df = pd.DataFrame(name_pairs)
        names_df = names_df.sort_values("similarity_score", ascending=False)
        names_file = output_path / "linked_by_similar_names.csv"
        names_df.to_csv(names_file, index=False)
        reports["similar_names"] = names_file
        
        print(f"\nFound {len(names_df)} similar name pairs (>85% similarity)")
        print(f"Saved to: {names_file}")
        
        for _, row in names_df.head(15).iterrows():
            fraud_1 = " [FRAUD]" if row["is_fraud_1"] else ""
            fraud_2 = " [FRAUD]" if row["is_fraud_2"] else ""
            print(f"\n  Similarity: {row['similarity_score']:.1%}")
            print(f"    → {row['customer_id_1']}: {row['name_1']}{fraud_1}")
            print(f"    → {row['customer_id_2']}: {row['name_2']}{fraud_2}")
    
    # =========================================================================
    # 4. COMBINED MASTER LINK REPORT
    # =========================================================================
    print("\n" + "=" * 70)
    print("4. MASTER LINK REPORT")
    print("=" * 70)
    
    # Create a master report with all link types
    master_links = []
    
    # Add IBAN links
    if "iban" in df.columns:
        for iban, group in df.groupby("iban"):
            if len(group) > 1:
                customer_ids = group["customer_id"].tolist()
                for i, cid in enumerate(customer_ids):
                    row = group[group["customer_id"] == cid].iloc[0]
                    other_ids = [c for c in customer_ids if c != cid]
                    master_links.append({
                        "customer_id": cid,
                        "link_type": "SHARED_IBAN",
                        "link_value": iban,
                        "linked_to": "; ".join(other_ids),
                        "surname": row.get("surname", ""),
                        "first_name": row.get("first_name", ""),
                        "address": row.get("address", ""),
                        "email": row.get("email", ""),
                        "is_fraud": row.get("is_fraud", False),
                        "fraud_type": row.get("fraud_type", ""),
                        "detected_fraud": row.get("detected_fraud", False),
                    })
    
    # Add address links
    if "address" in df.columns:
        df["address_norm"] = df["address"].str.lower().str.strip()
        for addr, group in df.groupby("address_norm"):
            if len(group) > 1 and pd.notna(addr) and addr != "":
                customer_ids = group["customer_id"].tolist()
                for cid in customer_ids:
                    row = group[group["customer_id"] == cid].iloc[0]
                    other_ids = [c for c in customer_ids if c != cid]
                    # Check if already added via IBAN
                    existing = [m for m in master_links 
                               if m["customer_id"] == cid and set(m["linked_to"].split("; ")) == set(other_ids)]
                    if not existing:
                        master_links.append({
                            "customer_id": cid,
                            "link_type": "SHARED_ADDRESS",
                            "link_value": row["address"],
                            "linked_to": "; ".join(other_ids),
                            "surname": row.get("surname", ""),
                            "first_name": row.get("first_name", ""),
                            "address": row.get("address", ""),
                            "email": row.get("email", ""),
                            "is_fraud": row.get("is_fraud", False),
                            "fraud_type": row.get("fraud_type", ""),
                            "detected_fraud": row.get("detected_fraud", False),
                        })
        df = df.drop(columns=["address_norm"], errors="ignore")
    
    if master_links:
        master_df = pd.DataFrame(master_links)
        master_df = master_df.sort_values(["link_type", "customer_id"])
        master_file = output_path / "master_link_report.csv"
        master_df.to_csv(master_file, index=False)
        reports["master"] = master_file
        
        print(f"\nTotal link entries: {len(master_df)}")
        print(f"By type:")
        for link_type, count in master_df["link_type"].value_counts().items():
            print(f"  - {link_type}: {count}")
        print(f"\nSaved to: {master_file}")
    
    print("\n" + "=" * 70)
    print("REPORTS GENERATED:")
    for name, path in reports.items():
        print(f"  - {name}: {path}")
    print("=" * 70)
    
    return reports


if __name__ == "__main__":
    generate_focused_links_report()
