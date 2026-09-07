"""
Preprocessing Module for Sales Transaction Data
AI Market Intelligence & Demand Forecasting Platform

This module handles:
1. Auditing raw sales data (duplicates, nulls, negative quantities, negative prices, negative sum_totals).
2. Dropping redundant index columns ('Unnamed: 0').
3. Converting date strings to pandas datetime.
4. Preserving return/adjustment records (negative quantities) with transparent reporting.
5. Correcting negative price_base records (Option A):
   - Replace negative price_base with the median positive price_base for the same (store_id, item_id).
   - If no positive price exists for that exact (store_id, item_id), assign NaN and report the case.
"""

import os
import sys
import pandas as pd
import numpy as np


def audit_sales_data(df: pd.DataFrame) -> dict:
    """
    Perform a comprehensive audit of the sales dataset.
    Returns a dictionary of audit metrics.
    """
    total_records = len(df)
    
    # 1. Unnamed: 0 column
    has_unnamed = "Unnamed: 0" in df.columns
    
    # 2. Duplicates
    dup_records = int(df.duplicated().sum())
    dup_pct = (dup_records / total_records) * 100 if total_records > 0 else 0.0
    
    # 3. Missing values
    missing_counts = df.isnull().sum().to_dict()
    
    # 4. Quantity checks
    neg_qty = int((df["quantity"] < 0).sum())
    neg_qty_pct = (neg_qty / total_records) * 100 if total_records > 0 else 0.0
    zero_qty = int((df["quantity"] == 0).sum())
    zero_qty_pct = (zero_qty / total_records) * 100 if total_records > 0 else 0.0
    
    # 5. Price checks
    neg_price = int((df["price_base"] < 0).sum())
    neg_price_pct = (neg_price / total_records) * 100 if total_records > 0 else 0.0
    zero_price = int((df["price_base"] == 0).sum())
    zero_price_pct = (zero_price / total_records) * 100 if total_records > 0 else 0.0
    
    # 6. Sum total checks
    neg_sum_total = int((df["sum_total"] < 0).sum())
    neg_sum_total_pct = (neg_sum_total / total_records) * 100 if total_records > 0 else 0.0
    
    audit_results = {
        "total_records": total_records,
        "has_unnamed": has_unnamed,
        "duplicate_records": dup_records,
        "duplicate_pct": dup_pct,
        "missing_values": missing_counts,
        "negative_quantity_count": neg_qty,
        "negative_quantity_pct": neg_qty_pct,
        "zero_quantity_count": zero_qty,
        "zero_quantity_pct": zero_qty_pct,
        "negative_price_count": neg_price,
        "negative_price_pct": neg_price_pct,
        "zero_price_count": zero_price,
        "zero_price_pct": zero_price_pct,
        "negative_sum_total_count": neg_sum_total,
        "negative_sum_total_pct": neg_sum_total_pct,
    }
    
    return audit_results


def correct_negative_prices(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Option A: Corrects negative price_base values.
    
    Rule:
    - Replace each negative price_base with the median positive price_base for the same item_id + store_id.
    - If no positive price exists for that exact item_id + store_id, set price_base to NaN and record the case.
    
    Returns:
    --------
    tuple[pd.DataFrame, dict]: (corrected dataframe, report dictionary)
    """
    neg_mask = df["price_base"] < 0
    neg_count = int(neg_mask.sum())
    
    if neg_count == 0:
        return df, {"total_negative_prices": 0, "imputed_count": 0, "unmatched_count": 0, "unmatched_records": []}
        
    print(f"\n[Price Correction - Option A] Found {neg_count} records with negative price_base.")
    
    # Compute median positive price per (store_id, item_id)
    pos_df = df[df["price_base"] > 0]
    store_item_medians = pos_df.groupby(["store_id", "item_id"])["price_base"].median()
    
    neg_indices = df[neg_mask].index
    imputed_records = []
    unmatched_records = []
    
    for idx in neg_indices:
        s_id = df.at[idx, "store_id"]
        it_id = df.at[idx, "item_id"]
        old_val = df.at[idx, "price_base"]
        
        if (s_id, it_id) in store_item_medians:
            med_price = float(store_item_medians.loc[(s_id, it_id)])
            df.at[idx, "price_base"] = med_price
            imputed_records.append({
                "index": idx,
                "date": df.at[idx, "date"],
                "store_id": s_id,
                "item_id": it_id,
                "old_price": old_val,
                "imputed_price": med_price
            })
        else:
            df.at[idx, "price_base"] = np.nan
            unmatched_records.append({
                "index": idx,
                "date": df.at[idx, "date"],
                "store_id": s_id,
                "item_id": it_id,
                "old_price": old_val,
                "imputed_price": np.nan,
                "note": "No positive price_base found for this exact (store_id, item_id)"
            })
            
    print(f"[Price Correction] Imputed {len(imputed_records)} records using median positive price for (store_id, item_id).")
    if unmatched_records:
        print(f"[Price Correction WARNING] {len(unmatched_records)} record(s) could not be matched with positive price at store; set to NaN:")
        for r in unmatched_records:
            print(f"  - Date: {r['date']}, Store: {r['store_id']}, Item: {r['item_id']}, Old Price: {r['old_price']}")
            
    report = {
        "total_negative_prices": neg_count,
        "imputed_count": len(imputed_records),
        "unmatched_count": len(unmatched_records),
        "unmatched_records": unmatched_records,
        "sample_imputed": imputed_records[:5]
    }
    
    return df, report


def print_audit_report(audit: dict):
    """
    Print a structured, readable audit report to stdout.
    """
    print("=" * 80)
    print("SALES DATA AUDIT REPORT")
    print("=" * 80)
    print(f"Total Records: {audit['total_records']:,}")
    print(f"Has 'Unnamed: 0' column: {audit['has_unnamed']}")
    print(f"Duplicate rows: {audit['duplicate_records']:,} ({audit['duplicate_pct']:.4f}%)")
    print("\nMissing values per column:")
    for col, cnt in audit["missing_values"].items():
        print(f"  - {col}: {cnt}")
    
    print("\nProblematic Records Analysis:")
    print(f"  - Negative Quantity: {audit['negative_quantity_count']:,} ({audit['negative_quantity_pct']:.4f}%)")
    print(f"    [Note: Represents returns/refunds/adjustments; preserved by design]")
    print(f"  - Zero Quantity:     {audit['zero_quantity_count']:,} ({audit['zero_quantity_pct']:.4f}%)")
    print(f"  - Negative Price:    {audit['negative_price_count']:,} ({audit['negative_price_pct']:.4f}%)")
    print(f"  - Zero Price:        {audit['zero_price_count']:,} ({audit['zero_price_pct']:.4f}%)")
    print(f"  - Negative Sum Total:{audit['negative_sum_total_count']:,} ({audit['negative_sum_total_pct']:.4f}%)")
    print(f"    [Note: Sum total is not the target; daily_demand = sum(quantity) is target]")
    print("=" * 80)


def load_and_preprocess_sales(
    filepath: str = "data/sales.csv",
    drop_unnamed: bool = True,
    convert_datetime: bool = True,
    preserve_negative_quantity: bool = True,
    apply_price_option_a: bool = True
) -> tuple[pd.DataFrame, dict]:
    """
    Loads raw sales dataset, runs auditing, applies cleaning and Option A price correction.

    Parameters:
    -----------
    filepath : str
        Path to sales.csv.
    drop_unnamed : bool
        Whether to drop 'Unnamed: 0'.
    convert_datetime : bool
        Whether to convert 'date' to datetime.
    preserve_negative_quantity : bool
        If True, keeps negative quantity values as returns/adjustments.
    apply_price_option_a : bool
        If True, replaces negative prices with median positive price per (store_id, item_id).

    Returns:
    --------
    tuple[pd.DataFrame, dict]: (Cleaned dataframe, dictionary of audit and correction reports)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Sales dataset not found at: {filepath}")

    print(f"Loading raw sales data from '{filepath}'...")
    df = pd.read_csv(filepath)
    
    # Run initial audit
    audit = audit_sales_data(df)
    print_audit_report(audit)
    
    # 1. Remove unnecessary Unnamed: 0 column
    if drop_unnamed and "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)
        print("[Preprocessing] Dropped column 'Unnamed: 0'")
        
    # 2. Convert date column to pandas datetime
    if convert_datetime:
        df["date"] = pd.to_datetime(df["date"])
        print(f"[Preprocessing] Converted 'date' to datetime (range: {df['date'].min()} to {df['date'].max()})")
        
    # 3. Negative quantities
    if preserve_negative_quantity:
        print(f"[Preprocessing] Preserved {audit['negative_quantity_count']:,} negative quantity records (returns/adjustments)")
    else:
        df = df[df["quantity"] >= 0].copy()
        print(f"[Preprocessing] Filtered out negative quantity records")
        
    # 4. Option A: Negative price correction
    price_report = {}
    if apply_price_option_a:
        df, price_report = correct_negative_prices(df)

    metadata = {
        "audit": audit,
        "price_correction": price_report
    }
    return df, metadata


if __name__ == "__main__":
    df, meta = load_and_preprocess_sales("data/sales.csv", apply_price_option_a=True)
    print(f"Preprocessed dataframe shape: {df.shape}")
    print(df.head())
