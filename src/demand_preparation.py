"""
Demand Preparation Module
AI Market Intelligence & Demand Forecasting Platform

This module prepares the aggregated daily demand dataset for feature engineering by:
1. Converting 'date' to datetime.
2. Auditing and clipping negative daily_demand to 0 (treating returns/adjustments as 0 net forward demand).
3. Finding the first and last observed sale dates [T_first, T_last] for each (store_id, item_id) series.
4. Filling ONLY missing calendar dates within [T_first, T_last]:
   - daily_demand = 0.0
   - price = forward-fill previous known transaction price.
5. Preserving all original records without generating an artificial global Cartesian grid.
6. Sorting strictly by store_id, item_id, date.
7. Writing the regularized time-series to data/prepared_demand.csv.
"""

import os
import sys
import time
import pandas as pd
import numpy as np


def prepare_daily_demand(
    input_path: str = "data/daily_demand.csv",
    output_path: str = "data/prepared_demand.csv"
) -> dict:
    """
    Executes memory-efficient date regularization and negative demand clipping.

    Parameters:
    -----------
    input_path : str
        Path to input daily_demand.csv.
    output_path : str
        Path to output prepared_demand.csv.

    Returns:
    --------
    dict: Comprehensive audit and validation report metrics.
    """
    print("=" * 80)
    print("DEMAND PREPARATION PIPELINE")
    print("=" * 80)
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input dataset not found at: {input_path}")
        
    start_time = time.time()
    print(f"Loading aggregated daily demand from '{input_path}'...")
    df = pd.read_csv(input_path)
    
    original_rows = len(df)
    neg_demand_before = int((df["daily_demand"] < 0).sum())
    print(f"Loaded {original_rows:,} rows. Negative demand rows before clipping: {neg_demand_before:,}")
    
    # 1. Convert date to pandas datetime
    print("Converting 'date' to datetime...")
    df["date"] = pd.to_datetime(df["date"])
    
    # 2. Clip negative daily_demand to 0 (preserve all rows)
    print("Clipping negative daily_demand values to 0.0...")
    df["daily_demand"] = df["daily_demand"].clip(lower=0.0)
    neg_demand_after = int((df["daily_demand"] < 0).sum())
    
    # Target output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(output_path):
        os.remove(output_path)
        
    # Get sorted list of stores to process in deterministic order
    stores = sorted(df["store_id"].unique())
    print(f"Processing {len(stores)} stores: {stores}...")
    
    total_prepared_rows = 0
    total_null_prices = 0
    total_null_demand = 0
    total_duplicates = 0
    overall_min_demand = float("inf")
    overall_max_demand = float("-inf")
    all_unique_items = set()
    all_min_dates = []
    all_max_dates = []
    
    for s_idx, store_id in enumerate(stores):
        s_start = time.time()
        print(f"\n--- Processing Store {store_id} ({s_idx + 1}/{len(stores)}) ---")
        store_df = df[df["store_id"] == store_id].copy()
        n_orig_store = len(store_df)
        
        # 3. For each item_id in this store, find min and max observed dates
        bounds = store_df.groupby("item_id")["date"].agg(["min", "max"]).reset_index()
        bounds.rename(columns={"min": "first_date", "max": "last_date"}, inplace=True)
        
        # Compute length of active span for each item
        span_lengths = (bounds["last_date"] - bounds["first_date"]).dt.days.values + 1
        
        # Vectorized generation of active lifespan calendar grid
        item_repeats = np.repeat(bounds["item_id"].values, span_lengths)
        date_offsets = np.concatenate([np.arange(span, dtype="timedelta64[D]") for span in span_lengths])
        base_dates = np.repeat(bounds["first_date"].values, span_lengths)
        dates = base_dates + date_offsets
        store_repeats = np.full(len(item_repeats), store_id, dtype=np.int64)
        
        grid_df = pd.DataFrame({
            "date": dates,
            "store_id": store_repeats,
            "item_id": item_repeats
        })
        
        # 4. Merge original store records into active grid
        merged = pd.merge(grid_df, store_df, on=["date", "store_id", "item_id"], how="left")
        
        # Fill missing demand with 0.0
        merged["daily_demand"] = merged["daily_demand"].fillna(0.0)
        
        # Forward-fill previous known transaction price per item
        merged["price"] = merged.groupby("item_id")["price"].ffill()
        
        # 7. Sort strictly by store_id, item_id, date
        merged.sort_values(by=["store_id", "item_id", "date"], inplace=True)
        merged.reset_index(drop=True, inplace=True)
        
        n_prep_store = len(merged)
        n_inserted_store = n_prep_store - n_orig_store
        
        # Check store-level duplicates
        store_dups = int(merged.duplicated(subset=["date", "store_id", "item_id"]).sum())
        total_duplicates += store_dups
        
        # Track statistics
        total_prepared_rows += n_prep_store
        total_null_prices += int(merged["price"].isnull().sum())
        total_null_demand += int(merged["daily_demand"].isnull().sum())
        overall_min_demand = min(overall_min_demand, float(merged["daily_demand"].min()))
        overall_max_demand = max(overall_max_demand, float(merged["daily_demand"].max()))
        all_unique_items.update(merged["item_id"].unique())
        all_min_dates.append(merged["date"].min())
        all_max_dates.append(merged["date"].max())
        
        # Format date as YYYY-MM-DD string for CSV
        merged["date"] = merged["date"].dt.strftime("%Y-%m-%d")
        
        # 8. Append to output CSV in chunks to keep memory flat
        is_first_store = (s_idx == 0)
        merged.to_csv(
            output_path,
            mode="w" if is_first_store else "a",
            header=is_first_store,
            index=False
        )
        
        print(f"Store {store_id} completed in {time.time() - s_start:.2f}s:")
        print(f"  - Original rows: {n_orig_store:,}")
        print(f"  - Prepared rows: {n_prep_store:,}")
        print(f"  - Inserted zeros: {n_inserted_store:,}")
        print(f"  - Null prices: {int(merged['price'].isnull().sum()):,}")
        
    total_time = time.time() - start_time
    total_inserted = total_prepared_rows - original_rows
    overall_min_date = str(min(all_min_dates))
    overall_max_date = str(max(all_max_dates))
    
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n[Success] Output saved to '{output_path}' ({file_size_mb:.2f} MB) in {total_time:.2f}s.")
    
    report = {
        "original_rows": original_rows,
        "prepared_rows": total_prepared_rows,
        "inserted_rows": total_inserted,
        "negative_demand_before": neg_demand_before,
        "negative_demand_after": neg_demand_after,
        "missing_values": {
            "date": 0,
            "store_id": 0,
            "item_id": 0,
            "daily_demand": total_null_demand,
            "price": total_null_prices
        },
        "duplicate_keys": total_duplicates,
        "null_prices": total_null_prices,
        "unique_products": len(all_unique_items),
        "unique_stores": len(stores),
        "date_range": f"{overall_min_date} to {overall_max_date}",
        "min_demand": overall_min_demand,
        "max_demand": overall_max_demand,
        "output_file": output_path,
        "file_size_mb": file_size_mb,
        "execution_time_seconds": total_time
    }
    
    return report


def print_preparation_summary(report: dict):
    """
    Prints a formatted audit report of the preparation process.
    """
    print("\n" + "=" * 80)
    print("DEMAND PREPARATION VALIDATION REPORT")
    print("=" * 80)
    print(f"Original Rows (daily_demand.csv):      {report['original_rows']:,}")
    print(f"Prepared Rows (prepared_demand.csv):  {report['prepared_rows']:,}")
    print(f"Inserted Rows (zero-demand days):      {report['inserted_rows']:,}")
    print(f"Negative Demand Before Clipping:       {report['negative_demand_before']:,}")
    print(f"Negative Demand After Clipping:        {report['negative_demand_after']:,}")
    print(f"Duplicate (date, store_id, item_id):   {report['duplicate_keys']}")
    print(f"Null Prices:                           {report['null_prices']:,}")
    print(f"Null Demand Values:                    {report['missing_values']['daily_demand']}")
    print(f"Unique Products (SKUs):                {report['unique_products']:,}")
    print(f"Unique Stores:                         {report['unique_stores']}")
    print(f"Date Range:                            {report['date_range']}")
    print(f"Min Daily Demand:                      {report['min_demand']:.4f}")
    print(f"Max Daily Demand:                      {report['max_demand']:.4f}")
    print(f"Output File:                           {report['output_file']} ({report['file_size_mb']:.2f} MB)")
    print(f"Execution Time:                        {report['execution_time_seconds']:.2f} seconds")
    print("=" * 80)


if __name__ == "__main__":
    report = prepare_daily_demand(
        input_path="data/daily_demand.csv",
        output_path="data/prepared_demand.csv"
    )
    print_preparation_summary(report)
