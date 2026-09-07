"""
Daily Demand Aggregation Module
AI Market Intelligence & Demand Forecasting Platform

This module handles:
1. Aggregating transaction-level sales data to daily demand granularity:
   (date, store_id, item_id)
2. Computing target variable:
   daily_demand = sum(quantity)
3. Computing daily price feature:
   price = mean(price_base)  [computed after Option A price correction]
4. Validating the aggregated dataset (nulls, duplicates, ranges, negative values).
5. Saving validation visualization to disk.
6. Saving clean aggregated dataset to data/daily_demand.csv.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Headless backend for clean script execution
import matplotlib.pyplot as plt

# Enable importing from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.preprocessing import load_and_preprocess_sales


def aggregate_daily_demand(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates transaction-level data to (date, store_id, item_id) granularity.

    Granularity:
    - date: calendar date of sale
    - store_id: unique store identifier
    - item_id: unique product SKU/item identifier

    Metrics:
    - daily_demand: sum of quantity sold
    - price: mean transaction base price for that item-store-date combination
    """
    print("\n" + "=" * 80)
    print("AGGREGATING SALES DATA TO DAILY DEMAND")
    print("=" * 80)
    print("Grouping by: ['date', 'store_id', 'item_id']...")
    
    # Ensure date is standard string or datetime
    if not np.issubdtype(df["date"].dtype, np.datetime64):
        df["date"] = pd.to_datetime(df["date"])

    agg_df = df.groupby(["date", "store_id", "item_id"], as_index=False).agg(
        daily_demand=("quantity", "sum"),
        price=("price_base", "mean")
    )
    
    # Sort chronologically and by store/item for deterministic ordering
    agg_df.sort_values(by=["date", "store_id", "item_id"], inplace=True)
    agg_df.reset_index(drop=True, inplace=True)
    
    print(f"Aggregation complete. Resulting shape: {agg_df.shape}")
    return agg_df


def validate_daily_demand(df: pd.DataFrame) -> dict:
    """
    Validates the aggregated daily demand dataset against required checks:
    - number of rows
    - number of unique products
    - number of unique stores
    - minimum date
    - maximum date
    - missing values
    - duplicate (date, store_id, item_id) combinations
    - minimum daily_demand
    - maximum daily_demand
    - mean daily_demand
    - number of negative daily_demand values
    - price distribution checks (min, max, mean, nulls, negatives)
    """
    total_rows = len(df)
    unique_products = int(df["item_id"].nunique())
    unique_stores = int(df["store_id"].nunique())
    min_date = str(df["date"].min())
    max_date = str(df["date"].max())
    missing_vals = df.isnull().sum().to_dict()
    
    dup_keys = int(df.duplicated(subset=["date", "store_id", "item_id"]).sum())
    
    min_demand = float(df["daily_demand"].min())
    max_demand = float(df["daily_demand"].max())
    mean_demand = float(df["daily_demand"].mean())
    median_demand = float(df["daily_demand"].median())
    
    neg_demand = int((df["daily_demand"] < 0).sum())
    neg_demand_pct = (neg_demand / total_rows) * 100 if total_rows > 0 else 0.0
    zero_demand = int((df["daily_demand"] == 0).sum())
    
    # Price metrics
    min_price = float(df["price"].min(skipna=True))
    max_price = float(df["price"].max(skipna=True))
    mean_price = float(df["price"].mean(skipna=True))
    neg_price_count = int((df["price"] < 0).sum())
    null_price_count = int(df["price"].isnull().sum())
    
    validation_results = {
        "number_of_rows": total_rows,
        "unique_products": unique_products,
        "unique_stores": unique_stores,
        "min_date": min_date,
        "max_date": max_date,
        "missing_values": missing_vals,
        "duplicate_combinations": dup_keys,
        "min_daily_demand": min_demand,
        "max_daily_demand": max_demand,
        "mean_daily_demand": mean_demand,
        "median_daily_demand": median_demand,
        "negative_daily_demand_count": neg_demand,
        "negative_daily_demand_pct": neg_demand_pct,
        "zero_daily_demand_count": zero_demand,
        "min_price": min_price,
        "max_price": max_price,
        "mean_price": mean_price,
        "negative_price_count": neg_price_count,
        "null_price_count": null_price_count,
    }
    
    return validation_results


def print_validation_report(val: dict):
    """
    Print formatted validation report to stdout.
    """
    print("\n" + "=" * 80)
    print("AGGREGATED DATASET VALIDATION REPORT")
    print("=" * 80)
    print(f"Total Rows:                  {val['number_of_rows']:,}")
    print(f"Unique Products (SKUs):      {val['unique_products']:,}")
    print(f"Unique Stores:               {val['unique_stores']:,}")
    print(f"Date Range:                  {val['min_date']} to {val['max_date']}")
    print(f"Duplicate (date,store,item): {val['duplicate_combinations']}")
    print("\nMissing values:")
    for col, cnt in val["missing_values"].items():
        print(f"  - {col}: {cnt}")
    print("\nTarget Variable (daily_demand) Distribution:")
    print(f"  - Minimum:                 {val['min_daily_demand']:.4f}")
    print(f"  - Maximum:                 {val['max_daily_demand']:.4f}")
    print(f"  - Mean:                    {val['mean_daily_demand']:.4f}")
    print(f"  - Median:                  {val['median_daily_demand']:.4f}")
    print(f"  - Negative Demand Rows:    {val['negative_daily_demand_count']:,} ({val['negative_daily_demand_pct']:.4f}%)")
    print(f"  - Zero Demand Rows:        {val['zero_daily_demand_count']:,}")
    print("\nPrice Feature Distribution (after Option A Correction):")
    print(f"  - Minimum:                 {val['min_price']:.4f}")
    print(f"  - Maximum:                 {val['max_price']:.4f}")
    print(f"  - Mean:                    {val['mean_price']:.4f}")
    print(f"  - Negative Price Rows:     {val['negative_price_count']}")
    print(f"  - Null Price Rows:         {val['null_price_count']} (unmatchable negative price item)")
    print("=" * 80)


def plot_daily_demand_validation(
    df: pd.DataFrame,
    output_plot_path: str = "data/daily_demand_validation_plot.png"
):
    """
    Creates and saves a validation plot showing overall daily demand over time.
    """
    print(f"\nGenerating daily demand validation plot -> '{output_plot_path}'...")
    daily_total = df.groupby("date")["daily_demand"].sum()
    
    plt.figure(figsize=(14, 6), dpi=120)
    plt.plot(daily_total.index, daily_total.values, color="#1f77b4", linewidth=1.5, label="Total Market Daily Demand")
    
    # 7-day moving average for trend visualization
    rolling_7d = daily_total.rolling(window=7, min_periods=1).mean()
    plt.plot(daily_total.index, rolling_7d.values, color="#ff7f0e", linewidth=2.0, linestyle="--", label="7-Day Rolling Trend")
    
    plt.title("Overall Daily Demand Over Time (Validation Plot)", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Total Quantity Sold", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)
    plt.savefig(output_plot_path)
    plt.close()
    print(f"Validation plot saved successfully to '{output_plot_path}'.")


def save_daily_demand(df: pd.DataFrame, output_path: str = "data/daily_demand.csv"):
    """
    Saves the validated aggregated dataset to CSV.
    """
    print(f"\nSaving aggregated dataset to '{output_path}'...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Format date cleanly as YYYY-MM-DD
    if np.issubdtype(df["date"].dtype, np.datetime64):
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")
        
    df.to_csv(output_path, index=False)
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Dataset saved successfully. File size: {file_size_mb:.2f} MB ({len(df):,} rows).")


def run_pipeline():
    """
    Full execution pipeline for Data Preprocessing + Daily Demand Aggregation.
    """
    # Step 1: Preprocess raw sales with Option A price correction
    sales_df, meta = load_and_preprocess_sales(
        filepath="data/sales.csv",
        drop_unnamed=True,
        convert_datetime=True,
        preserve_negative_quantity=True,
        apply_price_option_a=True
    )
    
    # Step 2: Aggregate to (date, store_id, item_id)
    agg_df = aggregate_daily_demand(sales_df)
    
    # Step 3: Validate
    validation_results = validate_daily_demand(agg_df)
    print_validation_report(validation_results)
    
    # Step 4: Generate validation plot
    plot_daily_demand_validation(agg_df, "data/daily_demand_validation_plot.png")
    
    # Step 5: Save dataset
    save_daily_demand(agg_df, "data/daily_demand.csv")
    
    return agg_df, validation_results, meta


if __name__ == "__main__":
    run_pipeline()
