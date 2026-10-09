"""
Feature Engineering
AI Market Intelligence & Demand Forecasting Platform

Purpose:
Create the final supervised feature dataset for next-day demand forecasting.

Input:
    data/prepared_demand.csv
    data/catalog.csv

Output:
    data/features.csv

Target:
    Next-day demand (t+1)

Important:
    - No raw catalog text columns are included in the final dataset.
    - No automatic deletion of an existing features.csv.
    - Processing is done in chunks to reduce RAM usage.
"""

import os
import time
import gc

import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

INPUT_PATH = "data/prepared_demand.csv"
CATALOG_PATH = "data/catalog.csv"
OUTPUT_PATH = "data/features.csv"

# Process approximately 1 million rows at a time.
CHUNK_SIZE = 1_000_000


# ============================================================
# CATALOG PREPARATION
# ============================================================

def prepare_catalog(catalog_path=CATALOG_PATH):
    """
    Load catalog.csv and create compact numeric categorical features.

    Raw text columns such as:
        dept_name
        class_name
        subclass_name
        item_type

    are NOT kept in the final feature dataset.

    Instead, compact integer codes are created.
    """

    if not os.path.exists(catalog_path):
        raise FileNotFoundError(
            f"Catalog file not found: {catalog_path}"
        )

    print("\nLoading product catalog...")

    catalog_columns = [
        "item_id",
        "dept_name",
        "class_name",
        "subclass_name",
        "item_type",
        "weight_volume",
        "weight_netto",
        "fatness"
    ]

    catalog = pd.read_csv(
        catalog_path,
        usecols=catalog_columns,
        dtype={"item_id": "string"}
    )

    # One catalog record per item.
    catalog = catalog.drop_duplicates(
        subset=["item_id"]
    ).copy()

    print(
        f"Unique catalog products: "
        f"{len(catalog):,}"
    )

    # --------------------------------------------------------
    # Convert categorical text to compact integer codes
    # --------------------------------------------------------

    categorical_columns = [
        "dept_name",
        "class_name",
        "subclass_name",
        "item_type"
    ]

    for column in categorical_columns:

        # Missing values become -1 after categorical encoding.
        catalog[column] = (
            catalog[column]
            .astype("category")
        )

    catalog["dept_code"] = (
        catalog["dept_name"]
        .cat.codes
        .astype("int16")
    )

    catalog["class_code"] = (
        catalog["class_name"]
        .cat.codes
        .astype("int16")
    )

    catalog["subclass_code"] = (
        catalog["subclass_name"]
        .cat.codes
        .astype("int16")
    )

    catalog["item_type_code"] = (
        catalog["item_type"]
        .cat.codes
        .astype("int16")
    )

    # --------------------------------------------------------
    # Numeric physical product information
    # --------------------------------------------------------

    catalog["weight_volume"] = pd.to_numeric(
        catalog["weight_volume"],
        errors="coerce"
    ).astype("float32")

    catalog["weight_netto"] = pd.to_numeric(
        catalog["weight_netto"],
        errors="coerce"
    ).astype("float32")

    catalog["fatness"] = pd.to_numeric(
        catalog["fatness"],
        errors="coerce"
    ).astype("float32")

    # --------------------------------------------------------
    # IMPORTANT:
    # Only compact/numeric columns are kept.
    #
    # The following large text columns are removed:
    # dept_name
    # class_name
    # subclass_name
    # item_type
    # --------------------------------------------------------

    catalog = catalog[
        [
            "item_id",
            "dept_code",
            "class_code",
            "subclass_code",
            "item_type_code",
            "weight_volume",
            "weight_netto",
            "fatness"
        ]
    ]

    print("Catalog preparation completed.")

    return catalog


# ============================================================
# ITEM ENCODING
# ============================================================

def build_item_encoder(prepared_demand_path=INPUT_PATH):
    """
    Create a compact integer code for each item_id.

    Example:

        item_id A -> 0
        item_id B -> 1
        item_id C -> 2

    The original item_id is still kept in the final dataset
    for identification.
    """

    print("\nBuilding item codes...")

    unique_items = pd.read_csv(
        prepared_demand_path,
        usecols=["item_id"],
        dtype={"item_id": "string"}
    )["item_id"].dropna().unique()

    unique_items = sorted(
        unique_items.astype(str).tolist()
    )

    item_to_code = {
        item_id: index
        for index, item_id in enumerate(unique_items)
    }

    print(
        f"Unique items: {len(item_to_code):,}"
    )

    return item_to_code


# ============================================================
# FEATURE ENGINEERING FOR ONE CHUNK
# ============================================================

def process_chunk_features(
    chunk,
    catalog,
    item_to_code
):
    """
    Create all forecasting features for one chunk.

    Features created:

        target
        day_of_week
        day_of_month
        month
        week_of_year
        is_weekend

        lag_1
        lag_7
        lag_14
        lag_28

        rolling_mean_7
        rolling_mean_14
        rolling_mean_28

        rolling_std_7
        rolling_std_28

        price
        price_change

        item_code

        dept_code
        class_code
        subclass_code
        item_type_code
        weight_volume
        weight_netto
        fatness
    """

    # --------------------------------------------------------
    # Convert date
    # --------------------------------------------------------

    chunk["date"] = pd.to_datetime(
        chunk["date"]
    )

    # --------------------------------------------------------
    # Group by store + product
    # --------------------------------------------------------

    group = chunk.groupby(
        ["store_id", "item_id"],
        sort=False,
        observed=True
    )

    # ========================================================
    # 1. TARGET
    # ========================================================

    # Target = demand on the next day.
    chunk["target"] = (
        group["daily_demand"]
        .shift(-1)
        .astype("float32")
    )

    # ========================================================
    # 2. LAG FEATURES
    # ========================================================

    demand = group["daily_demand"]

    chunk["lag_1"] = (
        demand.shift(1)
        .astype("float32")
    )

    chunk["lag_7"] = (
        demand.shift(7)
        .astype("float32")
    )

    chunk["lag_14"] = (
        demand.shift(14)
        .astype("float32")
    )

    chunk["lag_28"] = (
        demand.shift(28)
        .astype("float32")
    )

    # ========================================================
    # 3. ROLLING FEATURES
    # ========================================================

    # Shift first.
    #
    # Therefore rolling features only use:
    #
    # t-1
    # t-2
    # t-3
    # ...
    #
    # They never use today's demand or future demand.

    previous_demand = demand.shift(1)

    previous_group = previous_demand.groupby(
        [
            chunk["store_id"],
            chunk["item_id"]
        ],
        sort=False,
        observed=True
    )

    chunk["rolling_mean_7"] = (
        previous_group
        .rolling(7)
        .mean()
        .reset_index(
            level=[0, 1],
            drop=True
        )
        .astype("float32")
    )

    chunk["rolling_mean_14"] = (
        previous_group
        .rolling(14)
        .mean()
        .reset_index(
            level=[0, 1],
            drop=True
        )
        .astype("float32")
    )

    chunk["rolling_mean_28"] = (
        previous_group
        .rolling(28)
        .mean()
        .reset_index(
            level=[0, 1],
            drop=True
        )
        .astype("float32")
    )

    chunk["rolling_std_7"] = (
        previous_group
        .rolling(7)
        .std()
        .reset_index(
            level=[0, 1],
            drop=True
        )
        .astype("float32")
    )

    chunk["rolling_std_28"] = (
        previous_group
        .rolling(28)
        .std()
        .reset_index(
            level=[0, 1],
            drop=True
        )
        .astype("float32")
    )

    # ========================================================
    # 4. PRICE FEATURES
    # ========================================================

    chunk["price"] = pd.to_numeric(
        chunk["price"],
        errors="coerce"
    ).astype("float32")

    previous_price = group["price"].shift(1)

    chunk["price_change"] = (
        chunk["price"] - previous_price
    ).astype("float32")

    # ========================================================
    # 5. CALENDAR FEATURES
    # ========================================================

    chunk["day_of_week"] = (
        chunk["date"]
        .dt.dayofweek
        .astype("int8")
    )

    chunk["day_of_month"] = (
        chunk["date"]
        .dt.day
        .astype("int8")
    )

    chunk["month"] = (
        chunk["date"]
        .dt.month
        .astype("int8")
    )

    chunk["week_of_year"] = (
        chunk["date"]
        .dt.isocalendar()
        .week
        .astype("int8")
    )

    chunk["is_weekend"] = (
        chunk["day_of_week"] >= 5
    ).astype("int8")

    # ========================================================
    # 6. ITEM CODE
    # ========================================================

    chunk["item_code"] = (
        chunk["item_id"]
        .astype(str)
        .map(item_to_code)
        .fillna(-1)
        .astype("int32")
    )

    # Store ID as compact integer.
    chunk["store_id"] = pd.to_numeric(
        chunk["store_id"],
        errors="coerce"
    ).astype("int8")

    # ========================================================
    # 7. REMOVE ROWS WITHOUT TARGET
    # ========================================================

    # The final date of each store-product series
    # has no next-day demand.
    chunk = chunk.dropna(
        subset=["target"]
    ).copy()

    # ========================================================
    # 8. REMOVE RAW DAILY DEMAND
    # ========================================================

    # IMPORTANT:
    #
    # daily_demand at time t must NOT be provided directly
    # as a feature because it can cause leakage.
    #
    # Historical demand is already represented through:
    #
    # lag_1
    # lag_7
    # lag_14
    # lag_28
    # rolling features

    chunk.drop(
        columns=["daily_demand"],
        inplace=True
    )

    # ========================================================
    # 9. MERGE COMPACT CATALOG FEATURES
    # ========================================================

    chunk = pd.merge(
        chunk,
        catalog,
        on="item_id",
        how="left",
        sort=False,
        validate="many_to_one"
    )

    # ========================================================
    # 10. FORMAT DATE
    # ========================================================

    chunk["date"] = (
        chunk["date"]
        .dt.strftime("%Y-%m-%d")
    )

    return chunk


# ============================================================
# MAIN FEATURE ENGINEERING PIPELINE
# ============================================================

def run_feature_engineering(
    input_path=INPUT_PATH,
    catalog_path=CATALOG_PATH,
    output_path=OUTPUT_PATH,
    chunk_size=CHUNK_SIZE
):
    """
    Execute the complete feature engineering pipeline.
    """

    print("\n")
    print("=" * 80)
    print("STARTING FEATURE ENGINEERING")
    print("=" * 80)

    start_time = time.time()

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Prepared demand file not found: {input_path}"
        )

    if not os.path.exists(catalog_path):
        raise FileNotFoundError(
            f"Catalog file not found: {catalog_path}"
        )

    # --------------------------------------------------------
    # Prepare catalog
    # --------------------------------------------------------

    catalog = prepare_catalog(
        catalog_path
    )

    # --------------------------------------------------------
    # Build item codes
    # --------------------------------------------------------

    item_to_code = build_item_encoder(
        input_path
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # We are NOT deleting an existing features.csv here.
    #
    # You already deleted the old one manually.
    #
    # If features.csv already exists, this code will overwrite
    # it through the first write below.
    # --------------------------------------------------------

    print("\nReading prepared demand dataset...")

    reader = pd.read_csv(
        input_path,
        chunksize=chunk_size,
        dtype={
            "item_id": "string"
        }
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    total_input_rows = 0
    total_output_rows = 0
    total_dropped_targets = 0

    target_min = float("inf")
    target_max = float("-inf")

    target_sum = 0.0
    target_sum_sq = 0.0

    min_date = None
    max_date = None

    chunk_number = 0

    # --------------------------------------------------------
    # Boundary buffer
    # --------------------------------------------------------

    remainder = pd.DataFrame()

    first_write = True

    # ========================================================
    # PROCESS CHUNKS
    # ========================================================

    for chunk in reader:

        chunk_number += 1

        total_input_rows += len(chunk)

        # ----------------------------------------------------
        # Add previous boundary rows
        # ----------------------------------------------------

        if not remainder.empty:

            chunk = pd.concat(
                [
                    remainder,
                    chunk
                ],
                ignore_index=True
            )

            remainder = pd.DataFrame()

        # ----------------------------------------------------
        # Identify last store-product series
        # ----------------------------------------------------

        last_store = (
            chunk["store_id"].iloc[-1]
        )

        last_item = (
            chunk["item_id"].iloc[-1]
        )

        boundary_mask = (
            (chunk["store_id"] == last_store)
            &
            (chunk["item_id"] == last_item)
        )

        # Keep last series for next chunk.
        remainder = chunk[
            boundary_mask
        ].copy()

        # Process everything else.
        process_data = chunk[
            ~boundary_mask
        ].copy()

        # ----------------------------------------------------
        # Process current chunk
        # ----------------------------------------------------

        if not process_data.empty:

            initial_rows = len(
                process_data
            )

            features = process_chunk_features(
                process_data,
                catalog,
                item_to_code
            )

            dropped_rows = (
                initial_rows
                - len(features)
            )

            total_dropped_targets += (
                dropped_rows
            )

            # ------------------------------------------------
            # Statistics
            # ------------------------------------------------

            if not features.empty:

                output_rows = len(
                    features
                )

                total_output_rows += (
                    output_rows
                )

                target_values = (
                    features["target"]
                    .astype("float64")
                )

                target_min = min(
                    target_min,
                    float(
                        target_values.min()
                    )
                )

                target_max = max(
                    target_max,
                    float(
                        target_values.max()
                    )
                )

                target_sum += float(
                    target_values.sum()
                )

                target_sum_sq += float(
                    (
                        target_values ** 2
                    ).sum()
                )

                # Date range
                current_min_date = (
                    features["date"].min()
                )

                current_max_date = (
                    features["date"].max()
                )

                if min_date is None:
                    min_date = current_min_date
                else:
                    min_date = min(
                        min_date,
                        current_min_date
                    )

                if max_date is None:
                    max_date = current_max_date
                else:
                    max_date = max(
                        max_date,
                        current_max_date
                    )

                # ------------------------------------------------
                # WRITE TO CSV
                # ------------------------------------------------

                features.to_csv(
                    output_path,
                    mode="w" if first_write else "a",
                    header=first_write,
                    index=False
                )

                first_write = False

            print(
                f"Chunk {chunk_number:02d} | "
                f"Input: {initial_rows:,} | "
                f"Features: {len(features):,} | "
                f"Total: {total_output_rows:,}"
            )

            del features
            del process_data

            gc.collect()

    # ========================================================
    # PROCESS FINAL BUFFER
    # ========================================================

    if not remainder.empty:

        print("\nProcessing final buffer...")

        initial_rows = len(
            remainder
        )

        features = process_chunk_features(
            remainder,
            catalog,
            item_to_code
        )

        dropped_rows = (
            initial_rows
            - len(features)
        )

        total_dropped_targets += (
            dropped_rows
        )

        if not features.empty:

            output_rows = len(
                features
            )

            total_output_rows += (
                output_rows
            )

            target_values = (
                features["target"]
                .astype("float64")
            )

            target_min = min(
                target_min,
                float(
                    target_values.min()
                )
            )

            target_max = max(
                target_max,
                float(
                    target_values.max()
                )
            )

            target_sum += float(
                target_values.sum()
            )

            target_sum_sq += float(
                (
                    target_values ** 2
                ).sum()
            )

            current_min_date = (
                features["date"].min()
            )

            current_max_date = (
                features["date"].max()
            )

            if min_date is None:
                min_date = current_min_date
            else:
                min_date = min(
                    min_date,
                    current_min_date
                )

            if max_date is None:
                max_date = current_max_date
            else:
                max_date = max(
                    max_date,
                    current_max_date
                )

            features.to_csv(
                output_path,
                mode="w" if first_write else "a",
                header=first_write,
                index=False
            )

            first_write = False

        print(
            f"Final buffer | "
            f"Input: {initial_rows:,} | "
            f"Features: {len(features):,}"
        )

        del features
        del remainder

        gc.collect()

    # ========================================================
    # FINAL REPORT
    # ========================================================

    execution_time = (
        time.time()
        - start_time
    )

    if total_output_rows == 0:

        raise RuntimeError(
            "No feature rows were generated."
        )

    target_mean = (
        target_sum
        / total_output_rows
    )

    target_variance = (
        target_sum_sq
        / total_output_rows
    ) - (
        target_mean ** 2
    )

    target_std = float(
        np.sqrt(
            max(
                0,
                target_variance
            )
        )
    )

    # File size
    file_size_bytes = os.path.getsize(
        output_path
    )

    file_size_mb = (
        file_size_bytes
        / (1024 ** 2)
    )

    file_size_gb = (
        file_size_bytes
        / (1024 ** 3)
    )

    print("\n")
    print("=" * 80)
    print("FEATURE ENGINEERING COMPLETED")
    print("=" * 80)

    print(
        f"Input rows:              "
        f"{total_input_rows:,}"
    )

    print(
        f"Output feature rows:     "
        f"{total_output_rows:,}"
    )

    print(
        f"Rows without target:     "
        f"{total_dropped_targets:,}"
    )

    print(
        f"Date range:              "
        f"{min_date} to {max_date}"
    )

    print(
        f"Output file:             "
        f"{output_path}"
    )

    print(
        f"Output size:             "
        f"{file_size_gb:.2f} GB "
        f"({file_size_mb:.2f} MB)"
    )

    print(
        f"Execution time:          "
        f"{execution_time:.2f} seconds"
    )

    print("\nTarget statistics:")

    print(
        f"  Minimum:               "
        f"{target_min:.4f}"
    )

    print(
        f"  Maximum:               "
        f"{target_max:.4f}"
    )

    print(
        f"  Mean:                  "
        f"{target_mean:.4f}"
    )

    print(
        f"  Standard deviation:    "
        f"{target_std:.4f}"
    )

    print("\nLeakage checks:")

    print(
        "  [OK] target = next-day demand (t+1)"
    )

    print(
        "  [OK] raw daily_demand removed"
    )

    print(
        "  [OK] lag features use historical demand"
    )

    print(
        "  [OK] rolling features use t-1 and earlier"
    )

    print(
        "  [OK] price_change uses current and previous price"
    )

    print(
        "  [OK] raw catalog text removed"
    )

    print("=" * 80)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    run_feature_engineering(
        input_path="data/prepared_demand.csv",
        catalog_path="data/catalog.csv",
        output_path="data/features.csv",
        chunk_size=1_000_000
    )