"""
XGBoost Demand Forecasting - Error Analysis
=============================================

Loads the saved XGBoost model and evaluates predictions
on the test period (2024-07-01 to 2024-09-25).

Produces:
    1. Overall metrics (MAE, RMSE, R2, WAPE)
    2. Error distribution statistics
    3. Actual vs predicted demand distributions
    4. Performance by demand range
    5. High-demand underprediction analysis
    6. Store-level error breakdown
    7. Largest absolute prediction errors
    8. Mean actual vs mean predicted demand
    9. Prediction bias
   10. RMSE contribution by observation type

Input:
    - models/xgboost_demand_model.json
    - data/features.csv (read in chunks)

Output:
    - Console report only. No large files created.

Does NOT retrain or modify the model.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.stdout.reconfigure(encoding="utf-8")


# ============================================================
# CONFIGURATION
# ============================================================

FEATURES_FILE = Path("data/features.csv")
MODEL_FILE = Path("models/xgboost_demand_model.json")

CHUNK_SIZE = 500_000

TEST_START = pd.Timestamp("2024-07-01")
TEST_END = pd.Timestamp("2024-09-25")


# ============================================================
# FEATURES (must match training script exactly)
# ============================================================

FEATURE_COLUMNS = [
    "price",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_28",
    "rolling_std_7",
    "rolling_std_28",
    "price_change",
    "day_of_week",
    "day_of_month",
    "month",
    "week_of_year",
    "is_weekend",
    "store_id",
    "item_code",
    "dept_code",
    "class_code",
    "subclass_code",
    "item_type_code",
    "weight_volume",
    "weight_netto",
    "fatness",
]

TARGET_COLUMN = "target"

CATEGORICAL_COLUMNS = [
    "store_id",
    "item_code",
    "dept_code",
    "class_code",
    "subclass_code",
    "item_type_code",
]

FEATURE_TYPES = [
    "q",  # price
    "q",  # lag_1
    "q",  # lag_7
    "q",  # lag_14
    "q",  # lag_28
    "q",  # rolling_mean_7
    "q",  # rolling_mean_14
    "q",  # rolling_mean_28
    "q",  # rolling_std_7
    "q",  # rolling_std_28
    "q",  # price_change
    "q",  # day_of_week
    "q",  # day_of_month
    "q",  # month
    "q",  # week_of_year
    "q",  # is_weekend
    "c",  # store_id
    "c",  # item_code
    "c",  # dept_code
    "c",  # class_code
    "c",  # subclass_code
    "c",  # item_type_code
    "q",  # weight_volume
    "q",  # weight_netto
    "q",  # fatness
]


# ============================================================
# PREPARE FEATURES (identical to training script)
# ============================================================

def prepare_features(chunk):
    """
    Prepare one CSV chunk for XGBoost prediction.

    Replicates the exact same transformation used during
    training (XGboost_training.py, prepare_features).
    """

    X = chunk[FEATURE_COLUMNS].copy()

    # Numeric columns
    numeric_columns = [
        col for col in FEATURE_COLUMNS
        if col not in CATEGORICAL_COLUMNS
    ]

    for col in numeric_columns:
        X[col] = pd.to_numeric(
            X[col], errors="coerce"
        ).astype(np.float32)

    # Categorical columns: shift by +1 so -1/missing -> 0
    for col in CATEGORICAL_COLUMNS:

        values = pd.to_numeric(
            X[col], errors="coerce"
        )

        values = (
            values
            .fillna(-1)
            .astype(np.int32)
            + 1
        )

        X[col] = values

    return X


# ============================================================
# CHECKS
# ============================================================

if not FEATURES_FILE.exists():
    raise FileNotFoundError(f"Missing: {FEATURES_FILE}")

if not MODEL_FILE.exists():
    raise FileNotFoundError(f"Missing: {MODEL_FILE}")


# ============================================================
# LOAD MODEL
# ============================================================

start_time = time.time()

print("=" * 70)
print("XGBOOST ERROR ANALYSIS")
print("=" * 70)

print(f"\nModel file : {MODEL_FILE}")
print(f"Data file  : {FEATURES_FILE}")
print(f"Test period: {TEST_START.date()} to {TEST_END.date()}")
print(f"Chunk size : {CHUNK_SIZE:,}")

model = xgb.Booster()
model.load_model(str(MODEL_FILE))

print("\nModel loaded successfully.")


# ============================================================
# PASS 1: COLLECT TEST PREDICTIONS
# ============================================================

print("\n")
print("=" * 70)
print("PASS 1: GENERATING TEST PREDICTIONS")
print("=" * 70)

# We collect lightweight arrays only:
#   actual, predicted, store_id (original), date
# This is ~2.68M rows x 4 columns, ~80 MB in memory.

actual_parts = []
predicted_parts = []
store_parts = []
date_parts = []

total_test_rows = 0
chunk_number = 0

for chunk in pd.read_csv(
    FEATURES_FILE,
    chunksize=CHUNK_SIZE,
    low_memory=False
):

    chunk["date"] = pd.to_datetime(
        chunk["date"], errors="coerce"
    )

    mask = (
        (chunk["date"] >= TEST_START)
        & (chunk["date"] <= TEST_END)
    )

    chunk = chunk.loc[mask]

    if chunk.empty:
        chunk_number += 1
        continue

    X = prepare_features(chunk)

    y = pd.to_numeric(
        chunk[TARGET_COLUMN], errors="coerce"
    ).astype(np.float32)

    matrix = xgb.DMatrix(
        X,
        feature_names=FEATURE_COLUMNS,
        feature_types=FEATURE_TYPES,
        enable_categorical=True
    )

    preds = model.predict(matrix)

    actual_parts.append(y.values)
    predicted_parts.append(preds)
    store_parts.append(chunk["store_id"].values.copy())
    date_parts.append(chunk["date"].values.copy())

    total_test_rows += len(chunk)
    chunk_number += 1

    if chunk_number % 5 == 0:
        print(
            f"  Chunk {chunk_number:02d} | "
            f"Test rows so far: {total_test_rows:,}"
        )

print(f"\n  Total test rows: {total_test_rows:,}")


# ============================================================
# CONSOLIDATE
# ============================================================

actual = np.concatenate(actual_parts)
predicted = np.concatenate(predicted_parts)
stores = np.concatenate(store_parts)
dates = np.concatenate(date_parts)

errors = predicted - actual
abs_errors = np.abs(errors)
squared_errors = errors ** 2

# Free the parts
del actual_parts, predicted_parts, store_parts, date_parts


# ============================================================
# 1. OVERALL METRICS
# ============================================================

print("\n")
print("=" * 70)
print("1. OVERALL TEST METRICS")
print("=" * 70)

mae = mean_absolute_error(actual, predicted)
rmse = np.sqrt(mean_squared_error(actual, predicted))
r2 = r2_score(actual, predicted)

total_actual = np.sum(np.abs(actual))
wape = (np.sum(abs_errors) / total_actual) * 100 if total_actual > 0 else np.nan

print(f"\n  Rows evaluated : {total_test_rows:,}")
print(f"  MAE            : {mae:.4f}")
print(f"  RMSE           : {rmse:.4f}")
print(f"  R2             : {r2:.4f}")
print(f"  WAPE           : {wape:.2f}%")


# ============================================================
# 2. ERROR DISTRIBUTION
# ============================================================

print("\n")
print("=" * 70)
print("2. ERROR DISTRIBUTION (predicted - actual)")
print("=" * 70)

print(f"\n  Count            : {len(errors):,}")
print(f"  Mean error       : {np.mean(errors):.4f}")
print(f"  Median error     : {np.median(errors):.4f}")
print(f"  Std dev          : {np.std(errors):.4f}")
print(f"  Min error        : {np.min(errors):.4f}")
print(f"  Max error        : {np.max(errors):.4f}")

print(f"\n  Absolute error:")
print(f"    Mean           : {np.mean(abs_errors):.4f}")
print(f"    Median         : {np.median(abs_errors):.4f}")
print(f"    90th percentile: {np.percentile(abs_errors, 90):.4f}")
print(f"    95th percentile: {np.percentile(abs_errors, 95):.4f}")
print(f"    99th percentile: {np.percentile(abs_errors, 99):.4f}")
print(f"    Max            : {np.max(abs_errors):.4f}")

# Signed error percentiles
print(f"\n  Signed error percentiles:")
for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
    val = np.percentile(errors, p)
    print(f"    P{p:<3d}           : {val:.4f}")

# Over/under prediction counts
overpredict = np.sum(errors > 0)
underpredict = np.sum(errors < 0)
exact = np.sum(errors == 0)

print(f"\n  Overpredictions  : {overpredict:,} ({overpredict/len(errors)*100:.2f}%)")
print(f"  Underpredictions : {underpredict:,} ({underpredict/len(errors)*100:.2f}%)")
print(f"  Exact matches    : {exact:,} ({exact/len(errors)*100:.2f}%)")


# ============================================================
# 3. ACTUAL VS PREDICTED DEMAND DISTRIBUTIONS
# ============================================================

print("\n")
print("=" * 70)
print("3. ACTUAL VS PREDICTED DEMAND DISTRIBUTIONS")
print("=" * 70)

print(f"\n  {'Statistic':<20} {'Actual':>12} {'Predicted':>12}")
print(f"  {'-'*20} {'-'*12} {'-'*12}")

stats = [
    ("Mean", np.mean, actual, predicted),
    ("Median", np.median, actual, predicted),
    ("Std Dev", np.std, actual, predicted),
    ("Min", np.min, actual, predicted),
    ("Max", np.max, actual, predicted),
]

for label, func, a, p in stats:
    print(f"  {label:<20} {func(a):>12.4f} {func(p):>12.4f}")

# Percentile comparison
print(f"\n  {'Percentile':<20} {'Actual':>12} {'Predicted':>12}")
print(f"  {'-'*20} {'-'*12} {'-'*12}")

for p_val in [10, 25, 50, 75, 90, 95, 99]:
    a_p = np.percentile(actual, p_val)
    p_p = np.percentile(predicted, p_val)
    print(f"  P{p_val:<18d} {a_p:>12.4f} {p_p:>12.4f}")

# Zero-demand analysis
actual_zeros = np.sum(actual == 0)
pred_near_zero = np.sum(np.abs(predicted) < 0.5)

print(f"\n  Actual zeros     : {actual_zeros:,} ({actual_zeros/len(actual)*100:.2f}%)")
print(f"  Predicted < 0.5  : {pred_near_zero:,} ({pred_near_zero/len(predicted)*100:.2f}%)")
print(f"  Negative preds   : {np.sum(predicted < 0):,}")


# ============================================================
# 4. PERFORMANCE BY DEMAND RANGE
# ============================================================

print("\n")
print("=" * 70)
print("4. PERFORMANCE BY DEMAND RANGE")
print("=" * 70)

demand_ranges = [
    ("demand = 0", actual == 0),
    ("demand 1-5", (actual >= 1) & (actual <= 5)),
    ("demand 6-20", (actual >= 6) & (actual <= 20)),
    ("demand 21-100", (actual >= 21) & (actual <= 100)),
    ("demand > 100", actual > 100),
]

print(
    f"\n  {'Range':<16} {'Count':>10} {'%':>7} "
    f"{'MAE':>8} {'RMSE':>8} {'MeanAct':>8} {'MeanPred':>9} {'Bias':>8}"
)
print(f"  {'-'*16} {'-'*10} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*9} {'-'*8}")

for label, mask in demand_ranges:

    count = mask.sum()

    if count == 0:
        print(f"  {label:<16} {0:>10} {0:>7.2f}%")
        continue

    pct = count / len(actual) * 100
    a = actual[mask]
    p = predicted[mask]
    e = errors[mask]

    range_mae = np.mean(np.abs(e))
    range_rmse = np.sqrt(np.mean(e ** 2))
    mean_a = np.mean(a)
    mean_p = np.mean(p)
    bias = np.mean(e)

    print(
        f"  {label:<16} {count:>10,} {pct:>6.2f}% "
        f"{range_mae:>8.2f} {range_rmse:>8.2f} "
        f"{mean_a:>8.2f} {mean_p:>9.2f} {bias:>8.2f}"
    )


# ============================================================
# 5. HIGH-DEMAND UNDERPREDICTION ANALYSIS
# ============================================================

print("\n")
print("=" * 70)
print("5. HIGH-DEMAND UNDERPREDICTION ANALYSIS")
print("=" * 70)

high_demand_mask = actual > 20

high_actual = actual[high_demand_mask]
high_predicted = predicted[high_demand_mask]
high_errors = errors[high_demand_mask]

print(f"\n  High-demand observations (actual > 20): {high_demand_mask.sum():,}")

if high_demand_mask.sum() > 0:

    high_underpredicted = np.sum(high_errors < 0)
    high_overpredicted = np.sum(high_errors > 0)

    print(
        f"  Underpredicted: {high_underpredicted:,} "
        f"({high_underpredicted/high_demand_mask.sum()*100:.1f}%)"
    )
    print(
        f"  Overpredicted : {high_overpredicted:,} "
        f"({high_overpredicted/high_demand_mask.sum()*100:.1f}%)"
    )

    print(f"\n  Mean actual   : {np.mean(high_actual):.2f}")
    print(f"  Mean predicted: {np.mean(high_predicted):.2f}")
    print(f"  Mean bias     : {np.mean(high_errors):.2f}")
    print(f"  MAE           : {np.mean(np.abs(high_errors)):.2f}")
    print(f"  RMSE          : {np.sqrt(np.mean(high_errors**2)):.2f}")

    # Ratio analysis: predicted/actual
    ratios = high_predicted / np.maximum(high_actual, 1e-6)
    print(f"\n  Prediction ratio (predicted/actual):")
    print(f"    Mean   : {np.mean(ratios):.4f}")
    print(f"    Median : {np.median(ratios):.4f}")
    print(f"    < 0.5  : {np.sum(ratios < 0.5):,} (severe underprediction)")
    print(f"    0.5-0.8: {np.sum((ratios >= 0.5) & (ratios < 0.8)):,}")
    print(f"    0.8-1.2: {np.sum((ratios >= 0.8) & (ratios <= 1.2)):,} (good)")
    print(f"    > 1.2  : {np.sum(ratios > 1.2):,}")


# ============================================================
# 6. ERRORS BY STORE
# ============================================================

print("\n")
print("=" * 70)
print("6. ERRORS BY STORE")
print("=" * 70)

unique_stores = np.unique(stores)

print(
    f"\n  {'Store':>6} {'Rows':>10} "
    f"{'MAE':>8} {'RMSE':>10} {'R2':>8} "
    f"{'WAPE':>8} {'Bias':>8} {'MeanAct':>8} {'MeanPred':>9}"
)
print(
    f"  {'-'*6} {'-'*10} "
    f"{'-'*8} {'-'*10} {'-'*8} "
    f"{'-'*8} {'-'*8} {'-'*8} {'-'*9}"
)

for store in sorted(unique_stores):

    store_mask = stores == store
    s_actual = actual[store_mask]
    s_predicted = predicted[store_mask]
    s_errors = errors[store_mask]

    s_count = store_mask.sum()
    s_mae = np.mean(np.abs(s_errors))
    s_rmse = np.sqrt(np.mean(s_errors ** 2))
    s_r2 = r2_score(s_actual, s_predicted) if len(s_actual) > 1 else np.nan

    s_total_actual = np.sum(np.abs(s_actual))
    s_wape = (
        np.sum(np.abs(s_errors)) / s_total_actual * 100
        if s_total_actual > 0
        else np.nan
    )

    s_bias = np.mean(s_errors)
    s_mean_act = np.mean(s_actual)
    s_mean_pred = np.mean(s_predicted)

    print(
        f"  {store:>6} {s_count:>10,} "
        f"{s_mae:>8.4f} {s_rmse:>10.4f} {s_r2:>8.4f} "
        f"{s_wape:>7.2f}% {s_bias:>8.4f} "
        f"{s_mean_act:>8.4f} {s_mean_pred:>9.4f}"
    )


# ============================================================
# 7. LARGEST ABSOLUTE PREDICTION ERRORS
# ============================================================

print("\n")
print("=" * 70)
print("7. TOP 30 LARGEST ABSOLUTE ERRORS")
print("=" * 70)

top_n = 30
top_indices = np.argsort(abs_errors)[-top_n:][::-1]

print(
    f"\n  {'Rank':>4} {'Date':>12} {'Store':>6} "
    f"{'Actual':>10} {'Predicted':>10} {'Error':>10} {'AbsError':>10}"
)
print(
    f"  {'-'*4} {'-'*12} {'-'*6} "
    f"{'-'*10} {'-'*10} {'-'*10} {'-'*10}"
)

for rank, idx in enumerate(top_indices, 1):

    date_val = pd.Timestamp(dates[idx])
    date_str = date_val.strftime("%Y-%m-%d")

    print(
        f"  {rank:>4} {date_str:>12} {stores[idx]:>6} "
        f"{actual[idx]:>10.2f} {predicted[idx]:>10.2f} "
        f"{errors[idx]:>10.2f} {abs_errors[idx]:>10.2f}"
    )


# ============================================================
# 8. MEAN ACTUAL VS MEAN PREDICTED
# ============================================================

print("\n")
print("=" * 70)
print("8. MEAN ACTUAL VS MEAN PREDICTED DEMAND")
print("=" * 70)

mean_actual = np.mean(actual)
mean_predicted = np.mean(predicted)

print(f"\n  Mean actual demand    : {mean_actual:.6f}")
print(f"  Mean predicted demand : {mean_predicted:.6f}")
print(f"  Difference            : {mean_predicted - mean_actual:.6f}")
print(f"  Ratio (pred/actual)   : {mean_predicted / mean_actual:.6f}" if mean_actual != 0 else "")

# Sum comparison
sum_actual = np.sum(actual)
sum_predicted = np.sum(predicted)

print(f"\n  Total actual demand   : {sum_actual:,.2f}")
print(f"  Total predicted demand: {sum_predicted:,.2f}")
print(f"  Total difference      : {sum_predicted - sum_actual:,.2f}")
print(
    f"  Total ratio           : {sum_predicted / sum_actual:.6f}"
    if sum_actual != 0 else ""
)


# ============================================================
# 9. PREDICTION BIAS
# ============================================================

print("\n")
print("=" * 70)
print("9. PREDICTION BIAS  mean(predicted - actual)")
print("=" * 70)

bias = np.mean(errors)
median_bias = np.median(errors)

print(f"\n  Mean bias   : {bias:.6f}")
print(f"  Median bias : {median_bias:.6f}")

if bias > 0:
    print(f"\n  Direction: Model OVERPREDICTS on average by {bias:.4f} units")
elif bias < 0:
    print(f"\n  Direction: Model UNDERPREDICTS on average by {abs(bias):.4f} units")
else:
    print(f"\n  Direction: No systematic bias")

# Bias by demand level
print(f"\n  Bias by demand level:")
print(f"  {'Range':<16} {'Bias':>10} {'Direction':<20}")
print(f"  {'-'*16} {'-'*10} {'-'*20}")

for label, mask in demand_ranges:

    if mask.sum() == 0:
        continue

    range_bias = np.mean(errors[mask])
    direction = "overpredicts" if range_bias > 0 else "underpredicts"

    print(f"  {label:<16} {range_bias:>10.4f} {direction:<20}")


# ============================================================
# 10. RMSE CONTRIBUTION BY OBSERVATION TYPE
# ============================================================

print("\n")
print("=" * 70)
print("10. RMSE CONTRIBUTION BY DEMAND RANGE")
print("=" * 70)

total_sse = np.sum(squared_errors)

print(
    f"\n  Total SSE: {total_sse:,.2f}"
)
print(
    f"  Total rows: {len(actual):,}"
)
print(
    f"  Overall RMSE: {rmse:.4f}"
)

print(
    f"\n  {'Range':<16} {'Count':>10} {'%Rows':>7} "
    f"{'SSE':>14} {'%SSE':>7} {'RMSE':>8} {'Comment'}"
)
print(
    f"  {'-'*16} {'-'*10} {'-'*7} "
    f"{'-'*14} {'-'*7} {'-'*8} {'-'*20}"
)

for label, mask in demand_ranges:

    count = mask.sum()

    if count == 0:
        continue

    pct_rows = count / len(actual) * 100
    range_sse = np.sum(squared_errors[mask])
    pct_sse = range_sse / total_sse * 100
    range_rmse = np.sqrt(np.mean(squared_errors[mask]))

    # Disproportionate if %SSE >> %Rows
    if pct_sse > pct_rows * 2:
        comment = "DISPROPORTIONATE"
    elif pct_sse < pct_rows * 0.5:
        comment = "low contribution"
    else:
        comment = "proportionate"

    print(
        f"  {label:<16} {count:>10,} {pct_rows:>6.2f}% "
        f"{range_sse:>14,.2f} {pct_sse:>6.2f}% "
        f"{range_rmse:>8.2f} {comment}"
    )


# ============================================================
# SUMMARY & RECOMMENDATIONS
# ============================================================

print("\n")
print("=" * 70)
print("SUMMARY & KEY FINDINGS")
print("=" * 70)

print(f"""
  Test Period      : {TEST_START.date()} to {TEST_END.date()}
  Test Rows        : {total_test_rows:,}

  MAE              : {mae:.4f}
  RMSE             : {rmse:.4f}
  R2               : {r2:.4f}
  WAPE             : {wape:.2f}%

  Mean Bias        : {bias:.4f}
  Mean Actual      : {mean_actual:.4f}
  Mean Predicted   : {mean_predicted:.4f}

  Overpredictions  : {overpredict:,} ({overpredict/len(errors)*100:.1f}%)
  Underpredictions : {underpredict:,} ({underpredict/len(errors)*100:.1f}%)
  Actual zeros     : {actual_zeros:,} ({actual_zeros/len(actual)*100:.1f}%)
""")


# ============================================================
# DONE
# ============================================================

elapsed = time.time() - start_time

print("=" * 70)
print(f"Error analysis completed in {elapsed:.1f} seconds "
      f"({elapsed/60:.1f} minutes)")
print("=" * 70)

print("\nNo files were created or modified.")
print("The model and features.csv remain unchanged.")

