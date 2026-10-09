import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# CONFIGURATION
# ============================================================

FEATURES_FILE = Path("data/features.csv")
MODEL_DIR = Path("models")
MODEL_FILE = MODEL_DIR / "xgboost_demand_model.json"

CHUNK_SIZE = 500_000

# Chronological split
TRAIN_START = pd.Timestamp("2022-08-28")
TRAIN_END = pd.Timestamp("2024-03-31")

VALIDATION_START = pd.Timestamp("2024-04-01")
VALIDATION_END = pd.Timestamp("2024-06-30")

TEST_START = pd.Timestamp("2024-07-01")
TEST_END = pd.Timestamp("2024-09-25")


# These are the exact row counts verified in the previous step.
EXPECTED_TRAIN_ROWS = 14_114_747
EXPECTED_VALIDATION_ROWS = 3_133_253
EXPECTED_TEST_ROWS = 2_680_885


# ============================================================
# FEATURES
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


# ============================================================
# CATEGORICAL FEATURES
# ============================================================

CATEGORICAL_COLUMNS = [
    "store_id",
    "item_code",
    "dept_code",
    "class_code",
    "subclass_code",
    "item_type_code",
]


# XGBoost feature types:
# q = quantitative/numeric
# c = categorical
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
# BASIC CHECKS
# ============================================================

if not FEATURES_FILE.exists():
    raise FileNotFoundError(
        f"Could not find: {FEATURES_FILE}"
    )

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(chunk):
    """
    Prepare one CSV chunk for XGBoost.

    Important:
    Category codes created during feature engineering are already
    compact integer codes. We shift them by +1 so that:
        original -1 / missing -> 0
        original 0            -> 1
        original 1            -> 2
        ...

    This gives XGBoost non-negative categorical codes.
    """

    X = chunk[FEATURE_COLUMNS].copy()

    # -------------------------------
    # Numeric columns
    # -------------------------------

    numeric_columns = [
        col
        for col in FEATURE_COLUMNS
        if col not in CATEGORICAL_COLUMNS
    ]

    for col in numeric_columns:

        X[col] = pd.to_numeric(
            X[col],
            errors="coerce"
        ).astype(np.float32)

    # -------------------------------
    # Categorical columns
    # -------------------------------

    for col in CATEGORICAL_COLUMNS:

        values = pd.to_numeric(
            X[col],
            errors="coerce"
        )

        # Check for unexpected negative values
        negative_values = values.dropna()

        if len(negative_values) > 0:
            if (negative_values < -1).any():
                raise ValueError(
                    f"Unexpected value below -1 found in "
                    f"categorical column: {col}"
                )

        # Missing/-1 becomes 0.
        # Normal code 0 becomes 1, etc.
        values = (
            values
            .fillna(-1)
            .astype(np.int32)
            + 1
        )

        X[col] = values

    return X


# ============================================================
# XGBOOST DATA ITERATOR
# ============================================================

class CSVDateIterator(xgb.DataIter):

    def __init__(
        self,
        csv_path,
        start_date,
        end_date,
        chunk_size,
        cache_prefix
    ):

        self.csv_path = str(csv_path)
        self.start_date = start_date
        self.end_date = end_date
        self.chunk_size = chunk_size

        self.reader = None
        self.batch_number = 0

        super().__init__(
            cache_prefix=cache_prefix
        )

    def reset(self):

        if self.reader is not None:
            self.reader.close()

        self.reader = pd.read_csv(
            self.csv_path,
            chunksize=self.chunk_size,
            low_memory=False
        )

        self.batch_number = 0

    def next(self, input_data):

        if self.reader is None:
            self.reset()

        while True:

            try:
                chunk = next(self.reader)

            except StopIteration:

                if self.reader is not None:
                    self.reader.close()

                return False

            chunk["date"] = pd.to_datetime(
                chunk["date"],
                errors="coerce"
            )

            mask = (
                (chunk["date"] >= self.start_date)
                &
                (chunk["date"] <= self.end_date)
            )

            chunk = chunk.loc[mask]

            if chunk.empty:
                continue

            X = prepare_features(chunk)

            y = pd.to_numeric(
                chunk[TARGET_COLUMN],
                errors="coerce"
            ).astype(np.float32)

            input_data(
                data=X,
                label=y,
                feature_names=FEATURE_COLUMNS,
                feature_types=FEATURE_TYPES
            )

            self.batch_number += 1

            return True


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model(
    model,
    csv_path,
    start_date,
    end_date,
    label
):

    actual_parts = []
    prediction_parts = []

    total_rows = 0

    reader = pd.read_csv(
        csv_path,
        chunksize=CHUNK_SIZE,
        low_memory=False
    )

    batch_number = 0

    print("\n")
    print("=" * 70)
    print(f"EVALUATING {label.upper()}")
    print("=" * 70)

    for chunk in reader:

        chunk["date"] = pd.to_datetime(
            chunk["date"],
            errors="coerce"
        )

        mask = (
            (chunk["date"] >= start_date)
            &
            (chunk["date"] <= end_date)
        )

        chunk = chunk.loc[mask]

        if chunk.empty:
            continue

        X = prepare_features(chunk)

        y = pd.to_numeric(
            chunk[TARGET_COLUMN],
            errors="coerce"
        ).astype(np.float32)

        # Build a small DMatrix only for this chunk.
        matrix = xgb.DMatrix(
            X,
            label=y,
            feature_names=FEATURE_COLUMNS,
            feature_types=FEATURE_TYPES,
            enable_categorical=True
        )

        predictions = model.predict(matrix)

        actual_parts.append(
            y.to_numpy()
        )

        prediction_parts.append(
            predictions
        )

        total_rows += len(chunk)
        batch_number += 1

        if batch_number % 5 == 0:
            print(
                f"Evaluation batch {batch_number}"
                f" | Rows processed: {total_rows:,}"
            )

    if total_rows == 0:
        raise RuntimeError(
            f"No rows found for {label} period."
        )

    actual = np.concatenate(actual_parts)
    predicted = np.concatenate(prediction_parts)

    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    r2 = r2_score(
        actual,
        predicted
    )

    total_actual = np.sum(
        np.abs(actual)
    )

    if total_actual > 0:

        wape = (
            np.sum(
                np.abs(actual - predicted)
            )
            / total_actual
        ) * 100

    else:

        wape = np.nan

    return {
        "rows": total_rows,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "wape": wape,
        "actual": actual,
        "predicted": predicted,
    }


# ============================================================
# START
# ============================================================

start_time = time.time()

print("=" * 70)
print("XGBOOST DEMAND FORECASTING - MODEL TRAINING")
print("=" * 70)

print("\nXGBoost version:")
print(f"  {xgb.__version__}")

print("\nInput dataset:")
print(f"  {FEATURES_FILE}")

file_size_gb = (
    FEATURES_FILE.stat().st_size
    / (1024 ** 3)
)

print(f"  Size: {file_size_gb:.2f} GB")

print("\nChronological periods:")
print(
    f"  TRAIN      : "
    f"{TRAIN_START.date()} → {TRAIN_END.date()}"
)

print(
    f"  VALIDATION : "
    f"{VALIDATION_START.date()} → {VALIDATION_END.date()}"
)

print(
    f"  TEST       : "
    f"{TEST_START.date()} → {TEST_END.date()}"
)

print("\nVerified row counts:")
print(
    f"  TRAIN      : "
    f"{EXPECTED_TRAIN_ROWS:,}"
)

print(
    f"  VALIDATION : "
    f"{EXPECTED_VALIDATION_ROWS:,}"
)

print(
    f"  TEST       : "
    f"{EXPECTED_TEST_ROWS:,}"
)

print(
    f"  TOTAL      : "
    f"{EXPECTED_TRAIN_ROWS + EXPECTED_VALIDATION_ROWS + EXPECTED_TEST_ROWS:,}"
)

print("\nModel features:")
print(f"  {len(FEATURE_COLUMNS)} features")

print("\nTarget:")
print(f"  {TARGET_COLUMN}")

print("\nCategorical features:")
for col in CATEGORICAL_COLUMNS:
    print(f"  - {col}")


# ============================================================
# TEMPORARY CACHE DIRECTORY
# ============================================================

cache_dir = Path(
    tempfile.mkdtemp(
        prefix="xgb_cache_",
        dir="data"
    )
)

print("\nTemporary XGBoost cache:")
print(f"  {cache_dir}")


try:

    # ========================================================
    # CREATE TRAINING ITERATOR
    # ========================================================

    print("\n")
    print("=" * 70)
    print("BUILDING TRAINING MATRIX")
    print("=" * 70)

    train_iterator = CSVDateIterator(
        FEATURES_FILE,
        TRAIN_START,
        TRAIN_END,
        CHUNK_SIZE,
        str(cache_dir / "train")
    )

    train_matrix = xgb.ExtMemQuantileDMatrix(
        train_iterator,
        max_bin=256,
        enable_categorical=True,
        nthread=max(
            (os.cpu_count() or 4) - 1,
            1
        )
    )

    print("\nTraining matrix ready.")


    # ========================================================
    # CREATE VALIDATION ITERATOR
    # ========================================================

    print("\n")
    print("=" * 70)
    print("BUILDING VALIDATION MATRIX")
    print("=" * 70)

    validation_iterator = CSVDateIterator(
        FEATURES_FILE,
        VALIDATION_START,
        VALIDATION_END,
        CHUNK_SIZE,
        str(cache_dir / "validation")
    )

    validation_matrix = xgb.ExtMemQuantileDMatrix(
        validation_iterator,
        max_bin=256,
        ref=train_matrix,
        enable_categorical=True,
        nthread=max(
            (os.cpu_count() or 4) - 1,
            1
        )
    )

    print("\nValidation matrix ready.")


    # ========================================================
    # XGBOOST PARAMETERS
    # ========================================================

    params = {

        # Regression because target is next-day demand.
        "objective": "reg:squarederror",

        # Validation metric monitored during training.
        "eval_metric": "rmse",

        # Efficient tree algorithm for large datasets.
        "tree_method": "hist",

        # Tree complexity.
        "max_depth": 8,

        # Learning rate.
        "learning_rate": 0.05,

        # Minimum child weight.
        "min_child_weight": 5,

        # Row subsampling.
        "subsample": 0.8,

        # Feature subsampling.
        "colsample_bytree": 0.8,

        # L1 regularization.
        "reg_alpha": 0.0,

        # L2 regularization.
        "reg_lambda": 1.0,

        # Histogram resolution.
        "max_bin": 256,

        # Native categorical support.
        "enable_categorical": True,

        # Number of categories considered
        # for partition-based categorical splits.
        "max_cat_threshold": 64,

        # Threads.
        "nthread": max(
            (os.cpu_count() or 4) - 1,
            1
        ),
    }


    # ========================================================
    # PRINT PARAMETERS
    # ========================================================

    print("\n")
    print("=" * 70)
    print("XGBOOST PARAMETERS")
    print("=" * 70)

    for key, value in params.items():
        print(f"{key:25s}: {value}")


    # ========================================================
    # TRAINING
    # ========================================================

    print("\n")
    print("=" * 70)
    print("XGBOOST TRAINING STARTED")
    print("=" * 70)

    print(
        "\nThe model will train for up to 200 boosting rounds."
    )

    print(
        "Early stopping will stop training when validation "
        "RMSE stops improving."
    )

    model = xgb.train(
        params=params,
        dtrain=train_matrix,
        num_boost_round=200,
        evals=[
            (train_matrix, "train"),
            (validation_matrix, "validation")
        ],
        callbacks=[
            xgb.callback.EarlyStopping(
                rounds=20,
                metric_name="rmse",
                data_name="validation",
                save_best=True
            )
        ],
        verbose_eval=10
    )


    # ========================================================
    # TRAINING SUMMARY
    # ========================================================

    print("\n")
    print("=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)

    print(
        f"\nBest iteration: "
        f"{model.best_iteration}"
    )

    print(
        f"Best validation RMSE: "
        f"{model.best_score}"
    )


    # ========================================================
    # SAVE MODEL
    # ========================================================

    print("\n")
    print("=" * 70)
    print("SAVING MODEL")
    print("=" * 70)

    model.save_model(
        MODEL_FILE
    )

    print(
        f"\nModel saved successfully:"
    )

    print(
        f"  {MODEL_FILE}"
    )


    # ========================================================
    # VALIDATION METRICS
    # ========================================================

    validation_results = evaluate_model(
        model,
        FEATURES_FILE,
        VALIDATION_START,
        VALIDATION_END,
        "validation"
    )


    print("\n")
    print("=" * 70)
    print("VALIDATION PERFORMANCE")
    print("=" * 70)

    print(
        f"\nRows evaluated : "
        f"{validation_results['rows']:,}"
    )

    print(
        f"MAE            : "
        f"{validation_results['mae']:.4f}"
    )

    print(
        f"RMSE           : "
        f"{validation_results['rmse']:.4f}"
    )

    print(
        f"R²             : "
        f"{validation_results['r2']:.4f}"
    )

    print(
        f"WAPE           : "
        f"{validation_results['wape']:.2f}%"
    )


    # ========================================================
    # FINAL TEST
    # ========================================================

    print("\n")
    print("=" * 70)
    print("FINAL TEST EVALUATION")
    print("=" * 70)

    print(
        "\nThe test period was not used to train the model."
    )

    print(
        "Evaluating on the unseen future period..."
    )

    test_results = evaluate_model(
        model,
        FEATURES_FILE,
        TEST_START,
        TEST_END,
        "test"
    )


    # ========================================================
    # TEST METRICS
    # ========================================================

    print("\n")
    print("=" * 70)
    print("FINAL TEST PERFORMANCE")
    print("=" * 70)

    print(
        f"\nRows evaluated : "
        f"{test_results['rows']:,}"
    )

    print(
        f"MAE            : "
        f"{test_results['mae']:.4f}"
    )

    print(
        f"RMSE           : "
        f"{test_results['rmse']:.4f}"
    )

    print(
        f"R²             : "
        f"{test_results['r2']:.4f}"
    )

    print(
        f"WAPE           : "
        f"{test_results['wape']:.2f}%"
    )


    # ========================================================
    # ACTUAL VS PREDICTED
    # ========================================================

    print("\n")
    print("=" * 70)
    print("SAMPLE ACTUAL VS PREDICTED")
    print("=" * 70)

    actual = test_results["actual"]
    predicted = test_results["predicted"]

    sample_count = min(
        20,
        len(actual)
    )

    print(
        f"\n{'Actual':>12}"
        f"{'Predicted':>15}"
        f"{'Absolute Error':>18}"
    )

    print("-" * 50)

    for i in range(sample_count):

        actual_value = actual[i]
        predicted_value = predicted[i]

        error = abs(
            actual_value - predicted_value
        )

        print(
            f"{actual_value:>12.2f}"
            f"{predicted_value:>15.2f}"
            f"{error:>18.2f}"
        )


    # ========================================================
    # FINAL MODEL REPORT
    # ========================================================

    elapsed = time.time() - start_time

    print("\n")
    print("=" * 70)
    print("XGBOOST MODEL TRAINING COMPLETED")
    print("=" * 70)

    print("\nDATA")
    print("-" * 70)
    print(
        f"Total feature rows : "
        f"{EXPECTED_TRAIN_ROWS + EXPECTED_VALIDATION_ROWS + EXPECTED_TEST_ROWS:,}"
    )

    print(
        f"Training rows      : "
        f"{EXPECTED_TRAIN_ROWS:,}"
    )

    print(
        f"Validation rows    : "
        f"{EXPECTED_VALIDATION_ROWS:,}"
    )

    print(
        f"Test rows          : "
        f"{EXPECTED_TEST_ROWS:,}"
    )

    print("\nMODEL")
    print("-" * 70)
    print(
        f"Best iteration     : "
        f"{model.best_iteration}"
    )

    print(
        f"Best validation RMSE : "
        f"{model.best_score}"
    )

    print("\nFINAL TEST METRICS")
    print("-" * 70)

    print(
        f"MAE                : "
        f"{test_results['mae']:.4f}"
    )

    print(
        f"RMSE               : "
        f"{test_results['rmse']:.4f}"
    )

    print(
        f"R²                 : "
        f"{test_results['r2']:.4f}"
    )

    print(
        f"WAPE               : "
        f"{test_results['wape']:.2f}%"
    )

    print("\nMODEL FILE")
    print("-" * 70)
    print(
        f"{MODEL_FILE}"
    )

    print(
        f"\nExecution time: "
        f"{elapsed / 60:.2f} minutes"
    )

    print(
        "\nOriginal features.csv was NOT modified."
    )

    print(
        "No train/validation/test CSV files were created."
    )


finally:

    # ========================================================
    # CLEAN TEMPORARY CACHE
    # ========================================================

    print("\n")
    print("=" * 70)
    print("CLEANING TEMPORARY XGBOOST CACHE")
    print("=" * 70)

    try:

        shutil.rmtree(
            cache_dir,
            ignore_errors=True
        )

        print(
            "\nTemporary cache removed."
        )

    except Exception as cleanup_error:

        print(
            f"\nWarning: could not completely remove "
            f"temporary cache: {cleanup_error}"
        )