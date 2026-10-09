import pandas as pd
import time
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

FEATURES_FILE = Path("data/features.csv")

TRAIN_START = "2022-08-28"
TRAIN_END = "2024-03-31"

VALIDATION_START = "2024-04-01"
VALIDATION_END = "2024-06-30"

TEST_START = "2024-07-01"
TEST_END = "2024-09-25"

CHUNK_SIZE = 500_000


# ============================================================
# START
# ============================================================

start_time = time.time()

print("=" * 70)
print("XGBOOST CHRONOLOGICAL SPLIT INSPECTION")
print("=" * 70)

print(f"\nInput file: {FEATURES_FILE}")
print(f"Chunk size: {CHUNK_SIZE:,}")

if not FEATURES_FILE.exists():
    raise FileNotFoundError(
        f"\nERROR: Could not find {FEATURES_FILE}"
    )

print("\nReading features.csv chunk by chunk...")


# ============================================================
# DATE CONVERSION
# ============================================================

train_start = pd.Timestamp(TRAIN_START)
train_end = pd.Timestamp(TRAIN_END)

validation_start = pd.Timestamp(VALIDATION_START)
validation_end = pd.Timestamp(VALIDATION_END)

test_start = pd.Timestamp(TEST_START)
test_end = pd.Timestamp(TEST_END)


# ============================================================
# COUNTERS
# ============================================================

total_rows = 0

train_rows = 0
validation_rows = 0
test_rows = 0

outside_rows = 0

overall_min_date = None
overall_max_date = None

train_min_date = None
train_max_date = None

validation_min_date = None
validation_max_date = None

test_min_date = None
test_max_date = None


# ============================================================
# READ DATA IN CHUNKS
# ============================================================

chunk_number = 0

for chunk in pd.read_csv(
    FEATURES_FILE,
    chunksize=CHUNK_SIZE,
    low_memory=False
):

    chunk_number += 1

    # Convert date
    chunk["date"] = pd.to_datetime(
        chunk["date"],
        errors="coerce"
    )

    total_rows += len(chunk)

    # --------------------------------------------------------
    # Overall date range
    # --------------------------------------------------------

    chunk_min = chunk["date"].min()
    chunk_max = chunk["date"].max()

    if overall_min_date is None or chunk_min < overall_min_date:
        overall_min_date = chunk_min

    if overall_max_date is None or chunk_max > overall_max_date:
        overall_max_date = chunk_max

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    train_mask = (
        (chunk["date"] >= train_start)
        &
        (chunk["date"] <= train_end)
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    validation_mask = (
        (chunk["date"] >= validation_start)
        &
        (chunk["date"] <= validation_end)
    )

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    test_mask = (
        (chunk["date"] >= test_start)
        &
        (chunk["date"] <= test_end)
    )

    # --------------------------------------------------------
    # Count
    # --------------------------------------------------------

    current_train = train_mask.sum()
    current_validation = validation_mask.sum()
    current_test = test_mask.sum()

    train_rows += current_train
    validation_rows += current_validation
    test_rows += current_test

    # Rows that don't belong to any period
    assigned = (
        train_mask
        | validation_mask
        | test_mask
    )

    outside_rows += (~assigned).sum()

    # --------------------------------------------------------
    # Train date range
    # --------------------------------------------------------

    if current_train > 0:

        dates = chunk.loc[train_mask, "date"]

        current_min = dates.min()
        current_max = dates.max()

        if train_min_date is None or current_min < train_min_date:
            train_min_date = current_min

        if train_max_date is None or current_max > train_max_date:
            train_max_date = current_max

    # --------------------------------------------------------
    # Validation date range
    # --------------------------------------------------------

    if current_validation > 0:

        dates = chunk.loc[
            validation_mask,
            "date"
        ]

        current_min = dates.min()
        current_max = dates.max()

        if (
            validation_min_date is None
            or current_min < validation_min_date
        ):
            validation_min_date = current_min

        if (
            validation_max_date is None
            or current_max > validation_max_date
        ):
            validation_max_date = current_max

    # --------------------------------------------------------
    # Test date range
    # --------------------------------------------------------

    if current_test > 0:

        dates = chunk.loc[
            test_mask,
            "date"
        ]

        current_min = dates.min()
        current_max = dates.max()

        if test_min_date is None or current_min < test_min_date:
            test_min_date = current_min

        if test_max_date is None or current_max > test_max_date:
            test_max_date = current_max

    print(
        f"Chunk {chunk_number:02d} | "
        f"Rows: {len(chunk):,} | "
        f"Train: {current_train:,} | "
        f"Validation: {current_validation:,} | "
        f"Test: {current_test:,}"
    )


# ============================================================
# RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("CHRONOLOGICAL SPLIT RESULTS")
print("=" * 70)

print(f"\nTotal rows:       {total_rows:,}")

print("\nTRAIN")
print("-" * 70)
print(f"Requested range:  {TRAIN_START} → {TRAIN_END}")
print(f"Actual range:     {train_min_date.date()} → {train_max_date.date()}")
print(f"Rows:             {train_rows:,}")
print(f"Percentage:       {(train_rows / total_rows) * 100:.2f}%")

print("\nVALIDATION")
print("-" * 70)
print(f"Requested range:  {VALIDATION_START} → {VALIDATION_END}")
print(
    f"Actual range:     "
    f"{validation_min_date.date()} → {validation_max_date.date()}"
)
print(f"Rows:             {validation_rows:,}")
print(f"Percentage:       {(validation_rows / total_rows) * 100:.2f}%")

print("\nTEST")
print("-" * 70)
print(f"Requested range:  {TEST_START} → {TEST_END}")
print(f"Actual range:     {test_min_date.date()} → {test_max_date.date()}")
print(f"Rows:             {test_rows:,}")
print(f"Percentage:       {(test_rows / total_rows) * 100:.2f}%")


# ============================================================
# VALIDATION CHECKS
# ============================================================

print("\n")
print("=" * 70)
print("VALIDATION CHECKS")
print("=" * 70)

assigned_rows = (
    train_rows
    + validation_rows
    + test_rows
)

print(f"\nAssigned rows:    {assigned_rows:,}")
print(f"Original rows:    {total_rows:,}")
print(f"Outside periods:  {outside_rows:,}")

# Check 1: all rows accounted for
if assigned_rows + outside_rows == total_rows:
    print("\n✓ Row accounting check PASSED")
else:
    print("\n✗ Row accounting check FAILED")

# Check 2: chronological order
if train_end < validation_start < validation_end < test_start:
    print("✓ Chronological ordering check PASSED")
else:
    print("✗ Chronological ordering check FAILED")

# Check 3: no overlap
if train_end < validation_start and validation_end < test_start:
    print("✓ No-overlap check PASSED")
else:
    print("✗ No-overlap check FAILED")

# Check 4: expected total
if total_rows == 19_928_885:
    print("✓ Expected feature-row count PASSED")
else:
    print(
        f"⚠ Expected 19,928,885 rows, "
        f"but found {total_rows:,}"
    )


# ============================================================
# STORAGE CHECK
# ============================================================

print("\n")
print("=" * 70)
print("STORAGE")
print("=" * 70)

file_size_gb = FEATURES_FILE.stat().st_size / (1024 ** 3)

print(f"\nExisting features.csv: {file_size_gb:.2f} GB")
print("Additional split files: 0")
print("Additional storage required: ~0 GB")


# ============================================================
# FINAL
# ============================================================

elapsed = time.time() - start_time

print("\n")
print("=" * 70)
print("INSPECTION COMPLETED")
print("=" * 70)

print(f"\nExecution time: {elapsed:.2f} seconds")

print("\nNo train/validation/test CSV files were created.")
print("The original features.csv remains unchanged.")
print("\nReady for XGBoost training.")