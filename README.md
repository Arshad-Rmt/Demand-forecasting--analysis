# AI Market Intelligence & Demand Forecasting Platform

A machine learning platform for enterprise retail demand forecasting and inventory intelligence, built with Python, Pandas, and XGBoost. This project develops a next-day SKU-level demand forecasting engine trained on over 19.9 million historical observations across multi-store retail environments, providing automated predictions and operational insights.

---

## 1. Project Overview

In enterprise retail, inventory mismanagement is one of the largest sources of capital loss. Inaccurate demand projections lead to two costly outcomes:
- **Stockouts (Under-forecasting):** Missed sales revenue, degraded customer retention, and brand erosion.
- **Overstocking (Over-forecasting):** Capital lockup, storage bottlenecks, and inventory shrinkage from perishable degradation or obsolescence.

The **AI Market Intelligence & Demand Forecasting Platform** addresses this operational challenge by transforming raw point-of-sale (POS) transaction logs into an automated, supervised demand forecasting system. By predicting unit demand at the exact operational granularity required for store replenishment—**`(date, store_id, item_id)`**—the platform equips inventory planners and category managers with granular, data-driven purchasing recommendations.

---

## 2. Problem Statement and Objectives

### The Forecasting Challenge
Retail sales data presents specific structural difficulties that invalidate traditional time-series methods (e.g., ARIMA or exponential smoothing):
1. **High Dimensionality & Sparsity:** Over 28,000 distinct product SKUs sold across multiple store locations.
2. **Intermittent Demand:** Over 95% of active store-SKU time series exhibit intermittent sales (days with zero transactions between purchases).
3. **Complex Drivers:** Demand is governed by an interplay of calendar seasonality (day-of-week, month, weekend effects), price fluctuations, recent velocity trends, and store-level foot traffic.
4. **Cold-Start Introductions:** New products continually enter the assortment without historical transaction records.

### Primary Objectives
- **Data Engineering:** Ingest and preprocess over 7.4 million raw sales transactions, resolve data anomalies (negative prices, transaction returns), and construct continuous calendar time series.
- **Feature Pipeline:** Build a leakage-safe supervised feature engineering pipeline generating temporal lags, rolling statistics, calendar drivers, and product hierarchical codes.
- **Supervised Modeling:** Train an out-of-core, high-performance gradient-boosted decision tree baseline using XGBoost on a chronological train-validation-test split.
- **Diagnostic Rigor:** Perform granular error analysis, audit extreme prediction errors against raw POS data, and evaluate cold-start forecasting behavior on previously unseen products.

---

## 3. Technology Stack and Justification

| Technology | Role in Project | Engineering Justification |
| :--- | :--- | :--- |
| **Python 3.11** | Core Programming Language | Provides a robust scientific ecosystem, native 64-bit memory management, and compatibility with modern ML runtimes. |
| **Pandas & NumPy** | Data Wrangling & Feature Engineering | Utilized with chunked streaming (`chunksize=500_000` to `1_000_000`) and aggressive memory downcasting (`float32`, `int16`, `int8`). This allows processing 20M+ rows (~2.8 GB on disk) without memory exhaustion on consumer hardware. |
| **XGBoost (v3.2+)** | Primary Machine Learning Engine | Employs `xgb.Booster` with the histogram tree method (`tree_method="hist"`), native categorical support (`enable_categorical=True`), and out-of-core memory quantization (`ExtMemQuantileDMatrix`). Chosen for its ability to handle non-linear interactions, missing values, and tabular data without one-hot encoding explosions. |
| **Scikit-Learn** | Validation & Performance Metrics | Used for standard metric evaluations (`mean_absolute_error`, `mean_squared_error`, `r2_score`), providing validated implementations of regression metrics. |
| **Matplotlib** | Diagnostic Visualization | Configured with the headless backend (`Agg`) to generate automated data quality and aggregation validation plots directly to disk. |

---

## 4. Dataset and Forecasting Target

### Source Dataset Characteristics
The platform is built on real-world retail point-of-sale transaction logs:

- **Source File:** `data/sales.csv` (7,432,685 raw transaction records, ~361 MB)
- **Time Horizon:** August 28, 2022 to September 26, 2024 (761 calendar days)
- **Store Footprint:** 4 distinct retail store locations
- **Product Assortment:** 28,182 unique product SKUs across 58,027 active store-SKU combinations
- **Catalog Metadata:** `data/catalog.csv` covering product departments, classes, subclasses, item types, and physical attributes (net weight, volume, fatness)

### Key Columns in Raw POS Data
| Column | Type | Description |
| :--- | :--- | :--- |
| `date` | String / Date | Transaction date (`YYYY-MM-DD`) |
| `store_id` | Integer (1–4) | Identifier of the retail store branch |
| `item_id` | String (Hex) | Unique product SKU identifier (e.g., `00179dda14f8`) |
| `quantity` | Float / Numeric | Number of units purchased (or returned, if negative) |
| `price_base` | Float / Numeric | Baseline selling price per unit |
| `sum_total` | Float / Numeric | Total transaction value recorded at the register |

### The Forecasting Target
- **Target Variable:** Next-day demand ($t+1$), defined as the total unit quantity sold for a given product at a specific store on the following day:
  $$\text{target}_{s, i, t} = \text{daily\_demand}_{s, i, t+1}$$
- **Why Quantity Instead of Revenue:** Revenue fluctuates with price changes, discounts, and inflation. Inventory replenishment and supply chain logistics operate exclusively on physical unit quantities.
- **Why Next-Day ($t+1$):** Retail procurement decisions for fast-moving goods are executed daily for next-day dispatch.
- **Granularity:** `(date, store_id, item_id)`. Forecasting at this granular level ensures that predictions can be fed directly into automated reorder point formulas.

---

## 5. End-to-End Methodology

The platform implements an end-to-end data and modeling pipeline:

```
[ sales.csv ] (7.43M raw transactions)
      │
      ▼
1. Preprocessing (`src/preprocessing.py`)
   - Drop redundant index columns
   - Datetime conversion & negative quantity audit
   - Negative price imputation (Option A: store-item median)
      │
      ▼
2. Daily Demand Aggregation (`src/aggregation.py`)
   - Group by (date, store_id, item_id)
   - daily_demand = sum(quantity), price = mean(price_base)
   - Outputs: `data/daily_demand.csv` (7.43M rows)
      │
      ▼
3. Demand Preparation (`src/demand_preparation.py`)
   - Clip negative demand (returns) to 0
   - Continuous date zero-filling within [T_first, T_last]
   - Forward-fill prices across missing days
   - Outputs: `data/prepared_demand.csv` (19.98M rows)
      │
      ▼
4. Supervised Feature Engineering (`src/feature_engineering.py`)
   - Construct next-day target: shift(-1)
   - Historical lags: t-1, t-7, t-14, t-28
   - Rolling stats (mean & std): 7d, 14d, 28d
   - Calendar seasonality & price dynamics
   - Merge compact catalog codes
   - Outputs: `data/features.csv` (19.92M rows, 28 columns)
      │
      ▼
5. Chronological Splitting (`src/XGBoost.py`)
   - Train: 2022-08-28 to 2024-03-31 (14.11M rows, 70.8%)
   - Validation: 2024-04-01 to 2024-06-30 (3.13M rows, 15.7%)
   - Test: 2024-07-01 to 2024-09-25 (2.68M rows, 13.5%)
      │
      ▼
6. Model Training (`src/XGboost_training.py`)
   - Out-of-core ExtMemQuantileDMatrix
   - Histogram-based gradient boosting
   - Early stopping on validation RMSE
   - Saved Model: `models/xgboost_demand_model.json`
      │
      ▼
7. Comprehensive Diagnostics (`src/xgboost_error_analysis.py`, etc.)
   - Evaluation on unseen test period
   - Error tier decomposition & bias auditing
   - Extreme error root-cause investigation
   - Cold-start cohort evaluation
```

---

## 6. Feature Engineering — Technical Architecture

Feature engineering transforms raw historical demand into tabular inputs suitable for gradient boosting, while preventing data leakage.

```
Time Horizon:
   t-28 ... t-14 ... t-7 ... t-1       t            t+1
  [───────── Past History ─────────] [ Today ]  [ Forecast Target ]
                                     Features   Target (next-day demand)
```

### 1. Leakage Prevention Mechanism
A critical error in demand forecasting is target leakage—using information from today ($t$) or the future ($t+1$) to predict $t+1$:
- The target is generated using a forward shift: `group['daily_demand'].shift(-1)`.
- All historical lag and rolling features are shifted by at least 1 day into the past: `previous_demand = demand.shift(1)`.
- **Guarantee:** Features computed for date $t$ strictly use transactions from dates $\le t-1$, preventing the model from learning shortcuts that fail in production.

### 2. Feature Taxonomy

| Feature Category | Column Name(s) | Mathematical Formulation | Business & Modeling Rationale |
| :--- | :--- | :--- | :--- |
| **Short-Term Lag** | `lag_1` | $\text{demand}_{t-1}$ | Captures immediate day-prior momentum and stockout hangover. |
| **Seasonal Lags** | `lag_7`, `lag_14`, `lag_28` | $\text{demand}_{t-7}, \text{demand}_{t-14}, \text{demand}_{t-28}$ | Captures weekly shopping cycles (e.g., Saturday surges) and monthly payday patterns. |
| **Rolling Means** | `rolling_mean_7`, `rolling_mean_14`, `rolling_mean_28` | $\frac{1}{W} \sum_{k=1}^W \text{demand}_{t-k}$ | Establishes recent baseline sales velocity across 1-week, 2-week, and 4-week horizons, smoothing out daily noise. |
| **Rolling Volatility** | `rolling_std_7`, `rolling_std_28` | $\sqrt{\frac{1}{W-1} \sum_{k=1}^W (\text{demand}_{t-k} - \mu)^2}$ | Quantifies demand stability vs. volatility for safety-stock calculations. |
| **Pricing Signals** | `price`, `price_change` | $\text{price}_t$, $\text{price}_t - \text{price}_{t-1}$ | Encodes price elasticity and promotional discount incentives. |
| **Calendar Drivers** | `day_of_week`, `day_of_month`, `month`, `week_of_year`, `is_weekend` | Extracted from calendar date | Captures structural human behavior: weekend shopping surges, holiday build-ups, and intra-month seasonality. |
| **Hierarchical Codes** | `store_id`, `item_code`, `dept_code`, `class_code`, `subclass_code`, `item_type_code` | Integer categorical label codes | Allows trees to learn store-specific foot-traffic baselines and cross-category demand profiles. |
| **Physical Attributes**| `weight_volume`, `weight_netto`, `fatness` | Continuous float values from catalog | Distinguishes bulk goods from small impulse items. |

### 3. Categorical Encoding Strategy
Rather than generating over 28,000 sparse columns via One-Hot Encoding (which would cause memory exhaustion and dilute tree split quality), the pipeline encodes categories as compact integer identifiers (`int16`/`int32`) and utilizes XGBoost's native partition-based categorical splitting (`enable_categorical=True`, `max_cat_threshold=64`).

---

## 7. Why XGBoost?

XGBoost (Extreme Gradient Boosting) was selected as the primary forecasting engine based on several technical considerations:

### Conceptual Mechanism
Gradient Boosted Decision Trees (GBDT) fit an ensemble of shallow regression trees sequentially. Each subsequent tree models the pseudo-residuals (negative gradient of the loss function) of the existing ensemble:
$$\hat{y}_i^{(m)} = \hat{y}_i^{(m-1)} + \eta \cdot f_m(x_i)$$
where $\eta$ is the learning rate and $f_m$ is a regression tree optimizing mean squared error.

### Architectural Advantages for Retail Tabular Data
1. **Heterogeneous Feature Handling:** Seamlessly combines continuous metrics (rolling means, prices) with discrete categories (store ID, department codes) without requiring complex normalization.
2. **Non-Linear Interactions:** Automatically discovers complex conditional relationships (e.g., *a 20% price drop increases demand only on weekends at Store 1*).
3. **Native Missing Value Handling:** When new products lack historical lags (`NaN`), XGBoost learns optimal default branch routing during training rather than requiring arbitrary zero-imputation.
4. **Computational Efficiency:** The histogram algorithm (`tree_method="hist"`) discretizes continuous features into 256 bins, enabling training across 14 million rows in minutes with multi-threaded CPU execution.

### Limitations in This Architecture
- **Inability to Extrapolate:** Tree models partition feature space into orthogonal bounding boxes. They cannot predict a value higher than the maximum leaf average observed during training.
- **Tail Regularization:** Minimizing global mean squared error over zero-inflated distributions (>95% of demand $\le 5$) penalizes large predictions unless reinforced by strong historical lag signals.

---

## 8. Training Strategy and Evaluation

### Chronological Data Splitting
Random $k$-fold cross-validation is **invalid** for time-series forecasting because it trains on future records to predict past records, causing lookahead leakage and unrealistically optimistic evaluations.

This platform enforces a strict **chronological train-validation-test split**:

| Split | Date Range | Calendar Days | Observation Count | Percentage | Operational Role |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Train** | `2022-08-28` to `2024-03-31` | 582 days | 14,114,747 | 70.83% | Model parameter learning |
| **Validation** | `2024-04-01` to `2024-06-30` | 91 days | 3,133,253 | 15.72% | Early stopping & hyperparameter tuning |
| **Test** | `2024-07-01` to `2024-09-25` | 87 days | 2,680,885 | 13.45% | Final out-of-sample performance evaluation |

### Model Hyperparameters
- **Objective:** `reg:squarederror`
- **Tree Algorithm:** `hist` (Histogram-based)
- **Max Tree Depth:** `8`
- **Learning Rate ($\eta$):** `0.05`
- **Subsample Ratio:** `0.8` (Row subsampling per tree)
- **Column Subsample Ratio:** `0.8` (Feature subsampling per tree)
- **Minimum Child Weight:** `5`
- **L2 Regularization ($\lambda$):** `1.0`
- **Early Stopping:** 20 rounds monitored on validation RMSE

### Verified Evaluation Results

| Dataset Split | Rows Evaluated | MAE (units) | RMSE (units) | $R^2$ Score | WAPE (%) | Mean Actual | Mean Predicted |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Validation Set** | 3,133,253 | **1.2396** | **4.8500** | **0.9228** | **54.36%** | 2.2801 | 2.2114 |
| **Test Set (Unseen)** | 2,680,885 | **1.3841** | **12.7371** | **0.6585** | **53.69%** | 2.5779 | 2.3787 |

### Metric Definitions and Interpretation
- **MAE (Mean Absolute Error = 1.38 units):** On average, the model's next-day forecast deviates from actual demand by only 1.38 product units across the entire catalog.
- **RMSE (Root Mean Squared Error = 12.74 units):** Because RMSE squares errors before averaging, it heavily penalizes rare, large forecast errors. The increase from validation (4.85) to test (12.74) indicates the presence of extreme tail outliers in the test period.
- **$R^2$ (Coefficient of Determination = 0.6585):** Measures the proportion of variance in test demand explained by the model relative to a naive mean forecast. It is **not** an accuracy percentage.
- **WAPE (Weighted Absolute Percentage Error = 53.69%):** Total absolute forecast errors divided by total actual demand ($\frac{\sum |y - \hat{y}|}{\sum y}$). In sparse, zero-inflated retail datasets with thousands of slow-moving SKUs, a WAPE of ~53% represents a stable operational baseline.

---

## 9. Important Findings and Current Limitations

### The Extreme Error Investigation
During diagnostic error analysis (`src/investigate_extreme_errors.py`), the test period RMSE was found to be disproportionately inflated. A targeted audit of the **30 largest absolute prediction errors** uncovered the following:

1. **A Single SKU Caused the Distortion:**
   All 30 of the largest errors in the test period belonged to one product: **`item_id = 327c5bc1e583`** (Store 4: 23 records, Store 1: 7 records).
2. **Massive Squared Error Concentration:**
   These 30 observations alone accounted for **308,529,344 of the total 434,929,996 test SSE—representing 70.94% of the entire test set squared error** from just 30 out of 2.68 million rows (0.0011%).
   - Actual Demand: 2,329 to 4,952 units/day
   - Model Prediction: 4.8 to 44.2 units/day
   - Excluding this single SKU, test RMSE drops from **12.74 to ~6.8**.

### Raw POS Data Quality Audit
A dedicated verification script (`src/check_product_data_quality.py`) cross-referenced raw transactions in `data/sales.csv`, `data/daily_demand.csv`, `data/prepared_demand.csv`, and metadata tables:
- **Zero Preprocessing Inflation:** Raw `sales.csv` already contained recorded daily quantities of 2,000–5,000 units. The pipeline reproduced the source data with zero discrepancy.
- **Financial Profile:** The base price was recorded at `0.01` across all stores. Total revenue generated across 152,687 units sold was **1,300.67 currency units**, indicating checkout packaging, carrier bags, or till vouchers.
- **Price History Trace:** `data/price_history.csv` confirmed this SKU was first created on **2024-08-20** at price 0.01.
- **Catalog Absence:** The product had no entry in `data/catalog.csv` (`dept_name = NaN`).

### The Cold-Start Limitation
This investigation highlights the primary limitation of lag-dependent gradient boosting:
- Product `327c5bc1e583` first sold on **2024-09-02**. It had **zero records in the 19-month training set** and **zero records in validation**.
- Because it was uncataloged, all category codes were mapped to `0` (missing), and its lag/rolling features were initially `NaN`.
- XGBoost had never observed an uncataloged product selling thousands of units at price 0.01, and regularized its predictions toward the training conditional mean (~30 units).
- **Key Takeaway:** Unseen, uncataloged products require specialized cold-start modeling strategies, such as store-level velocity priors and dedicated handling for high-volume checkout supplies.

---

## 10. Current Progress and Next Steps

```
[ COMPLETED ]
  ├── POS Data Preprocessing & Price Imputation (`src/preprocessing.py`)
  ├── Daily Demand Aggregation (`src/aggregation.py`)
  ├── Lifespan Zero-Demand Preparation (`src/demand_preparation.py`)
  ├── 28-Feature Supervised Engineering (`src/feature_engineering.py`)
  ├── Chronological Train-Val-Test Splitting (`src/XGBoost.py`)
  ├── XGBoost Out-of-Core Baseline Training (`src/XGboost_training.py`)
  ├── Full Model Error Analysis (`src/xgboost_error_analysis.py`)
  ├── Extreme Prediction Error Investigation (`src/investigate_extreme_errors.py`)
  ├── Raw POS Data Quality Verification (`src/check_product_data_quality.py`)
  └── Cold-Start Cohort Diagnostic Script (`src/audit_cold_start_cohorts.py`)

[ PLANNED / NEXT PHASES ]
  ├── Cold-Start Model Enhancements (Category-level priors & new product indicators)
  ├── Explainable AI via SHAP (Feature impact and attribution analysis)
  ├── LLM-Powered Business Intelligence (Natural language insights for store managers)
  ├── Batch Inference Pipeline (Predictions on user-uploaded sales CSV files)
  └── Interactive Operational Dashboard (Web interface for inventory managers)
```

---

## 11. Project Structure

```
d:/final year project/
├── data/                                 # Datasets (Raw, Processed, and Engineered)
│   ├── sales.csv                         # Raw POS transaction records (7.43M rows)
│   ├── catalog.csv                       # Product metadata (departments, classes, weights)
│   ├── price_history.csv                 # Historical price tracking table
│   ├── discounts_history.csv             # Promotional campaign records
│   ├── daily_demand.csv                  # Daily aggregated demand (date, store, item)
│   ├── prepared_demand.csv               # Continuous zero-filled demand series (19.98M rows)
│   └── features.csv                      # Final supervised feature dataset (19.92M rows, 28 cols)
├── models/                               # Serialized Model Artifacts
│   └── xgboost_demand_model.json         # Trained XGBoost baseline model booster
├── reports/                              # Automated Diagnostic Reports
│   ├── extreme_error_investigation.csv   # Top 30 extreme error event data
│   ├── extreme_error_investigation.txt   # Detailed root-cause investigation report
│   └── product_327c5bc1e583_data_quality.txt # Raw POS audit for high-volume anomaly product
├── src/                                  # Source Code Modules
│   ├── preprocessing.py                  # Raw sales cleaning & negative price imputation
│   ├── aggregation.py                    # Daily demand aggregation logic
│   ├── demand_preparation.py             # Active lifespan gap-filling & negative clipping
│   ├── feature_engineering.py            # Temporal lags, rolling statistics & catalog encoding
│   ├── XGBoost.py                        # Chronological data split inspection utility
│   ├── XGboost_training.py               # Out-of-core XGBoost model training & validation
│   ├── xgboost_error_analysis.py         # Full test-set performance & error tier analysis
│   ├── investigate_extreme_errors.py     # Diagnostic deep-dive into largest absolute errors
│   ├── check_product_data_quality.py     # POS raw data quality & aggregation verifier
│   └── audit_cold_start_cohorts.py       # Seen vs. unseen product evaluation script
├── my4venv/                              # Project Virtual Environment (Python 3.11)
└── README.md                             # Comprehensive Project Documentation
```

---

## 12. Reproduction and Usage

To run any stage of the pipeline using the project virtual environment in Windows PowerShell:

```powershell
# 1. Activate the environment
& "d:\final year project\my4venv\Scripts\Activate.ps1"

# 2. Run data quality audit for the anomalous product
& "d:\final year project\my4venv\Scripts\python.exe" src/check_product_data_quality.py

# 3. Run extreme error investigation
& "d:\final year project\my4venv\Scripts\python.exe" src/investigate_extreme_errors.py

# 4. Run cold-start cohort evaluation
& "d:\final year project\my4venv\Scripts\python.exe" src/audit_cold_start_cohorts.py
```
