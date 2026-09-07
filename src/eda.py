import pandas as pd
import matplotlib.pyplot as plt

# ==========================================================
# Load Dataset
# ==========================================================

print("=" * 70)
print("Loading Dataset...")
print("=" * 70)

sales = pd.read_csv("data/sales.csv")
catalog = pd.read_csv("data/catalog.csv")

# ==========================================================
# EDA STEP 1 - BASIC INFORMATION
# ==========================================================

print("\nSales Shape")
print(sales.shape)

print("\nCatalog Shape")
print(catalog.shape)

print("\nSales Columns")
print(sales.columns)

print("\nCatalog Columns")
print(catalog.columns)

print("\nSales Data Types")
print(sales.dtypes)

print("\nCatalog Data Types")
print(catalog.dtypes)

print("\nSales Missing Values")
print(sales.isnull().sum())

print("\nCatalog Missing Values")
print(catalog.isnull().sum())

print("\nSales Summary")
print(sales.describe())

print("\nCatalog Summary")
print(catalog.describe())

print("\nUnique Products Sold :", sales["item_id"].nunique())
print("Unique Stores :", sales["store_id"].nunique())

print("=" * 70)
print("EDA Step 1 Completed")
print("=" * 70)

# ==========================================================
# EDA STEP 2 - PRODUCT ANALYSIS
# ==========================================================

print("\n" + "=" * 70)
print("EDA Step 2 - Product Analysis")
print("=" * 70)

# Merge sales and catalog
merged = pd.merge(
    sales,
    catalog,
    on="item_id",
    how="left"
)

print("\nMerged Dataset Shape")
print(merged.shape)

print("\nMerged Dataset Preview")
print(merged.head())

# ==========================================================
# TOP 10 SELLING PRODUCTS
# ==========================================================

print("\n" + "=" * 70)
print("Top 10 Selling Products")
print("=" * 70)

top_products = (
    merged.groupby("item_id")["quantity"]
          .sum()
          .sort_values(ascending=False)
          .head(10)
)

print(top_products)

# Plot Top Products

plt.figure(figsize=(12,6))

top_products.plot(
    kind="bar",
    color="royalblue"
)

plt.title("Top 10 Selling Products")
plt.xlabel("Product ID")
plt.ylabel("Total Quantity Sold")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# ==========================================================
# DEPARTMENT-WISE SALES
# ==========================================================

print("\n" + "=" * 70)
print("Department-wise Sales")
print("=" * 70)

department_sales = (
    merged.groupby("dept_name")["quantity"]
          .sum()
          .sort_values(ascending=False)
)

print(department_sales.head(15))

# Plot Department Sales

plt.figure(figsize=(14,7))

department_sales.head(15).plot(
    kind="bar",
    color="darkgreen"
)

plt.title("Top 15 Departments by Quantity Sold")
plt.xlabel("Department")
plt.ylabel("Total Quantity Sold")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

print("\n" + "=" * 70)
print("EDA Step 2 Completed Successfully")
print("=" * 70)