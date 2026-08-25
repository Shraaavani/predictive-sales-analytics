import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


DATA_PATH = "data/processed/sales_processed.csv"
FIGURE_PATH = "reports/figures"

os.makedirs(FIGURE_PATH, exist_ok=True)

# Load data
df = pd.read_csv(DATA_PATH)

print("=" * 50)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 50)

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistical Summary:")
print(df.describe())


# ==========================================
# 1. SALES DISTRIBUTION
# ==========================================

plt.figure(figsize=(10, 6))

sns.histplot(
    df["Sales"],
    bins=50,
    kde=True
)

plt.title("Sales Distribution")
plt.xlabel("Sales")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(
    f"{FIGURE_PATH}/sales_distribution.png"
)

plt.close()


# ==========================================
# 2. SALES BY CATEGORY
# ==========================================

category_sales = (
    df.groupby("Category")["Sales"]
    .sum()
    .sort_values(ascending=False)
)

plt.figure(figsize=(10, 6))

category_sales.plot(
    kind="bar"
)

plt.title("Total Sales by Category")
plt.xlabel("Category")
plt.ylabel("Sales")

plt.tight_layout()

plt.savefig(
    f"{FIGURE_PATH}/sales_by_category.png"
)

plt.close()


# ==========================================
# 3. SALES BY REGION
# ==========================================

region_sales = (
    df.groupby("Region")["Sales"]
    .sum()
    .sort_values(ascending=False)
)

plt.figure(figsize=(10, 6))

region_sales.plot(
    kind="bar"
)

plt.title("Total Sales by Region")
plt.xlabel("Region")
plt.ylabel("Sales")

plt.tight_layout()

plt.savefig(
    f"{FIGURE_PATH}/sales_by_region.png"
)

plt.close()


# ==========================================
# 4. PROFIT BY CATEGORY
# ==========================================

category_profit = (
    df.groupby("Category")["Profit"]
    .sum()
    .sort_values(ascending=False)
)

plt.figure(figsize=(10, 6))

category_profit.plot(
    kind="bar"
)

plt.title("Total Profit by Category")
plt.xlabel("Category")
plt.ylabel("Profit")

plt.tight_layout()

plt.savefig(
    f"{FIGURE_PATH}/profit_by_category.png"
)

plt.close()


# ==========================================
# 5. SALES VS PROFIT
# ==========================================

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="Sales",
    y="Profit",
    alpha=0.5
)

plt.title("Sales vs Profit")
plt.xlabel("Sales")
plt.ylabel("Profit")

plt.tight_layout()

plt.savefig(
    f"{FIGURE_PATH}/sales_vs_profit.png"
)

plt.close()


# ==========================================
# 6. DISCOUNT VS PROFIT
# ==========================================

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="Discount",
    y="Profit",
    alpha=0.5
)

plt.title("Discount vs Profit")
plt.xlabel("Discount")
plt.ylabel("Profit")

plt.tight_layout()

plt.savefig(
    f"{FIGURE_PATH}/discount_vs_profit.png"
)

plt.close()


# ==========================================
# 7. SALES BY SUB-CATEGORY
# ==========================================

subcategory_sales = (
    df.groupby("Sub-Category")["Sales"]
    .sum()
    .sort_values(ascending=False)
)

plt.figure(figsize=(12, 6))

subcategory_sales.plot(
    kind="bar"
)

plt.title("Sales by Sub-Category")
plt.xlabel("Sub-Category")
plt.ylabel("Sales")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    f"{FIGURE_PATH}/sales_by_subcategory.png"
)

plt.close()


# ==========================================
# BUSINESS INSIGHTS
# ==========================================

print("\n" + "=" * 50)
print("BUSINESS INSIGHTS")
print("=" * 50)

print(
    "\nHighest Sales Category:",
    category_sales.idxmax()
)

print(
    "Highest Sales Region:",
    region_sales.idxmax()
)

print(
    "Most Profitable Category:",
    category_profit.idxmax()
)

print(
    "Best Selling Sub-Category:",
    subcategory_sales.idxmax()
)

print(
    "\nTotal Sales:",
    round(df["Sales"].sum(), 2)
)

print(
    "Total Profit:",
    round(df["Profit"].sum(), 2)
)

print(
    "Average Sales:",
    round(df["Sales"].mean(), 2)
)

print("\nEDA completed successfully!")

print(
    f"\nCharts saved in: {FIGURE_PATH}"
)