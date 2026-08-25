import pandas as pd
import os


RAW_DATA = "data/raw/sales.csv"
PROCESSED_DATA = "data/processed/sales_processed.csv"


def load_data():
    """Load raw sales dataset."""

    df = pd.read_csv(RAW_DATA)

    print("Dataset loaded successfully!")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    return df


def clean_data(df):
    """Clean the dataset."""

    print("\nCleaning data...")

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Remove rows where Sales is missing
    df = df.dropna(subset=["Sales"])

    # Fill missing categorical values
    categorical_columns = [
        "Ship Mode",
        "Segment",
        "Country",
        "City",
        "State",
        "Region",
        "Category",
        "Sub-Category"
    ]

    for column in categorical_columns:

        if column in df.columns:
            df[column] = df[column].fillna("Unknown")

    # Fill missing numerical values
    numerical_columns = [
        "Postal Code",
        "Quantity",
        "Discount"
    ]

    for column in numerical_columns:

        if column in df.columns:
            df[column] = df[column].fillna(
                df[column].median()
            )

    return df


def feature_engineering(df):
    """Create business-related features."""

    print("Creating features...")

    # Discount percentage
    df["discount_percentage"] = (
        df["Discount"] * 100
    )

    # Quantity categories
    df["quantity_category"] = pd.cut(
        df["Quantity"],
        bins=[0, 2, 5, 10, float("inf")],
        labels=[
            "Low",
            "Medium",
            "High",
            "Very High"
        ]
    )

    # Discount categories
    df["discount_category"] = pd.cut(
        df["Discount"],
        bins=[
            -0.01,
            0,
            0.10,
            0.30,
            float("inf")
        ],
        labels=[
            "No Discount",
            "Low Discount",
            "Medium Discount",
            "High Discount"
        ]
    )

    return df


def save_data(df):
    """Save processed dataset."""

    os.makedirs(
        "data/processed",
        exist_ok=True
    )

    df.to_csv(
        PROCESSED_DATA,
        index=False
    )

    print(
        f"\nProcessed data saved to: {PROCESSED_DATA}"
    )


if __name__ == "__main__":

    df = load_data()

    df = clean_data(df)

    df = feature_engineering(df)

    print("\nFinal dataset:")
    print(df.head())

    print("\nFinal shape:")
    print(df.shape)

    save_data(df)

    print("\nPreprocessing completed successfully!")