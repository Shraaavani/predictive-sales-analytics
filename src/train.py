import pandas as pd
import numpy as np
import os
import time
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor


# ==========================================
# CONFIGURATION
# ==========================================

DATA_PATH = "data/processed/sales_processed.csv"

MODEL_DIR = "models"

RESULTS_DIR = "reports"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ==========================================
# LOAD DATA
# ==========================================

print("=" * 60)
print("PREDICTIVE SALES ANALYTICS - MODEL TRAINING")
print("=" * 60)

df = pd.read_csv(DATA_PATH)

print(f"\nDataset shape: {df.shape}")


# ==========================================
# REMOVE UNNECESSARY / LEAKAGE COLUMNS
# ==========================================

# Profit is excluded because it is directly related
# to Sales and would cause target leakage.

columns_to_drop = [
    "Sales",
    "Profit",
    "discount_percentage",
    "quantity_category",
    "discount_category"
]

X = df.drop(
    columns=columns_to_drop
)

y = df["Sales"]


# ==========================================
# FEATURE TYPES
# ==========================================

categorical_features = [
    "Ship Mode",
    "Segment",
    "Country",
    "City",
    "State",
    "Region",
    "Category",
    "Sub-Category"
]

numerical_features = [
    "Postal Code",
    "Quantity",
    "Discount"
]


# ==========================================
# PREPROCESSING
# ==========================================

preprocessor = ColumnTransformer(

    transformers=[

        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        ),

        (
            "numerical",
            "passthrough",
            numerical_features
        )

    ]
)


# ==========================================
# TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42
)


print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")


# ==========================================
# MODELS
# ==========================================

models = {

    "Linear Regression":
        LinearRegression(),

    "Random Forest":
        RandomForestRegressor(
            n_estimators=150,
            max_depth=15,
            random_state=42,
            n_jobs=-1
        ),

    "Gradient Boosting":
        GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=4,
            random_state=42
        ),

    "XGBoost":
        XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1
        )
}


# ==========================================
# TRAIN MODELS
# ==========================================

results = []

best_model = None
best_model_name = None
best_r2 = -float("inf")


for name, model in models.items():

    print("\n" + "-" * 60)
    print(f"Training: {name}")

    pipeline = Pipeline(

        steps=[

            (
                "preprocessor",
                preprocessor
            ),

            (
                "model",
                model
            )

        ]

    )

    start_time = time.time()

    pipeline.fit(
        X_train,
        y_train
    )

    training_time = time.time() - start_time

    # Predictions
    predictions = pipeline.predict(X_test)

    # Metrics
    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²:   {r2:.4f}")
    print(
        f"Training Time: "
        f"{training_time:.2f} seconds"
    )

    results.append({

        "Model": name,

        "MAE": mae,

        "RMSE": rmse,

        "R2": r2,

        "Training_Time": training_time

    })

    # Select best model based on R²
    if r2 > best_r2:

        best_r2 = r2

        best_model = pipeline

        best_model_name = name


# ==========================================
# SAVE RESULTS
# ==========================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="R2",
    ascending=False
)

results_df.to_csv(
    f"{RESULTS_DIR}/model_comparison.csv",
    index=False
)


# ==========================================
# SAVE BEST MODEL
# ==========================================

joblib.dump(
    best_model,
    f"{MODEL_DIR}/best_model.pkl"
)


# ==========================================
# FINAL OUTPUT
# ==========================================

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(
    results_df.to_string(
        index=False
    )
)

print("\n" + "=" * 60)

print(
    f"BEST MODEL: {best_model_name}"
)

print(
    f"BEST R²: {best_r2:.4f}"
)

print(
    f"\nModel saved to:"
    f" {MODEL_DIR}/best_model.pkl"
)

print(
    "\nComparison saved to:"
    f" {RESULTS_DIR}/model_comparison.csv"
)

print("=" * 60)