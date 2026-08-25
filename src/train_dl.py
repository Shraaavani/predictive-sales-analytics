"""
train_dl.py
------------
Adds a deep learning model (TensorFlow/Keras) to the existing model
comparison, alongside Linear Regression, Random Forest, Gradient Boosting,
and XGBoost trained in train.py.

Uses the same preprocessing pipeline (OneHotEncoder + passthrough numeric)
so the comparison is apples-to-apples with the other models, then appends
its results to reports/model_comparison.csv.
"""

import pandas as pd
import numpy as np
import os
import time
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.random.set_seed(42)
np.random.seed(42)

DATA_PATH = "data/processed/sales_processed.csv"
MODEL_DIR = "models"
RESULTS_DIR = "reports"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 60)
print("PREDICTIVE SALES ANALYTICS - DEEP LEARNING MODEL")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
print(f"\nDataset shape: {df.shape}")

columns_to_drop = [
    "Sales", "Profit", "discount_percentage",
    "quantity_category", "discount_category"
]

X = df.drop(columns=columns_to_drop)
y = df["Sales"]

categorical_features = [
    "Ship Mode", "Segment", "Country", "City",
    "State", "Region", "Category", "Sub-Category"
]
numerical_features = ["Postal Code", "Quantity", "Discount"]

# Deep learning needs scaled numeric inputs (unlike tree models),
# so this preprocessor differs slightly from train.py's.
preprocessor = ColumnTransformer(transformers=[
    ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ("numerical", StandardScaler(), numerical_features),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Fit preprocessor and transform to dense arrays for Keras
X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc = preprocessor.transform(X_test)

if hasattr(X_train_proc, "toarray"):
    X_train_proc = X_train_proc.toarray()
    X_test_proc = X_test_proc.toarray()

input_dim = X_train_proc.shape[1]
print(f"Input feature dimension after encoding: {input_dim}")


def build_model(input_dim):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.Dense(1),
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss="mse", metrics=["mae"])
    return model


model = build_model(input_dim)
model.summary()

early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)

print("\nTraining neural network...")
start_time = time.time()

history = model.fit(
    X_train_proc, y_train,
    validation_split=0.15,
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=2,
)

training_time = time.time() - start_time

predictions = model.predict(X_test_proc).flatten()

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print(f"\nMAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²:   {r2:.4f}")
print(f"Training Time: {training_time:.2f} seconds")
print(f"Epochs run: {len(history.history['loss'])} (early stopping)")

# ==========================================
# SAVE MODEL + PREPROCESSOR (Keras models need
# both saved together for later inference)
# ==========================================
model.save(f"{MODEL_DIR}/deep_learning_model.keras")
joblib.dump(preprocessor, f"{MODEL_DIR}/dl_preprocessor.pkl")

# ==========================================
# APPEND TO MODEL COMPARISON
# ==========================================
results_path = f"{RESULTS_DIR}/model_comparison.csv"

new_row = pd.DataFrame([{
    "Model": "Deep Learning (Keras MLP)",
    "MAE": mae,
    "RMSE": rmse,
    "R2": r2,
    "Training_Time": training_time,
}])

if os.path.exists(results_path):
    existing = pd.read_csv(results_path)
    # Remove any previous DL row so re-running this script doesn't duplicate it
    existing = existing[existing["Model"] != "Deep Learning (Keras MLP)"]
    combined = pd.concat([existing, new_row], ignore_index=True)
else:
    combined = new_row

combined = combined.sort_values(by="R2", ascending=False)
combined.to_csv(results_path, index=False)

print("\n" + "=" * 60)
print("UPDATED MODEL COMPARISON")
print("=" * 60)
print(combined.to_string(index=False))
print(f"\nDeep learning model saved to: {MODEL_DIR}/deep_learning_model.keras")
print(f"Comparison updated at: {results_path}")
