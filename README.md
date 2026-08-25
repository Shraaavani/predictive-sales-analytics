# Predictive Sales Analytics & AI Model Deployment

An end-to-end machine learning pipeline that predicts sales revenue from
order attributes (region, category, discount, quantity, etc.), compares
five modeling approaches (including a deep neural network), and deploys
the best-performing model as both a REST API and an interactive dashboard.

## Project Status

| Deliverable | Status |
|---|---|
| Data collection | ✅ Superstore sales dataset (9,995 orders) |
| Data preprocessing & feature engineering | ✅ `src/data_preprocessing.py` |
| Exploratory Data Analysis | ✅ `src/eda.py` (7 charts in `reports/figures/`) |
| Multiple ML models (Scikit-learn/XGBoost) | ✅ `src/train.py` |
| Deep learning model (TensorFlow/Keras) | ✅ `src/train_dl.py` |
| Model comparison | ✅ `reports/model_comparison.csv` |
| Model deployment (FastAPI) | ✅ `api/main.py` |
| Dashboard | ✅ `dashboard/app.py` (Streamlit) |
| Documentation | ✅ this file |
| `requirements.txt` | ✅ |
| `.gitignore` | ✅ |

## Architecture & Pipeline Workflow

```
data/raw/sales.csv
        │
        ▼
src/data_preprocessing.py   →  cleans data, engineers features
        │                       (discount %, quantity/discount tiers)
        ▼
data/processed/sales_processed.csv
        │
        ├──────────────────────────────┐
        ▼                              ▼
src/eda.py                     src/train.py + src/train_dl.py
(exploratory analysis,          (trains 5 models: Linear Regression,
 7 charts saved to               Random Forest, Gradient Boosting,
 reports/figures/)               XGBoost, and a Keras neural network;
                                  picks the best by R² score)
                                        │
                                        ▼
                          models/best_model.pkl
                          models/deep_learning_model.keras
                          reports/model_comparison.csv
                                        │
                        ┌───────────────┴───────────────┐
                        ▼                                ▼
                  api/main.py                    dashboard/app.py
              (FastAPI REST API,                 (Streamlit dashboard —
               POST /predict)                     KPIs, filters, charts,
                                                   model comparison,
                                                   live prediction form)
```

## Dataset

Superstore-style sales orders (9,995 rows) with columns: Ship Mode,
Segment, Country, City, State, Postal Code, Region, Category,
Sub-Category, Sales, Quantity, Discount, Profit.

**Target variable:** `Sales`. `Profit` is deliberately excluded from the
model's input features (`src/train.py`) since it is mathematically
derived from Sales and would leak the target into the model.

## Feature Engineering

- `discount_percentage` — Discount expressed as a percentage
- `quantity_category` — Low / Medium / High / Very High, binned from Quantity
- `discount_category` — No / Low / Medium / High Discount, binned from Discount
- Categorical features (Ship Mode, Segment, Region, Category, etc.) are
  one-hot encoded; numeric features are passed through as-is for tree
  models, and standardized (z-score) for the neural network.

## Models Trained & Compared

Five regression models were trained on an 80/20 train-test split
(random_state=42) and evaluated on held-out test data:

| Model | MAE | RMSE | R² | Training Time (s) |
|---|---|---|---|---|
| **Gradient Boosting** ⭐ | 180.37 | 491.57 | **0.3270** | 4.27 |
| Deep Learning (Keras MLP) | 182.14 | 492.94 | 0.3232 | 12.64 |
| XGBoost | 179.51 | 519.14 | 0.2494 | 1.53 |
| Linear Regression | 222.22 | 524.54 | 0.2337 | 0.91 |
| Random Forest | 177.62 | 527.70 | 0.2244 | 11.25 |

**Best model: Gradient Boosting** (selected automatically by `train.py`
based on highest R², and saved to `models/best_model.pkl`, which is what
the API and dashboard load for predictions).

### Deep Learning Model

`src/train_dl.py` adds a Keras multilayer perceptron (128 → 64 → 32 → 1
neurons, ReLU activations, dropout regularization, Adam optimizer,
early stopping on validation loss) trained on the same feature set with
standardized numeric inputs. It performs competitively with the best
tree-based model (R² 0.323 vs. 0.327) — evidence that the ceiling here is
largely the dataset's inherent noise/limited predictive signal rather
than model capacity. Results are automatically merged into
`reports/model_comparison.csv` alongside the other four models.

### Honest note on model performance

An R² around 0.32 means the models explain roughly a third of the
variance in Sales — a real, usable predictive signal, but this is not a
highly accurate model in absolute terms. This is a fair reflection of
how hard sales prediction is from order metadata alone, with no time
component, customer history, or seasonality features. If asked about
this in an interview or review, the honest framing is: "the model
captures a meaningful signal, and the deep learning result confirms
the current features are close to their information ceiling — the
highest-leverage next step would be adding time-series and customer-
level features, not a bigger model."

## Deployment

### REST API (FastAPI)

```bash
cd predictive-sales-analytics
uvicorn api.main:app --reload
```

`POST /predict` with a JSON body matching `SalesRequest` (ship_mode,
segment, country, city, state, postal_code, region, category,
sub_category, quantity, discount) returns a predicted sales value.

> **Note:** `src/predict.py` is an earlier, near-duplicate copy of
> `api/main.py`. `api/main.py` is the canonical, up-to-date version
> (it resolves the model path correctly regardless of working directory).
> Consider deleting `src/predict.py` to avoid confusion before pushing
> to GitHub.

### Dashboard (Streamlit)

```bash
streamlit run dashboard/app.py
```

Provides KPI cards (total sales, profit, records, average sale),
region/category filters, sales and profit breakdowns, the full model
comparison table, and an interactive form that calls the trained model
directly for live predictions.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

python src/data_preprocessing.py   # 1. clean + engineer features
python src/eda.py                  # 2. generate EDA charts
python src/train.py                # 3. train & compare 4 classical models
python src/train_dl.py             # 4. train & compare the deep learning model
uvicorn api.main:app --reload      # 5a. run the API
# or
streamlit run dashboard/app.py     # 5b. run the dashboard
```

## Project Structure

```
predictive-sales-analytics/
├── api/main.py                  # FastAPI deployment
├── dashboard/app.py             # Streamlit dashboard
├── data/
│   ├── raw/sales.csv
│   └── processed/sales_processed.csv
├── models/
│   ├── best_model.pkl           # Best classical model (Gradient Boosting)
│   ├── deep_learning_model.keras
│   └── dl_preprocessor.pkl
├── reports/
│   ├── figures/                 # 7 EDA charts
│   └── model_comparison.csv
├── src/
│   ├── data_preprocessing.py
│   ├── eda.py
│   ├── train.py                 # Classical ML models
│   └── train_dl.py              # Deep learning model
├── requirements.txt
├── .gitignore
└── README.md
```


