from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os


app = FastAPI(
    title="Predictive Sales Analytics API",
    description="Machine Learning API for Sales Prediction",
    version="1.0"
)


# Get project root directory
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "best_model.pkl"
)


# Load trained model
model = joblib.load(MODEL_PATH)


class SalesRequest(BaseModel):

    ship_mode: str
    segment: str
    country: str
    city: str
    state: str
    postal_code: int
    region: str
    category: str
    sub_category: str
    quantity: int
    discount: float


@app.get("/")
def home():

    return {
        "message": "Predictive Sales Analytics API",
        "status": "running",
        "model": "Gradient Boosting"
    }


@app.post("/predict")
def predict(request: SalesRequest):

    data = pd.DataFrame([{

        "Ship Mode": request.ship_mode,
        "Segment": request.segment,
        "Country": request.country,
        "City": request.city,
        "State": request.state,
        "Postal Code": request.postal_code,
        "Region": request.region,
        "Category": request.category,
        "Sub-Category": request.sub_category,
        "Quantity": request.quantity,
        "Discount": request.discount

    }])

    prediction = model.predict(data)[0]

    return {

        "predicted_sales": round(
            float(prediction),
            2
        )

    }