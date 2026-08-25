from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib


app = FastAPI(
    title="Predictive Sales Analytics API",
    description="Machine Learning API for sales prediction",
    version="1.0"
)


# Load model
model = joblib.load(
    "models/best_model.pkl"
)


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
        "status": "running"
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