"""FastAPI app for serving taxi fare predictions."""
import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Taxi Fare API")
model = joblib.load("models/taxi_model.pkl")

FEATURES = ["passenger_count", "trip_distance", "fare_amount", "tip_amount"]


class Trip(BaseModel):
    """Input trip features."""
    passenger_count: float
    trip_distance: float
    fare_amount: float
    tip_amount: float


@app.get("/health")
def health():
    """Health check."""
    return {"status": "ok"}


@app.post("/predict")
def predict(trip: Trip):
    """Predict the total fare amount."""
    row = pd.DataFrame([[
        trip.passenger_count,
        trip.trip_distance,
        trip.fare_amount,
        trip.tip_amount,
    ]], columns=FEATURES)
    prediction = float(model.predict(row)[0])
    return {"total_amount": prediction}


print('Model converted to fast-api')
