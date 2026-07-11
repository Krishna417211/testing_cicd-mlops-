"""Tests for the FastAPI app."""
import numpy as np
from fastapi.testclient import TestClient


def make_client(mocker):
    """Mock the model then build a test client."""
    fake_model = mocker.MagicMock()
    fake_model.predict.return_value = np.array([15.0])
    mocker.patch("joblib.load", return_value=fake_model)

    from src.app import app
    return TestClient(app)


def test_health(mocker):
    """Health endpoint returns ok."""
    client = make_client(mocker)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_predict(mocker):
    """Predict endpoint returns the predicted total."""
    client = make_client(mocker)
    payload = {
        "passenger_count": 1,
        "trip_distance": 2.5,
        "fare_amount": 10.0,
        "tip_amount": 2.0,
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    assert resp.json()["total_amount"] == 15.0
