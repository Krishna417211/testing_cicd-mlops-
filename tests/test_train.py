"""Tests for the training functions."""
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.train import train_model, evaluate_model, FEATURES, TARGET


def sample_frame():
    """Small dataset for training tests."""
    return pd.DataFrame({
        "passenger_count": [1, 2, 1, 1, 3, 2],
        "trip_distance": [1.0, 3.0, 0.8, 5.0, 2.2, 6.5],
        "fare_amount": [6.5, 14.0, 5.0, 19.5, 9.5, 24.0],
        "tip_amount": [1.0, 2.5, 1.0, 4.0, 1.5, 4.5],
        "total_amount": [8.0, 17.5, 6.5, 24.5, 11.8, 29.5],
    })


def test_train_model_returns_linear_regression():
    """train_model returns a fitted LinearRegression."""
    df = sample_frame()
    model = train_model(df[FEATURES], df[TARGET])
    assert isinstance(model, LinearRegression)
    assert hasattr(model, "coef_")
    assert len(model.coef_) == len(FEATURES)


def test_train_model_can_predict():
    """The trained model produces one prediction per input row."""
    df = sample_frame()
    model = train_model(df[FEATURES], df[TARGET])
    preds = model.predict(df[FEATURES])
    assert len(preds) == len(df)


def test_evaluate_model_returns_score():
    """evaluate_model returns a float R2 score."""
    df = sample_frame()
    model = train_model(df[FEATURES], df[TARGET])
    score = evaluate_model(model, df[FEATURES], df[TARGET])
    assert isinstance(score, float)
    assert score > 0.9
