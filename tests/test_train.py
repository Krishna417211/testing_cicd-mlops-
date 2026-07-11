"""Tests for train.py (LinearRegression + MLflow)."""
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.train import load_data, train_model, evaluate_model, FEATURES, TARGET


def sample_frame():
    """Small dataset for training tests."""
    return pd.DataFrame({
        "passenger_count": [1, 2, 1, 1, 3, 2],
        "trip_distance": [1.0, 3.0, 0.8, 5.0, 2.2, 6.5],
        "fare_amount": [6.5, 14.0, 5.0, 19.5, 9.5, 24.0],
        "tip_amount": [1.0, 2.5, 1.0, 4.0, 1.5, 4.5],
        "total_amount": [8.0, 17.5, 6.5, 24.5, 11.8, 29.5],
    })


def test_load_data_columns(tmp_path):
    """load_data returns the right feature columns and target."""
    csv = tmp_path / "sample.csv"
    sample_frame().to_csv(csv, index=False)

    features, target = load_data(str(csv))
    assert list(features.columns) == FEATURES
    assert target.name == TARGET
    assert len(features) == len(target) == 6


def test_train_model_returns_linear_regression():
    """train_model returns a fitted LinearRegression."""
    df = sample_frame()
    model = train_model(df[FEATURES], df[TARGET])
    assert isinstance(model, LinearRegression)
    # A fitted model has learned coefficients.
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
    r2 = evaluate_model(model, df[FEATURES], df[TARGET])
    assert isinstance(r2, float)
    # total_amount is close to a linear function of the features here,
    # so the fit on training data should be strong.
    assert r2 > 0.9


def test_main_runs_with_mocked_mlflow(tmp_path, mocker):
    """main() runs end to end with MLflow and joblib mocked out."""
    # Point load_data at a temp CSV.
    csv = tmp_path / "sample.csv"
    sample_frame().to_csv(csv, index=False)
    mocker.patch("src.train.load_data", return_value=(
        sample_frame()[FEATURES], sample_frame()[TARGET]
    ))

    # Mock MLflow so no tracking DB or server is touched.
    mocker.patch("src.train.mlflow.set_tracking_uri")
    mocker.patch("src.train.mlflow.set_experiment")
    mocker.patch("src.train.mlflow.start_run")
    mocker.patch("src.train.mlflow.log_metric")
    mocker.patch("src.train.mlflow.sklearn.log_model")

    # Mock the model save so no file is written.
    dump = mocker.patch("src.train.joblib.dump")

    from src.train import main
    main()

    # The model was saved exactly once.
    dump.assert_called_once()