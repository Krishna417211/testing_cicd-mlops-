"""Train a taxi fare model using LinearRegression and log to MLflow."""
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

FEATURES = ["passenger_count", "trip_distance", "fare_amount", "tip_amount"]
TARGET = "total_amount"


def load_data(path="data/sample.csv"):
    """Load features and target from the CSV."""
    df = pd.read_csv(path)
    features = df[FEATURES]
    target = df[TARGET]
    return features, target


def train_model(x_train, y_train):
    """Fit a LinearRegression model."""
    model = LinearRegression()
    model.fit(x_train, y_train)
    return model


def evaluate_model(model, x_test, y_test):
    """Return the R2 score on the test set."""
    predictions = model.predict(x_test)
    return r2_score(y_test, predictions)


def main():
    """Full training flow: load, split, train, log to MLflow, save."""
    features, target = load_data()

    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.25, random_state=42
    )

    model = train_model(x_train, y_train)
    r2 = evaluate_model(model, x_test, y_test)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("taxi_experiment")

    with mlflow.start_run():
        mlflow.log_metric("accuracy", r2)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="taxi_model",
            registered_model_name="taxi_model",
        )
        print("logged in")

    joblib.dump(model, "models/taxi_model.pkl")
    print(f"R2: {r2:.4f}")
    print("Model saved")


if __name__ == "__main__":
    main()


print('Model training completed')
