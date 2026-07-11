"""Train a taxi fare model using LinearRegression."""

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

FEATURES = ["passenger_count", "trip_distance", "fare_amount", "tip_amount"]
TARGET = "total_amount"


def load_data(path="data/taxi_clean.csv"):
    """Load features and target from the CSV."""
    df = pd.read_csv(path)
    x = df[FEATURES]
    y = df[TARGET]
    return x, y


def split_data():
    """Split the dataset into training and testing sets."""
    x, y = load_data()

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42
    )

    return x_train, x_test, y_train, y_test


def train_model(x_train, y_train):
    """Train the Linear Regression model."""
    model = LinearRegression()
    model.fit(x_train, y_train)
    return model


def evaluate_model(model, x_test, y_test):
    """Evaluate the model using the R² score."""
    predictions = model.predict(x_test)
    score = r2_score(y_test, predictions)
    return score


def save_model(model, path="models/taxi_model.pkl"):
    """Save the trained model."""
    joblib.dump(model, path)


def main():
    """Main function."""
    x_train, x_test, y_train, y_test = split_data()

    model = train_model(x_train, y_train)

    score = evaluate_model(model, x_test, y_test)

    save_model(model)

    print(f"R² Score: {score:.4f}")
    print("Model saved successfully.")
    print("Model training completed.")


if __name__ == "__main__":
    main()
