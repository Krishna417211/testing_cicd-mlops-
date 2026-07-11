"""Preprocess the taxi dataset."""
import pandas as pd

COLUMNS = [
    "passenger_count",
    "trip_distance",
    "fare_amount",
    "tip_amount",
    "total_amount",
]


def clean_data(df):
    """Drop nulls and remove non-positive rows."""
    df = df.dropna()
    df = df[df["fare_amount"] > 0]
    df = df[df["trip_distance"] > 0]
    df = df[df["total_amount"] > 0]
    return df.reset_index(drop=True)


def main():
    """Load, clean and save the data."""
    df = pd.read_csv("data/sample.csv", usecols=COLUMNS)
    df = clean_data(df)
    df.to_csv("data/taxi_clean.csv", index=False)
    print("Data saved")


if __name__ == "__main__":
    main()
