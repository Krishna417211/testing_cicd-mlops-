"""Tests for preprocessing."""
import numpy as np
import pandas as pd

from src.preprocess import clean_data


def make_df():
    """Small raw frame for testing."""
    return pd.DataFrame({
        "passenger_count": [1, 2, 1, 1],
        "trip_distance": [1.0, 0.0, 2.5, 3.0],
        "fare_amount": [5.0, 7.0, 0.0, 10.0],
        "tip_amount": [1.0, 0.0, 2.0, 3.0],
        "total_amount": [6.0, 7.0, 2.0, 13.0],
    })


def test_removes_non_positive_rows():
    """Rows with zero fare/distance are dropped."""
    result = clean_data(make_df())
    assert len(result) == 2
    assert (result["fare_amount"] > 0).all()
    assert (result["trip_distance"] > 0).all()


def test_drops_nulls():
    """Null rows are removed."""
    df = make_df()
    df.loc[0, "fare_amount"] = np.nan
    result = clean_data(df)
    assert not result.isnull().any().any()
