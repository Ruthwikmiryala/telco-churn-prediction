import pandas as pd

from TelcoFeatures import add_features, clean_data, encode


def sample():
    return pd.DataFrame({
        "customerID": ["a", "b"],
        "tenure": [0, 10],
        "TotalCharges": [" ", "800.0"],
        "MonthlyCharges": [50.0, 80.0],
        "Contract": ["Month-to-month", "Two year"],
        "OnlineSecurity": ["Yes", "No"], "OnlineBackup": ["Yes", "No"],
        "DeviceProtection": ["No", "No"], "TechSupport": ["No", "No"],
        "StreamingTV": ["No", "No"], "StreamingMovies": ["No", "No"],
        "Churn": ["Yes", "No"],
    })


def test_clean_data():
    df = clean_data(sample())
    assert "customerID" not in df.columns
    assert df["TotalCharges"].tolist() == [0.0, 800.0]
    assert df["Churn"].tolist() == [1, 0]


def test_add_features():
    df = add_features(clean_data(sample()))
    assert df["n_addons"].tolist() == [2, 0]
    assert df["is_month_to_month"].tolist() == [1, 0]
    assert df["avg_monthly_spend"].tolist() == [0.0, 80.0]


def test_encode_gives_numbers_only():
    data = encode(add_features(clean_data(sample())))
    assert all(pd.api.types.is_numeric_dtype(t) for t in data.dtypes)
