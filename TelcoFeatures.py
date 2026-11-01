import pandas as pd

ADDONS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
          "TechSupport", "StreamingTV", "StreamingMovies"]


def clean_data(df):
    """Drop the ID, fix TotalCharges, turn Churn into 1/0."""
    df = df.drop(columns="customerID")
    # TotalCharges has a few blank strings (new customers) -> number, blanks become 0
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    return df


def add_features(df):
    """Three simple extra columns."""
    df = df.copy()
    df["n_addons"] = (df[ADDONS] == "Yes").sum(axis=1)
    df["avg_monthly_spend"] = df["TotalCharges"] / df["tenure"].clip(lower=1)
    df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)
    return df


def encode(df):
    """Turn text columns into 0/1 columns."""
    return pd.get_dummies(df, drop_first=True, dtype=int)
