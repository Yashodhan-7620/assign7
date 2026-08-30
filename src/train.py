"""
Training pipeline for the California Housing regression model.

Refactored from the original notebook. Key fixes vs. the notebook:
  - test_model.predict() was called with no input -> now predicts on X_test_scaled
  - scaler is now pickled alongside the model (you can't reuse a fitted
    StandardScaler at inference time otherwise)
  - everything wrapped in functions so it's importable + testable with pytest
"""
import pickle
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODEL_DIR / "regression.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"


def load_data() -> pd.DataFrame:
    """Load the California housing dataset into a single DataFrame."""
    dataset = fetch_california_housing()
    data = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    data["Price"] = dataset.target
    return data


def split_data(data: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    X = data.iloc[:, :8]
    y = data.iloc[:, -1]
    return train_test_split(X, y, random_state=random_state, test_size=test_size)


def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def train_model(X_train_scaled, y_train) -> LinearRegression:
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    return model


def evaluate_model(model: LinearRegression, X_test_scaled, y_test) -> dict:
    y_pred = model.predict(X_test_scaled)
    n, p = X_test_scaled.shape[0], X_test_scaled.shape[1]
    r2 = r2_score(y_test, y_pred)
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    return {
        "mae": mean_absolute_error(y_test, y_pred),
        "mse": mean_squared_error(y_test, y_pred),
        "rmse": root_mean_squared_error(y_test, y_pred),
        "r2": r2,
        "adjusted_r2": adjusted_r2,
    }


def save_artifacts(model: LinearRegression, scaler: StandardScaler) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)


def run_pipeline() -> dict:
    """Run the full pipeline end to end and return the evaluation metrics."""
    data = load_data()
    X_train, X_test, y_train, y_test = split_data(data)
    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)
    model = train_model(X_train_scaled, y_train)
    metrics = evaluate_model(model, X_test_scaled, y_test)
    save_artifacts(model, scaler)
    return metrics


if __name__ == "__main__":
    metrics = run_pipeline()
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")