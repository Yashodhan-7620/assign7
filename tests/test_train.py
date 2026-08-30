"""
Pytest suite for src/train.py.

Run with: pytest -v
"""
import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from src.train import (
    MODEL_PATH,
    SCALER_PATH,
    evaluate_model,
    load_data,
    run_pipeline,
    save_artifacts,
    scale_data,
    split_data,
    train_model,
)


# ---------- fixtures ----------

@pytest.fixture(scope="module")
def data():
    return load_data()


@pytest.fixture(scope="module")
def split(data):
    return split_data(data)


@pytest.fixture(scope="module")
def scaled(split):
    X_train, X_test, y_train, y_test = split
    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)
    return X_train_scaled, X_test_scaled, scaler, y_train, y_test


@pytest.fixture(scope="module")
def trained_model(scaled):
    X_train_scaled, _, _, y_train, _ = scaled
    return train_model(X_train_scaled, y_train)


# ---------- data loading ----------

def test_load_data_shape(data):
    assert isinstance(data, pd.DataFrame)
    assert data.shape[0] == 20640
    assert data.shape[1] == 9  # 8 features + Price


def test_load_data_no_nulls(data):
    assert data.isnull().sum().sum() == 0


def test_load_data_has_price_column(data):
    assert "Price" in data.columns


# ---------- splitting ----------

def test_split_shapes(split, data):
    X_train, X_test, y_train, y_test = split
    assert len(X_train) + len(X_test) == len(data)
    assert X_train.shape[1] == 8
    assert X_test.shape[1] == 8


def test_split_is_reproducible(data):
    s1 = split_data(data, random_state=42)
    s2 = split_data(data, random_state=42)
    pd.testing.assert_frame_equal(s1[0], s2[0])


# ---------- scaling ----------

def test_scaler_output_shape(scaled, split):
    X_train_scaled, X_test_scaled, scaler, _, _ = scaled
    X_train, X_test, _, _ = split
    assert X_train_scaled.shape == X_train.shape
    assert X_test_scaled.shape == X_test.shape
    assert isinstance(scaler, StandardScaler)


def test_scaler_train_mean_near_zero(scaled):
    X_train_scaled, _, _, _, _ = scaled
    assert np.allclose(X_train_scaled.mean(axis=0), 0, atol=1e-6)


# ---------- training ----------

def test_train_model_returns_fitted_regressor(trained_model):
    assert isinstance(trained_model, LinearRegression)
    assert hasattr(trained_model, "coef_")
    assert trained_model.coef_.shape[0] == 8


# ---------- evaluation ----------

def test_evaluate_model_returns_expected_keys(trained_model, scaled):
    _, X_test_scaled, _, _, y_test = scaled
    metrics = evaluate_model(trained_model, X_test_scaled, y_test)
    assert set(metrics.keys()) == {"mae", "mse", "rmse", "r2", "adjusted_r2"}


def test_evaluate_model_reasonable_performance(trained_model, scaled):
    """
    Regression guardrail: fail CI if model quality drops well below
    what plain linear regression achieves on this dataset (~0.55-0.60 R2).
    Adjust the threshold if you intentionally change the model/features.
    """
    _, X_test_scaled, _, _, y_test = scaled
    metrics = evaluate_model(trained_model, X_test_scaled, y_test)
    assert metrics["r2"] > 0.4
    assert metrics["rmse"] < 1.0


# ---------- persistence ----------

def test_save_artifacts_creates_files(trained_model, scaled, tmp_path, monkeypatch):
    _, _, scaler, _, _ = scaled
    fake_model_path = tmp_path / "regression.pkl"
    fake_scaler_path = tmp_path / "scaler.pkl"
    monkeypatch.setattr("src.train.MODEL_PATH", fake_model_path)
    monkeypatch.setattr("src.train.SCALER_PATH", fake_scaler_path)
    monkeypatch.setattr("src.train.MODEL_DIR", tmp_path)

    save_artifacts(trained_model, scaler)

    assert fake_model_path.exists()
    assert fake_scaler_path.exists()

    with open(fake_model_path, "rb") as f:
        loaded_model = pickle.load(f)
    assert isinstance(loaded_model, LinearRegression)


# ---------- end-to-end ----------

def test_run_pipeline_end_to_end():
    metrics = run_pipeline()
    assert metrics["r2"] > 0.4
    assert MODEL_PATH.exists()
    assert SCALER_PATH.exists()
