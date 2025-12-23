"""Shared test fixtures and configuration."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        "numeric": np.random.randn(100),
        "category": np.random.choice(["A", "B", "C"], 100),
        "date": pd.date_range("2024-01-01", periods=100),
        "value": np.random.randint(0, 100, 100),
    })


@pytest.fixture
def classification_data():
    """Create sample classification data."""
    np.random.seed(42)
    y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 1, 1, 1, 0, 0])
    y_proba = np.array([0.1, 0.9, 0.4, 0.2, 0.8, 0.6, 0.7, 0.85, 0.15, 0.25])
    return y_true, y_pred, y_proba


@pytest.fixture
def regression_data():
    """Create sample regression data."""
    np.random.seed(42)
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    y_pred = np.array([1.1, 2.2, 2.9, 4.0, 4.8, 6.1, 7.2, 7.8, 9.1, 10.2])
    return y_true, y_pred


@pytest.fixture
def german_formatted_dataframe():
    """Create a DataFrame with German-formatted numbers."""
    return pd.DataFrame({
        "price": ["1.234,56", "789,12", "2.345.678,90"],
        "date": ["25.12.2024", "01.01.2025", "15.06.2024"],
    })


@pytest.fixture
def dataframe_with_missing():
    """Create a DataFrame with missing values."""
    return pd.DataFrame({
        "complete": [1, 2, 3, 4, 5],
        "with_missing": [1, None, 3, None, 5],
        "all_missing": [None, None, None, None, None],
    })


@pytest.fixture
def dataframe_with_outliers():
    """Create a DataFrame with outliers."""
    return pd.DataFrame({
        "normal": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "with_outlier": [1, 2, 3, 4, 5, 6, 7, 8, 9, 100],
    })
