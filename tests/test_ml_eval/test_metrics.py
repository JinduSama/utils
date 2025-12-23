"""Tests for ML evaluation metrics."""

import numpy as np
import pandas as pd
import pytest

from ds_utils.ml_eval.metrics import (
    calculate_lift,
    classification_summary,
    cross_validation_summary,
    regression_summary,
)


class TestClassificationSummary:
    """Tests for classification_summary function."""

    def test_binary_classification(self, classification_data):
        """Test binary classification metrics."""
        y_true, y_pred, _ = classification_data
        result = classification_summary(y_true, y_pred)

        assert isinstance(result, pd.DataFrame)
        assert "Metric" in result.columns
        assert "Value" in result.columns
        assert "Accuracy" in result["Metric"].values

    def test_multi_class_classification(self):
        """Test multi-class classification metrics."""
        y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 1, 0, 2, 2, 1, 1, 2])

        overall, per_class = classification_summary(y_true, y_pred)

        assert isinstance(overall, pd.DataFrame)
        assert isinstance(per_class, pd.DataFrame)
        assert len(per_class) == 3  # 3 classes

    def test_perfect_classification(self):
        """Test perfect classification results in accuracy of 1.0."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])

        result = classification_summary(y_true, y_pred)
        accuracy_row = result[result["Metric"] == "Accuracy"]

        assert float(accuracy_row["Value"].values[0]) == 1.0


class TestRegressionSummary:
    """Tests for regression_summary function."""

    def test_basic_regression_metrics(self, regression_data):
        """Test basic regression metrics calculation."""
        y_true, y_pred = regression_data
        result = regression_summary(y_true, y_pred)

        assert isinstance(result, pd.DataFrame)
        assert "MAE (Mean Absolute Error)" in result["Metric"].values
        assert "RMSE (Root Mean Squared Error)" in result["Metric"].values
        assert "R² Score" in result["Metric"].values

    def test_perfect_regression(self):
        """Test perfect regression results in R² of 1.0."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = regression_summary(y_true, y_pred)
        r2_row = result[result["Metric"] == "R² Score"]

        assert float(r2_row["Value"].values[0]) == pytest.approx(1.0)

    def test_residual_stats_included(self, regression_data):
        """Test that residual statistics are included."""
        y_true, y_pred = regression_data
        result = regression_summary(y_true, y_pred, include_residual_stats=True)

        assert "Residual Mean" in result["Metric"].values
        assert "Residual Std" in result["Metric"].values


class TestCrossValidationSummary:
    """Tests for cross_validation_summary function."""

    def test_cv_summary(self):
        """Test cross-validation summary calculation."""
        cv_results = {
            "test_accuracy": np.array([0.8, 0.85, 0.82, 0.88, 0.84]),
            "test_f1": np.array([0.75, 0.80, 0.78, 0.82, 0.79]),
        }

        result = cross_validation_summary(cv_results)

        assert isinstance(result, pd.DataFrame)
        assert "Mean" in result.columns
        assert "Std" in result.columns
        assert len(result) == 2


class TestCalculateLift:
    """Tests for calculate_lift function."""

    def test_lift_calculation(self, classification_data):
        """Test lift calculation."""
        y_true, _, y_proba = classification_data
        result = calculate_lift(y_true, y_proba, n_bins=3)

        assert isinstance(result, pd.DataFrame)
        assert "lift" in result.columns
        assert "cumulative_lift" in result.columns
