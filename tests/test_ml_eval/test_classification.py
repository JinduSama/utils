"""Tests for classification visualization."""

import matplotlib.pyplot as plt
import pytest

from ds_utils.ml_eval.classification import (
    plot_calibration_curve,
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_roc_curve,
)


@pytest.fixture(autouse=True)
def close_figures():
    """Close all figures after each test."""
    yield
    plt.close("all")


class TestPlotConfusionMatrix:
    """Tests for plot_confusion_matrix function."""

    def test_basic_confusion_matrix(self, classification_data):
        """Test creating a basic confusion matrix."""
        y_true, y_pred, _ = classification_data
        fig, ax = plot_confusion_matrix(y_true, y_pred)

        assert fig is not None
        assert ax is not None

    def test_normalized_confusion_matrix(self, classification_data):
        """Test normalized confusion matrix."""
        y_true, y_pred, _ = classification_data
        fig, ax = plot_confusion_matrix(y_true, y_pred, normalize="true")

        assert fig is not None

    def test_confusion_matrix_with_labels(self, classification_data):
        """Test confusion matrix with custom labels."""
        y_true, y_pred, _ = classification_data
        fig, ax = plot_confusion_matrix(y_true, y_pred, labels=["Negative", "Positive"])

        assert fig is not None


class TestPlotROCCurve:
    """Tests for plot_roc_curve function."""

    def test_basic_roc_curve(self, classification_data):
        """Test creating a basic ROC curve."""
        y_true, _, y_proba = classification_data
        fig, ax = plot_roc_curve(y_true, y_proba)

        assert fig is not None
        assert ax is not None
        assert len(ax.lines) >= 2  # ROC curve + diagonal

    def test_roc_curve_shows_auc(self, classification_data):
        """Test that ROC curve shows AUC in legend."""
        y_true, _, y_proba = classification_data
        fig, ax = plot_roc_curve(y_true, y_proba, show_auc=True)

        legend = ax.get_legend()
        assert legend is not None

    def test_roc_curve_dict_binary_treated_as_models(self, classification_data):
        """Test that dict input with binary y_true is treated as multiple models."""
        y_true, _, y_proba = classification_data
        model_scores = {
            "Model A": y_proba,
            "Model B": 1 - y_proba,
        }

        fig, ax = plot_roc_curve(y_true, model_scores)

        assert fig is not None
        assert ax is not None
        # Two model curves + diagonal reference
        assert len(ax.lines) >= 3


class TestPlotPrecisionRecallCurve:
    """Tests for plot_precision_recall_curve function."""

    def test_basic_pr_curve(self, classification_data):
        """Test creating a basic PR curve."""
        y_true, _, y_proba = classification_data
        fig, ax = plot_precision_recall_curve(y_true, y_proba)

        assert fig is not None
        assert ax is not None

    def test_pr_curve_with_baseline(self, classification_data):
        """Test PR curve with baseline."""
        y_true, _, y_proba = classification_data
        fig, ax = plot_precision_recall_curve(y_true, y_proba, show_baseline=True)

        assert len(ax.lines) >= 2

    def test_pr_curve_dict_binary_treated_as_models(self, classification_data):
        """Test that dict input with binary y_true is treated as multiple models."""
        y_true, _, y_proba = classification_data
        model_scores = {
            "Model A": y_proba,
            "Model B": 1 - y_proba,
        }

        fig, ax = plot_precision_recall_curve(y_true, model_scores, show_baseline=True)

        assert fig is not None
        assert ax is not None
        # Two model curves + baseline line
        assert len(ax.lines) >= 3


class TestPlotCalibrationCurve:
    """Tests for plot_calibration_curve function."""

    def test_basic_calibration_curve(self, classification_data):
        """Test creating a basic calibration curve."""
        y_true, _, y_proba = classification_data
        fig, ax = plot_calibration_curve(y_true, y_proba)

        assert fig is not None
        assert ax is not None
