"""ML Evaluation module for ds_utils.

This module provides comprehensive tools for evaluating machine learning models,
including metrics calculation, classification and regression visualizations,
feature importance analysis, and learning curves.
"""

from ds_utils.ml_eval.classification import (
    plot_calibration_curve,
    plot_classification_report,
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_roc_curve,
)
from ds_utils.ml_eval.feature_importance import (
    plot_feature_importance,
    plot_feature_importance_comparison,
    plot_permutation_importance,
    plot_shap_summary,
)
from ds_utils.ml_eval.learning_curves import (
    plot_cv_comparison,
    plot_cv_results,
    plot_learning_curve,
    plot_validation_curve,
)
from ds_utils.ml_eval.metrics import (
    calculate_lift,
    classification_summary,
    cross_validation_summary,
    regression_summary,
)
from ds_utils.ml_eval.regression import (
    plot_prediction_error,
    plot_residual_distribution,
    plot_residuals,
    plot_residuals_vs_feature,
)

__all__ = [
    # Metrics
    "classification_summary",
    "regression_summary",
    "cross_validation_summary",
    "calculate_lift",
    # Classification plots
    "plot_confusion_matrix",
    "plot_roc_curve",
    "plot_precision_recall_curve",
    "plot_calibration_curve",
    "plot_classification_report",
    # Regression plots
    "plot_residuals",
    "plot_prediction_error",
    "plot_residual_distribution",
    "plot_residuals_vs_feature",
    # Feature importance
    "plot_feature_importance",
    "plot_permutation_importance",
    "plot_shap_summary",
    "plot_feature_importance_comparison",
    # Learning curves
    "plot_learning_curve",
    "plot_validation_curve",
    "plot_cv_results",
    "plot_cv_comparison",
]
