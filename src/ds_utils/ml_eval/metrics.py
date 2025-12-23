"""Metrics calculation for ds_utils ML evaluation.

This module provides summary metrics for classification and regression tasks.
"""

from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    recall_score,
    r2_score,
)

from ds_utils.config.logging_config import get_logger

logger = get_logger("ml_eval.metrics")


def classification_summary(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    average: Literal["binary", "micro", "macro", "weighted"] = "weighted",
    labels: list[Any] | None = None,
    target_names: list[str] | None = None,
    include_per_class: bool = True,
) -> pd.DataFrame:
    """Calculate comprehensive classification metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        average: Averaging method for multi-class metrics.
            - 'binary': Only report results for the class specified by pos_label.
            - 'micro': Calculate metrics globally.
            - 'macro': Calculate metrics for each class and find unweighted mean.
            - 'weighted': Calculate metrics for each class and find weighted mean.
        labels: Optional list of labels to include.
        target_names: Optional names for the classes.
        include_per_class: Whether to include per-class metrics.

    Returns:
        DataFrame with metrics including accuracy, precision, recall, F1.

    Example:
        >>> y_true = [0, 1, 1, 0, 1]
        >>> y_pred = [0, 1, 0, 0, 1]
        >>> metrics = classification_summary(y_true, y_pred)
        >>> print(metrics)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Determine if binary or multi-class
    unique_classes = np.unique(np.concatenate([y_true, y_pred]))
    is_binary = len(unique_classes) == 2

    # Calculate overall metrics
    metrics_data: dict[str, Any] = {
        "Metric": [],
        "Value": [],
    }

    # Accuracy
    acc = accuracy_score(y_true, y_pred)
    metrics_data["Metric"].append("Accuracy")
    metrics_data["Value"].append(acc)

    # Precision, Recall, F1
    avg = "binary" if is_binary else average

    prec = precision_score(y_true, y_pred, average=avg, labels=labels, zero_division=0)
    metrics_data["Metric"].append(f"Precision ({avg})")
    metrics_data["Value"].append(prec)

    rec = recall_score(y_true, y_pred, average=avg, labels=labels, zero_division=0)
    metrics_data["Metric"].append(f"Recall ({avg})")
    metrics_data["Value"].append(rec)

    f1 = f1_score(y_true, y_pred, average=avg, labels=labels, zero_division=0)
    metrics_data["Metric"].append(f"F1 Score ({avg})")
    metrics_data["Value"].append(f1)

    # Sample counts
    metrics_data["Metric"].append("Total Samples")
    metrics_data["Value"].append(len(y_true))

    overall_df = pd.DataFrame(metrics_data)

    # Per-class metrics
    if include_per_class and not is_binary:
        per_class_data: dict[str, list[Any]] = {
            "Class": [],
            "Precision": [],
            "Recall": [],
            "F1 Score": [],
            "Support": [],
        }

        for i, cls in enumerate(unique_classes):
            class_name = target_names[i] if target_names else str(cls)
            per_class_data["Class"].append(class_name)

            # Binary metrics for this class
            y_true_binary = (y_true == cls).astype(int)
            y_pred_binary = (y_pred == cls).astype(int)

            per_class_data["Precision"].append(
                precision_score(y_true_binary, y_pred_binary, zero_division=0)
            )
            per_class_data["Recall"].append(
                recall_score(y_true_binary, y_pred_binary, zero_division=0)
            )
            per_class_data["F1 Score"].append(
                f1_score(y_true_binary, y_pred_binary, zero_division=0)
            )
            per_class_data["Support"].append(np.sum(y_true == cls))

        per_class_df = pd.DataFrame(per_class_data)

        # Return both as a dict-like structure or concatenate
        logger.info("Classification metrics calculated with per-class breakdown")
        return overall_df, per_class_df  # type: ignore

    logger.info("Classification metrics calculated")
    return overall_df


def regression_summary(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    include_residual_stats: bool = True,
) -> pd.DataFrame:
    """Calculate comprehensive regression metrics.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.
        include_residual_stats: Whether to include residual statistics.

    Returns:
        DataFrame with metrics including MAE, RMSE, R², MAPE.

    Example:
        >>> y_true = [1.0, 2.0, 3.0, 4.0, 5.0]
        >>> y_pred = [1.1, 2.2, 2.9, 4.0, 4.8]
        >>> metrics = regression_summary(y_true, y_pred)
        >>> print(metrics)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics_data: dict[str, Any] = {
        "Metric": [],
        "Value": [],
    }

    # Mean Absolute Error
    mae = mean_absolute_error(y_true, y_pred)
    metrics_data["Metric"].append("MAE (Mean Absolute Error)")
    metrics_data["Value"].append(mae)

    # Root Mean Squared Error
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    metrics_data["Metric"].append("RMSE (Root Mean Squared Error)")
    metrics_data["Value"].append(rmse)

    # R² Score
    r2 = r2_score(y_true, y_pred)
    metrics_data["Metric"].append("R² Score")
    metrics_data["Value"].append(r2)

    # MAPE (handle zero values)
    try:
        mape = mean_absolute_percentage_error(y_true, y_pred)
        metrics_data["Metric"].append("MAPE (Mean Absolute Percentage Error)")
        metrics_data["Value"].append(mape)
    except Exception:
        logger.warning("Could not calculate MAPE (possibly due to zero values)")

    # Sample count
    metrics_data["Metric"].append("Total Samples")
    metrics_data["Value"].append(len(y_true))

    # Residual statistics
    if include_residual_stats:
        residuals = y_true - y_pred

        metrics_data["Metric"].append("Residual Mean")
        metrics_data["Value"].append(np.mean(residuals))

        metrics_data["Metric"].append("Residual Std")
        metrics_data["Value"].append(np.std(residuals))

        metrics_data["Metric"].append("Residual Min")
        metrics_data["Value"].append(np.min(residuals))

        metrics_data["Metric"].append("Residual Max")
        metrics_data["Value"].append(np.max(residuals))

    logger.info("Regression metrics calculated")
    return pd.DataFrame(metrics_data)


def cross_validation_summary(
    cv_results: dict[str, np.ndarray],
    metrics: list[str] | None = None,
) -> pd.DataFrame:
    """Summarize cross-validation results.

    Args:
        cv_results: Dictionary from sklearn cross_validate.
        metrics: List of metrics to include. If None, includes all.

    Returns:
        DataFrame with mean, std, min, max for each metric.
    """
    if metrics is None:
        # Get all test metrics
        metrics = [k for k in cv_results.keys() if k.startswith("test_")]

    summary_data: dict[str, list[Any]] = {
        "Metric": [],
        "Mean": [],
        "Std": [],
        "Min": [],
        "Max": [],
    }

    for metric in metrics:
        if metric not in cv_results:
            continue

        values = cv_results[metric]
        clean_name = metric.replace("test_", "").replace("_", " ").title()

        summary_data["Metric"].append(clean_name)
        summary_data["Mean"].append(np.mean(values))
        summary_data["Std"].append(np.std(values))
        summary_data["Min"].append(np.min(values))
        summary_data["Max"].append(np.max(values))

    return pd.DataFrame(summary_data)


def calculate_lift(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray | pd.Series,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Calculate lift for binary classification.

    Args:
        y_true: Binary ground truth labels (0/1).
        y_proba: Predicted probabilities for positive class.
        n_bins: Number of bins (deciles by default).

    Returns:
        DataFrame with lift statistics per bin.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    # Create DataFrame for binning
    df = pd.DataFrame({"y_true": y_true, "y_proba": y_proba})
    df["bin"] = pd.qcut(df["y_proba"], q=n_bins, labels=False, duplicates="drop")

    # Calculate lift per bin
    baseline_rate = y_true.mean()

    lift_data = (
        df.groupby("bin")
        .agg(
            count=("y_true", "count"),
            positive_count=("y_true", "sum"),
            avg_proba=("y_proba", "mean"),
        )
        .reset_index()
    )

    lift_data["response_rate"] = lift_data["positive_count"] / lift_data["count"]
    lift_data["lift"] = lift_data["response_rate"] / baseline_rate
    lift_data["cumulative_positive"] = lift_data["positive_count"].cumsum()
    lift_data["cumulative_total"] = lift_data["count"].cumsum()
    lift_data["cumulative_response_rate"] = (
        lift_data["cumulative_positive"] / lift_data["cumulative_total"]
    )
    lift_data["cumulative_lift"] = lift_data["cumulative_response_rate"] / baseline_rate

    return lift_data
