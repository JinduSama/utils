"""Classification visualization for ds_utils ML evaluation.

This module provides visualization tools for classification model evaluation.
"""

from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from ds_utils.config import get_colors
from ds_utils.plotting.core import apply_corporate_style, create_figure

from ds_utils.config.logging_config import get_logger

logger = get_logger("ml_eval.classification")


def plot_confusion_matrix(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    labels: list[str] | None = None,
    normalize: Literal["true", "pred", "all", None] = None,
    title: str = "Confusion Matrix",
    figsize: str = "square",
    cmap: str | None = None,
    show_values: bool = True,
    value_format: str = ".0f",
    ax: Axes | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Plot a confusion matrix with corporate styling.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        labels: Display labels for classes.
        normalize: Normalization method. Options:
            - None: Show absolute counts
            - 'true': Normalize by true labels (rows)
            - 'pred': Normalize by predicted labels (columns)
            - 'all': Normalize by total count
        title: Plot title.
        figsize: Size preset.
        cmap: Colormap name. If None, uses corporate colors.
        show_values: Whether to display values in cells.
        value_format: Format string for values.
        ax: Existing axes to plot on.
        **kwargs: Additional arguments passed to ConfusionMatrixDisplay.

    Returns:
        Tuple of (figure, axes).

    Example:
        >>> y_true = [0, 1, 1, 0, 1, 0]
        >>> y_pred = [0, 1, 0, 0, 1, 1]
        >>> fig, ax = plot_confusion_matrix(y_true, y_pred, normalize='true')
    """
    apply_corporate_style()

    if ax is None:
        fig, ax = create_figure(size=figsize)
    else:
        fig = ax.figure

    if cmap is None:
        cmap = "Blues"

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred, normalize=normalize)

    # Determine format
    if normalize:
        fmt = ".2%"
    else:
        fmt = value_format

    # Create display
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(
        ax=ax,
        cmap=cmap,
        values_format=fmt if show_values else "",
        colorbar=True,
        **kwargs,
    )

    ax.set_title(title)
    plt.tight_layout()

    return fig, ax


def plot_roc_curve(
    y_true: np.ndarray | pd.Series,
    y_score: np.ndarray | pd.Series | dict[str, np.ndarray],
    title: str = "ROC Curve",
    figsize: str = "square",
    colors: list[str] | None = None,
    show_auc: bool = True,
    show_diagonal: bool = True,
    ax: Axes | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Plot ROC curve(s) for binary or multi-class classification.

    Args:
        y_true: Ground truth binary labels or one-hot encoded for multi-class.
        y_score: Predicted probabilities. For multi-class, provide a dict with
            class names as keys and probabilities as values.
        title: Plot title.
        figsize: Size preset.
        colors: Custom color palette.
        show_auc: Whether to show AUC in legend.
        show_diagonal: Whether to show diagonal reference line.
        ax: Existing axes to plot on.
        **kwargs: Additional arguments passed to plot.

    Returns:
        Tuple of (figure, axes).

    Example:
        >>> # Binary classification
        >>> fig, ax = plot_roc_curve(y_true, y_proba)

        >>> # Multi-class
        >>> fig, ax = plot_roc_curve(
        ...     y_true_onehot,
        ...     {'Class A': proba_a, 'Class B': proba_b}
        ... )
    """
    apply_corporate_style()

    if ax is None:
        fig, ax = create_figure(size=figsize)
    else:
        fig = ax.figure

    if colors is None:
        colors = get_colors("categorical")

    if isinstance(y_score, dict):
        # Multi-class ROC curves
        for i, (name, scores) in enumerate(y_score.items()):
            color = colors[i % len(colors)]

            # Get binary labels for this class
            if isinstance(y_true, np.ndarray) and y_true.ndim == 2:
                y_binary = y_true[:, i]
            else:
                y_binary = (np.asarray(y_true) == i).astype(int)

            fpr, tpr, _ = roc_curve(y_binary, scores)
            auc = roc_auc_score(y_binary, scores)

            label = f"{name} (AUC = {auc:.3f})" if show_auc else name
            ax.plot(fpr, tpr, color=color, label=label, linewidth=2, **kwargs)
    else:
        # Binary ROC curve
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)

        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)

        label = f"ROC (AUC = {auc:.3f})" if show_auc else "ROC"
        ax.plot(fpr, tpr, color=colors[0], label=label, linewidth=2, **kwargs)

    # Diagonal reference line
    if show_diagonal:
        ax.plot([0, 1], [0, 1], color="gray", linestyle="--", alpha=0.7, label="Random")

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    return fig, ax


def plot_precision_recall_curve(
    y_true: np.ndarray | pd.Series,
    y_score: np.ndarray | pd.Series | dict[str, np.ndarray],
    title: str = "Precision-Recall Curve",
    figsize: str = "square",
    colors: list[str] | None = None,
    show_ap: bool = True,
    show_baseline: bool = True,
    ax: Axes | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Plot Precision-Recall curve(s).

    Args:
        y_true: Ground truth binary labels.
        y_score: Predicted probabilities.
        title: Plot title.
        figsize: Size preset.
        colors: Custom color palette.
        show_ap: Whether to show Average Precision in legend.
        show_baseline: Whether to show baseline (random classifier).
        ax: Existing axes to plot on.
        **kwargs: Additional arguments.

    Returns:
        Tuple of (figure, axes).
    """
    apply_corporate_style()

    if ax is None:
        fig, ax = create_figure(size=figsize)
    else:
        fig = ax.figure

    if colors is None:
        colors = get_colors("categorical")

    if isinstance(y_score, dict):
        # Multi-class PR curves
        for i, (name, scores) in enumerate(y_score.items()):
            color = colors[i % len(colors)]

            if isinstance(y_true, np.ndarray) and y_true.ndim == 2:
                y_binary = y_true[:, i]
            else:
                y_binary = (np.asarray(y_true) == i).astype(int)

            precision, recall, _ = precision_recall_curve(y_binary, scores)
            ap = average_precision_score(y_binary, scores)

            label = f"{name} (AP = {ap:.3f})" if show_ap else name
            ax.plot(recall, precision, color=color, label=label, linewidth=2, **kwargs)
    else:
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)

        precision, recall, _ = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)

        label = f"PR (AP = {ap:.3f})" if show_ap else "PR"
        ax.plot(recall, precision, color=colors[0], label=label, linewidth=2, **kwargs)

        # Baseline
        if show_baseline:
            baseline = y_true.mean()
            ax.axhline(y=baseline, color="gray", linestyle="--", alpha=0.7,
                      label=f"Baseline ({baseline:.3f})")

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    return fig, ax


def plot_calibration_curve(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray | pd.Series | dict[str, np.ndarray],
    n_bins: int = 10,
    title: str = "Calibration Curve",
    figsize: str = "square",
    colors: list[str] | None = None,
    strategy: Literal["uniform", "quantile"] = "uniform",
    ax: Axes | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Plot calibration curve(s) for probability calibration assessment.

    Args:
        y_true: Ground truth binary labels.
        y_proba: Predicted probabilities. Can be dict for multiple models.
        n_bins: Number of bins for calibration.
        title: Plot title.
        figsize: Size preset.
        colors: Custom color palette.
        strategy: Binning strategy ('uniform' or 'quantile').
        ax: Existing axes to plot on.
        **kwargs: Additional arguments.

    Returns:
        Tuple of (figure, axes).
    """
    apply_corporate_style()

    if ax is None:
        fig, ax = create_figure(size=figsize)
    else:
        fig = ax.figure

    if colors is None:
        colors = get_colors("categorical")

    if isinstance(y_proba, dict):
        # Multiple models
        for i, (name, proba) in enumerate(y_proba.items()):
            color = colors[i % len(colors)]
            prob_true, prob_pred = calibration_curve(
                y_true, proba, n_bins=n_bins, strategy=strategy
            )
            ax.plot(prob_pred, prob_true, color=color, label=name,
                    marker="o", linewidth=2, **kwargs)
    else:
        y_true = np.asarray(y_true)
        y_proba = np.asarray(y_proba)

        prob_true, prob_pred = calibration_curve(
            y_true, y_proba, n_bins=n_bins, strategy=strategy
        )
        ax.plot(prob_pred, prob_true, color=colors[0], label="Model",
                marker="o", linewidth=2, **kwargs)

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", alpha=0.7,
            label="Perfectly Calibrated")

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    return fig, ax


def plot_classification_report(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    labels: list[str] | None = None,
    title: str = "Classification Report",
    figsize: str = "medium",
    cmap: str = "Blues",
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot classification report as a heatmap.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        labels: Class labels.
        title: Plot title.
        figsize: Size preset.
        cmap: Colormap name.
        ax: Existing axes to plot on.

    Returns:
        Tuple of (figure, axes).
    """
    from sklearn.metrics import classification_report

    apply_corporate_style()

    if ax is None:
        fig, ax = create_figure(size=figsize)
    else:
        fig = ax.figure

    # Get classification report as dict
    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)

    # Convert to DataFrame (exclude averages for cleaner view)
    classes = labels if labels else list(set(y_true) | set(y_pred))
    metrics = ["precision", "recall", "f1-score"]

    data = []
    for cls in classes:
        cls_str = str(cls)
        if cls_str in report:
            row = [report[cls_str][m] for m in metrics]
            data.append(row)

    df = pd.DataFrame(data, index=classes, columns=metrics)

    # Plot heatmap
    sns.heatmap(df, annot=True, fmt=".2f", cmap=cmap, ax=ax,
                vmin=0, vmax=1, cbar_kws={"label": "Score"})

    ax.set_title(title)
    ax.set_ylabel("Class")
    ax.set_xlabel("Metric")

    plt.tight_layout()
    return fig, ax
