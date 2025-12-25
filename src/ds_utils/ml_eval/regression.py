"""Regression visualization for ds_utils ML evaluation.

This module provides visualization tools for regression model evaluation.
"""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy import stats

from ds_utils.config import get_colors
from ds_utils.config.logging_config import get_logger
from ds_utils.plotting.core import apply_corporate_style, create_figure

logger = get_logger("ml_eval.regression")


def plot_residuals(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    title: str = "Residuals vs Predicted",
    figsize: str = "medium",
    color: str | None = None,
    alpha: float = 0.6,
    show_zero_line: bool = True,
    show_loess: bool = False,
    ax: Axes | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Plot residuals against predicted values.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.
        title: Plot title.
        figsize: Size preset.
        color: Point color.
        alpha: Point transparency.
        show_zero_line: Whether to show horizontal line at y=0.
        show_loess: Whether to show LOESS smoothing curve.
        ax: Existing axes to plot on.
        **kwargs: Additional arguments passed to scatter.

    Returns:
        Tuple of (figure, axes).
    """
    apply_corporate_style()

    if ax is None:
        fig, ax = create_figure(size=figsize)
    else:
        fig = ax.figure

    colors = get_colors("categorical")
    if color is None:
        color = colors[0]

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    residuals = y_true - y_pred

    # Scatter plot
    ax.scatter(y_pred, residuals, color=color, alpha=alpha, **kwargs)

    # Zero line
    if show_zero_line:
        ax.axhline(y=0, color="red", linestyle="--", linewidth=2, alpha=0.7)

    # LOESS smoothing
    if show_loess:
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess

            smoothed = lowess(residuals, y_pred, frac=0.3)
            ax.plot(
                smoothed[:, 0],
                smoothed[:, 1],
                color=colors[1],
                linewidth=2,
                label="LOESS",
            )
            ax.legend()
        except ImportError:
            logger.warning("statsmodels not installed. Skipping LOESS curve.")

    ax.set_xlabel("Predicted Values")
    ax.set_ylabel("Residuals")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    return fig, ax


def plot_prediction_error(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    title: str = "Actual vs Predicted",
    figsize: str = "square",
    color: str | None = None,
    alpha: float = 0.6,
    show_identity: bool = True,
    show_r2: bool = True,
    ax: Axes | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Plot actual vs predicted values.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.
        title: Plot title.
        figsize: Size preset.
        color: Point color.
        alpha: Point transparency.
        show_identity: Whether to show identity line (perfect prediction).
        show_r2: Whether to show R² in legend.
        ax: Existing axes to plot on.
        **kwargs: Additional arguments passed to scatter.

    Returns:
        Tuple of (figure, axes).
    """
    apply_corporate_style()

    if ax is None:
        fig, ax = create_figure(size=figsize)
    else:
        fig = ax.figure

    colors = get_colors("categorical")
    if color is None:
        color = colors[0]

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Calculate R²
    from sklearn.metrics import r2_score

    r2 = r2_score(y_true, y_pred)

    # Scatter plot
    label = f"Predictions (R² = {r2:.3f})" if show_r2 else "Predictions"
    ax.scatter(y_true, y_pred, color=color, alpha=alpha, label=label, **kwargs)

    # Identity line
    if show_identity:
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot(
            [min_val, max_val],
            [min_val, max_val],
            color="red",
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            label="Perfect Prediction",
        )

    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    return fig, ax


def plot_residual_distribution(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    title: str = "Residual Distribution",
    figsize: str = "large",
    color: str | None = None,
    show_qq: bool = True,
    **kwargs: Any,
) -> tuple[Figure, list[Axes]]:
    """Plot residual distribution with histogram and optional Q-Q plot.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.
        title: Overall title.
        figsize: Size preset.
        color: Histogram color.
        show_qq: Whether to include Q-Q plot.
        **kwargs: Additional arguments.

    Returns:
        Tuple of (figure, list of axes).
    """
    apply_corporate_style()

    ncols = 2 if show_qq else 1
    fig, axes = create_figure(size=figsize, ncols=ncols)

    if not isinstance(axes, np.ndarray):
        axes = [axes]
    else:
        axes = list(axes)

    colors = get_colors("categorical")
    if color is None:
        color = colors[0]

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    residuals = y_true - y_pred

    # Histogram
    ax_hist = axes[0]

    import seaborn as sns

    sns.histplot(residuals, kde=True, color=color, ax=ax_hist, **kwargs)

    # Add normal distribution overlay
    mu, std = residuals.mean(), residuals.std()
    x = np.linspace(residuals.min(), residuals.max(), 100)
    y = stats.norm.pdf(x, mu, std)

    # Scale to histogram
    ax_hist_twin = ax_hist.twinx()
    ax_hist_twin.plot(
        x, y, color=colors[1], linewidth=2, label=f"Normal (μ={mu:.2f}, σ={std:.2f})"
    )
    ax_hist_twin.set_ylabel("")
    ax_hist_twin.set_yticks([])

    ax_hist.set_xlabel("Residual")
    ax_hist.set_ylabel("Frequency")
    ax_hist.set_title("Residual Histogram")

    # Add statistics
    stats_text = f"Mean: {mu:.3f}\nStd: {std:.3f}\nSkew: {stats.skew(residuals):.3f}"
    ax_hist.text(
        0.95,
        0.95,
        stats_text,
        transform=ax_hist.transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )

    # Q-Q plot
    if show_qq:
        ax_qq = axes[1]
        stats.probplot(residuals, dist="norm", plot=ax_qq)
        ax_qq.get_lines()[0].set_markerfacecolor(color)
        ax_qq.get_lines()[0].set_markeredgecolor(color)
        ax_qq.set_title("Q-Q Plot")
        ax_qq.grid(True, linestyle="--", alpha=0.3)

    fig.suptitle(title, y=1.02)
    plt.tight_layout()

    return fig, axes


def plot_residuals_vs_feature(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    feature: np.ndarray | pd.Series,
    feature_name: str = "Feature",
    title: str | None = None,
    figsize: str = "medium",
    color: str | None = None,
    alpha: float = 0.6,
    show_loess: bool = True,
    ax: Axes | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Plot residuals against a specific feature.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.
        feature: Feature values to plot against.
        feature_name: Name of the feature for axis label.
        title: Plot title.
        figsize: Size preset.
        color: Point color.
        alpha: Point transparency.
        show_loess: Whether to show LOESS curve.
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

    colors = get_colors("categorical")
    if color is None:
        color = colors[0]

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    feature = np.asarray(feature)
    residuals = y_true - y_pred

    ax.scatter(feature, residuals, color=color, alpha=alpha, **kwargs)
    ax.axhline(y=0, color="red", linestyle="--", linewidth=2, alpha=0.7)

    if show_loess:
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess

            # Sort for proper line plotting
            sorted_idx = np.argsort(feature)
            smoothed = lowess(residuals[sorted_idx], feature[sorted_idx], frac=0.3)
            ax.plot(
                smoothed[:, 0],
                smoothed[:, 1],
                color=colors[1],
                linewidth=2,
                label="LOESS",
            )
            ax.legend()
        except ImportError:
            logger.warning("statsmodels not installed. Skipping LOESS curve.")

    if title is None:
        title = f"Residuals vs {feature_name}"

    ax.set_xlabel(feature_name)
    ax.set_ylabel("Residuals")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    return fig, ax
