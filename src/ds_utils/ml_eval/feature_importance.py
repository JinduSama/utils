"""Feature importance visualization for ds_utils ML evaluation.

This module provides visualization tools for feature importance analysis.
"""

from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ds_utils.config import get_colors
from ds_utils.plotting.core import apply_corporate_style, create_figure

from ds_utils.config.logging_config import get_logger

logger = get_logger("ml_eval.feature_importance")


def plot_feature_importance(
    importances: np.ndarray | pd.Series | dict[str, float],
    feature_names: list[str] | None = None,
    top_n: int | None = 20,
    title: str = "Feature Importance",
    xlabel: str = "Importance",
    figsize: str = "medium",
    color: str | None = None,
    orientation: Literal["horizontal", "vertical"] = "horizontal",
    show_values: bool = True,
    value_format: str = ".3f",
    sort: bool = True,
    ax: Axes | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Plot feature importance as horizontal bar chart.

    Args:
        importances: Feature importance values. Can be array, Series, or dict.
        feature_names: Names of features. Required if importances is array.
        top_n: Number of top features to show. If None, shows all.
        title: Plot title.
        xlabel: X-axis label.
        figsize: Size preset.
        color: Bar color.
        orientation: 'horizontal' or 'vertical'.
        show_values: Whether to show values on bars.
        value_format: Format string for values.
        sort: Whether to sort by importance.
        ax: Existing axes to plot on.
        **kwargs: Additional arguments passed to barh/bar.

    Returns:
        Tuple of (figure, axes).

    Example:
        >>> # From sklearn model
        >>> importances = model.feature_importances_
        >>> fig, ax = plot_feature_importance(importances, feature_names)

        >>> # From dict
        >>> importances = {'feature_a': 0.5, 'feature_b': 0.3}
        >>> fig, ax = plot_feature_importance(importances)
    """
    apply_corporate_style()

    if ax is None:
        fig, ax = create_figure(size=figsize)
    else:
        fig = ax.figure

    colors = get_colors("categorical")
    if color is None:
        color = colors[0]

    # Convert to DataFrame for easier manipulation
    if isinstance(importances, dict):
        df = pd.DataFrame({
            "feature": list(importances.keys()),
            "importance": list(importances.values()),
        })
    elif isinstance(importances, pd.Series):
        df = pd.DataFrame({
            "feature": importances.index,
            "importance": importances.values,
        })
    else:
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(len(importances))]
        df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances,
        })

    # Sort by importance
    if sort:
        df = df.sort_values("importance", ascending=False)

    # Limit to top_n
    if top_n is not None and len(df) > top_n:
        df = df.head(top_n)

    # Reverse for horizontal bar (so most important is at top)
    if orientation == "horizontal":
        df = df.iloc[::-1]

    # Plot
    if orientation == "horizontal":
        bars = ax.barh(df["feature"], df["importance"], color=color, **kwargs)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Feature")

        if show_values:
            for bar, val in zip(bars, df["importance"]):
                ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                        f"{val:{value_format}}", va="center", fontsize=9)
    else:
        bars = ax.bar(df["feature"], df["importance"], color=color, **kwargs)
        ax.set_ylabel(xlabel)
        ax.set_xlabel("Feature")
        plt.xticks(rotation=45, ha="right")

        if show_values:
            for bar, val in zip(bars, df["importance"]):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:{value_format}}", ha="center", fontsize=9)

    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.3,
            axis="x" if orientation == "horizontal" else "y")

    plt.tight_layout()
    return fig, ax


def plot_permutation_importance(
    result: Any,  # sklearn PermutationImportance result
    feature_names: list[str],
    top_n: int | None = 20,
    title: str = "Permutation Importance",
    figsize: str = "medium",
    color: str | None = None,
    show_std: bool = True,
    ax: Axes | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Plot permutation importance with error bars.

    Args:
        result: Result from sklearn.inspection.permutation_importance.
        feature_names: Names of features.
        top_n: Number of top features to show.
        title: Plot title.
        figsize: Size preset.
        color: Bar color.
        show_std: Whether to show standard deviation as error bars.
        ax: Existing axes to plot on.
        **kwargs: Additional arguments.

    Returns:
        Tuple of (figure, axes).

    Example:
        >>> from sklearn.inspection import permutation_importance
        >>> result = permutation_importance(model, X_test, y_test)
        >>> fig, ax = plot_permutation_importance(result, feature_names)
    """
    apply_corporate_style()

    if ax is None:
        fig, ax = create_figure(size=figsize)
    else:
        fig = ax.figure

    colors = get_colors("categorical")
    if color is None:
        color = colors[0]

    # Create DataFrame
    df = pd.DataFrame({
        "feature": feature_names,
        "importance": result.importances_mean,
        "std": result.importances_std,
    })

    # Sort and limit
    df = df.sort_values("importance", ascending=False)
    if top_n is not None and len(df) > top_n:
        df = df.head(top_n)

    # Reverse for horizontal bar
    df = df.iloc[::-1]

    # Plot with error bars
    if show_std:
        ax.barh(df["feature"], df["importance"], xerr=df["std"],
                color=color, capsize=3, **kwargs)
    else:
        ax.barh(df["feature"], df["importance"], color=color, **kwargs)

    ax.set_xlabel("Importance (decrease in score)")
    ax.set_ylabel("Feature")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.3, axis="x")

    plt.tight_layout()
    return fig, ax


def plot_shap_summary(
    shap_values: np.ndarray,
    features: pd.DataFrame | np.ndarray,
    feature_names: list[str] | None = None,
    top_n: int | None = 20,
    title: str = "SHAP Feature Importance",
    figsize: str = "medium",
    plot_type: Literal["bar", "beeswarm", "violin"] = "bar",
    ax: Axes | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Plot SHAP-based feature importance.

    Args:
        shap_values: SHAP values array from shap.Explainer.
        features: Feature values used for coloring (beeswarm/violin).
        feature_names: Names of features.
        top_n: Number of top features to show.
        title: Plot title.
        figsize: Size preset.
        plot_type: Type of plot ('bar', 'beeswarm', 'violin').
        ax: Existing axes to plot on.
        **kwargs: Additional arguments.

    Returns:
        Tuple of (figure, axes).

    Example:
        >>> import shap
        >>> explainer = shap.TreeExplainer(model)
        >>> shap_values = explainer.shap_values(X_test)
        >>> fig, ax = plot_shap_summary(shap_values, X_test)
    """
    try:
        import shap
    except ImportError:
        raise ImportError("shap package is required for SHAP plots. Install with: pip install shap")

    apply_corporate_style()

    if ax is None:
        fig, ax = create_figure(size=figsize)
    else:
        fig = ax.figure

    # Convert features to DataFrame if needed
    if isinstance(features, np.ndarray):
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(features.shape[1])]
        features = pd.DataFrame(features, columns=feature_names)

    # Use SHAP's built-in plotting
    plt.sca(ax)

    if plot_type == "bar":
        # Mean absolute SHAP values
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        plot_feature_importance(
            mean_abs_shap,
            feature_names=list(features.columns),
            top_n=top_n,
            title=title,
            xlabel="Mean |SHAP Value|",
            ax=ax,
            **kwargs,
        )
    else:
        # Use SHAP's native plotting for beeswarm/violin
        shap.summary_plot(
            shap_values,
            features,
            max_display=top_n or 20,
            plot_type="dot" if plot_type == "beeswarm" else "violin",
            show=False,
            **kwargs,
        )
        ax.set_title(title)

    plt.tight_layout()
    return fig, ax


def plot_feature_importance_comparison(
    importance_dict: dict[str, dict[str, float]],
    top_n: int | None = 15,
    title: str = "Feature Importance Comparison",
    figsize: str = "large",
    ax: Axes | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Compare feature importance from multiple methods or models.

    Args:
        importance_dict: Dictionary with method names as keys and
            {feature: importance} dicts as values.
        top_n: Number of top features to show.
        title: Plot title.
        figsize: Size preset.
        ax: Existing axes to plot on.
        **kwargs: Additional arguments.

    Returns:
        Tuple of (figure, axes).

    Example:
        >>> importance_dict = {
        ...     'Random Forest': {'feat_a': 0.5, 'feat_b': 0.3},
        ...     'XGBoost': {'feat_a': 0.4, 'feat_b': 0.4},
        ... }
        >>> fig, ax = plot_feature_importance_comparison(importance_dict)
    """
    apply_corporate_style()

    if ax is None:
        fig, ax = create_figure(size=figsize)
    else:
        fig = ax.figure

    colors = get_colors("categorical")

    # Combine all features
    all_features = set()
    for method_importances in importance_dict.values():
        all_features.update(method_importances.keys())

    # Create DataFrame
    data = []
    for method, importances in importance_dict.items():
        for feature in all_features:
            data.append({
                "Method": method,
                "Feature": feature,
                "Importance": importances.get(feature, 0),
            })

    df = pd.DataFrame(data)

    # Get top features by mean importance
    mean_importance = df.groupby("Feature")["Importance"].mean().sort_values(ascending=False)
    if top_n is not None:
        top_features = mean_importance.head(top_n).index.tolist()
        df = df[df["Feature"].isin(top_features)]

    # Pivot for grouped bar chart
    pivot_df = df.pivot(index="Feature", columns="Method", values="Importance")
    pivot_df = pivot_df.reindex(mean_importance.head(top_n).index[::-1])

    # Plot
    pivot_df.plot(kind="barh", ax=ax, color=colors[:len(importance_dict)], **kwargs)

    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.set_title(title)
    ax.legend(title="Method", loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.3, axis="x")

    plt.tight_layout()
    return fig, ax
