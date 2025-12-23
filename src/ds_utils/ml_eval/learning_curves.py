"""Learning curves visualization for ds_utils ML evaluation.

This module provides visualization tools for learning and validation curves.
"""

from typing import Any, Callable, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.model_selection import learning_curve, validation_curve

from ds_utils.config import get_colors
from ds_utils.plotting.core import apply_corporate_style, create_figure

from ds_utils.config.logging_config import get_logger

logger = get_logger("ml_eval.learning_curves")


def plot_learning_curve(
    estimator: Any,
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    cv: int | Any = 5,
    train_sizes: np.ndarray | None = None,
    scoring: str | Callable | None = None,
    title: str = "Learning Curve",
    figsize: str = "medium",
    colors: list[str] | None = None,
    show_std: bool = True,
    n_jobs: int | None = -1,
    ax: Axes | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Plot learning curve showing model performance vs training size.

    Args:
        estimator: Scikit-learn estimator.
        X: Training features.
        y: Training targets.
        cv: Cross-validation strategy.
        train_sizes: Relative or absolute training sizes to use.
        scoring: Scoring metric.
        title: Plot title.
        figsize: Size preset.
        colors: Custom color palette.
        show_std: Whether to show standard deviation bands.
        n_jobs: Number of parallel jobs.
        ax: Existing axes to plot on.
        **kwargs: Additional arguments passed to learning_curve.

    Returns:
        Tuple of (figure, axes).

    Example:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> model = RandomForestClassifier()
        >>> fig, ax = plot_learning_curve(model, X_train, y_train)
    """
    apply_corporate_style()

    if ax is None:
        fig, ax = create_figure(size=figsize)
    else:
        fig = ax.figure

    if colors is None:
        colors = get_colors("categorical")

    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 10)

    # Calculate learning curve
    train_sizes_abs, train_scores, test_scores = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        train_sizes=train_sizes,
        scoring=scoring,
        n_jobs=n_jobs,
        **kwargs,
    )

    # Calculate means and stds
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    test_mean = test_scores.mean(axis=1)
    test_std = test_scores.std(axis=1)

    # Plot training scores
    ax.plot(train_sizes_abs, train_mean, color=colors[0], marker="o",
            linewidth=2, label="Training Score")
    if show_std:
        ax.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std,
                        color=colors[0], alpha=0.2)

    # Plot validation scores
    ax.plot(train_sizes_abs, test_mean, color=colors[1], marker="o",
            linewidth=2, label="Validation Score")
    if show_std:
        ax.fill_between(train_sizes_abs, test_mean - test_std, test_mean + test_std,
                        color=colors[1], alpha=0.2)

    ax.set_xlabel("Training Set Size")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    return fig, ax


def plot_validation_curve(
    estimator: Any,
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    param_name: str,
    param_range: np.ndarray | list,
    cv: int | Any = 5,
    scoring: str | Callable | None = None,
    title: str | None = None,
    figsize: str = "medium",
    colors: list[str] | None = None,
    show_std: bool = True,
    log_scale: bool = False,
    n_jobs: int | None = -1,
    ax: Axes | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Plot validation curve showing model performance vs hyperparameter.

    Args:
        estimator: Scikit-learn estimator.
        X: Training features.
        y: Training targets.
        param_name: Name of hyperparameter to vary.
        param_range: Values of hyperparameter to try.
        cv: Cross-validation strategy.
        scoring: Scoring metric.
        title: Plot title.
        figsize: Size preset.
        colors: Custom color palette.
        show_std: Whether to show standard deviation bands.
        log_scale: Whether to use log scale for x-axis.
        n_jobs: Number of parallel jobs.
        ax: Existing axes to plot on.
        **kwargs: Additional arguments passed to validation_curve.

    Returns:
        Tuple of (figure, axes).

    Example:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> model = RandomForestClassifier()
        >>> fig, ax = plot_validation_curve(
        ...     model, X, y,
        ...     param_name='n_estimators',
        ...     param_range=[10, 50, 100, 200]
        ... )
    """
    apply_corporate_style()

    if ax is None:
        fig, ax = create_figure(size=figsize)
    else:
        fig = ax.figure

    if colors is None:
        colors = get_colors("categorical")

    # Calculate validation curve
    train_scores, test_scores = validation_curve(
        estimator,
        X,
        y,
        param_name=param_name,
        param_range=param_range,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        **kwargs,
    )

    # Calculate means and stds
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    test_mean = test_scores.mean(axis=1)
    test_std = test_scores.std(axis=1)

    # Plot
    if log_scale:
        ax.semilogx(param_range, train_mean, color=colors[0], marker="o",
                    linewidth=2, label="Training Score")
        ax.semilogx(param_range, test_mean, color=colors[1], marker="o",
                    linewidth=2, label="Validation Score")
    else:
        ax.plot(param_range, train_mean, color=colors[0], marker="o",
                linewidth=2, label="Training Score")
        ax.plot(param_range, test_mean, color=colors[1], marker="o",
                linewidth=2, label="Validation Score")

    if show_std:
        ax.fill_between(param_range, train_mean - train_std, train_mean + train_std,
                        color=colors[0], alpha=0.2)
        ax.fill_between(param_range, test_mean - test_std, test_mean + test_std,
                        color=colors[1], alpha=0.2)

    if title is None:
        title = f"Validation Curve ({param_name})"

    ax.set_xlabel(param_name)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    return fig, ax


def plot_cv_results(
    cv_results: dict[str, Any],
    param_name: str,
    scoring: str = "mean_test_score",
    title: str | None = None,
    figsize: str = "medium",
    colors: list[str] | None = None,
    show_std: bool = True,
    log_scale: bool = False,
    ax: Axes | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Plot results from GridSearchCV or RandomizedSearchCV.

    Args:
        cv_results: cv_results_ attribute from sklearn search.
        param_name: Name of parameter to plot on x-axis.
        scoring: Score column to plot.
        title: Plot title.
        figsize: Size preset.
        colors: Custom color palette.
        show_std: Whether to show standard deviation.
        log_scale: Whether to use log scale.
        ax: Existing axes to plot on.
        **kwargs: Additional arguments.

    Returns:
        Tuple of (figure, axes).

    Example:
        >>> from sklearn.model_selection import GridSearchCV
        >>> grid = GridSearchCV(model, param_grid, cv=5)
        >>> grid.fit(X, y)
        >>> fig, ax = plot_cv_results(grid.cv_results_, 'param_C')
    """
    apply_corporate_style()

    if ax is None:
        fig, ax = create_figure(size=figsize)
    else:
        fig = ax.figure

    if colors is None:
        colors = get_colors("categorical")

    # Extract data
    param_key = f"param_{param_name}"
    if param_key not in cv_results:
        raise ValueError(f"Parameter '{param_name}' not found in cv_results")

    param_values = cv_results[param_key]
    scores_mean = cv_results[scoring]
    scores_std = cv_results.get(scoring.replace("mean_", "std_"), np.zeros_like(scores_mean))

    # Sort by parameter value
    sorted_idx = np.argsort(param_values)
    param_values = np.array(param_values)[sorted_idx]
    scores_mean = np.array(scores_mean)[sorted_idx]
    scores_std = np.array(scores_std)[sorted_idx]

    # Plot
    if log_scale:
        ax.semilogx(param_values, scores_mean, color=colors[0], marker="o",
                    linewidth=2, label=scoring)
    else:
        ax.plot(param_values, scores_mean, color=colors[0], marker="o",
                linewidth=2, label=scoring)

    if show_std:
        ax.fill_between(param_values, scores_mean - scores_std, scores_mean + scores_std,
                        color=colors[0], alpha=0.2)

    # Mark best score
    best_idx = np.argmax(scores_mean)
    ax.scatter([param_values[best_idx]], [scores_mean[best_idx]],
               color=colors[1], s=200, marker="*", zorder=5,
               label=f"Best: {scores_mean[best_idx]:.4f}")

    if title is None:
        title = f"Cross-Validation Results ({param_name})"

    ax.set_xlabel(param_name)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    return fig, ax


def plot_cv_comparison(
    results: dict[str, dict[str, float]],
    metric: str = "test_score",
    title: str = "Cross-Validation Comparison",
    figsize: str = "medium",
    orientation: Literal["horizontal", "vertical"] = "horizontal",
    show_std: bool = True,
    ax: Axes | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Compare cross-validation results across multiple models.

    Args:
        results: Dictionary with model names as keys and
            {'mean': float, 'std': float} dicts as values.
        metric: Name of metric being compared.
        title: Plot title.
        figsize: Size preset.
        orientation: 'horizontal' or 'vertical'.
        show_std: Whether to show standard deviation as error bars.
        ax: Existing axes to plot on.
        **kwargs: Additional arguments.

    Returns:
        Tuple of (figure, axes).

    Example:
        >>> results = {
        ...     'Random Forest': {'mean': 0.85, 'std': 0.03},
        ...     'XGBoost': {'mean': 0.87, 'std': 0.02},
        ... }
        >>> fig, ax = plot_cv_comparison(results)
    """
    apply_corporate_style()

    if ax is None:
        fig, ax = create_figure(size=figsize)
    else:
        fig = ax.figure

    colors = get_colors("categorical")

    # Extract data
    models = list(results.keys())
    means = [results[m].get("mean", results[m]) if isinstance(results[m], dict)
             else results[m] for m in models]
    stds = [results[m].get("std", 0) if isinstance(results[m], dict)
            else 0 for m in models]

    # Sort by mean score
    sorted_idx = np.argsort(means)[::-1]
    models = [models[i] for i in sorted_idx]
    means = [means[i] for i in sorted_idx]
    stds = [stds[i] for i in sorted_idx]

    # Assign colors
    bar_colors = [colors[i % len(colors)] for i in range(len(models))]

    # Plot
    if orientation == "horizontal":
        # Reverse for horizontal (best at top)
        models = models[::-1]
        means = means[::-1]
        stds = stds[::-1]
        bar_colors = bar_colors[::-1]

        if show_std:
            ax.barh(models, means, xerr=stds, color=bar_colors, capsize=5, **kwargs)
        else:
            ax.barh(models, means, color=bar_colors, **kwargs)
        ax.set_xlabel(metric.replace("_", " ").title())
    else:
        if show_std:
            ax.bar(models, means, yerr=stds, color=bar_colors, capsize=5, **kwargs)
        else:
            ax.bar(models, means, color=bar_colors, **kwargs)
        ax.set_ylabel(metric.replace("_", " ").title())
        plt.xticks(rotation=45, ha="right")

    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.3,
            axis="x" if orientation == "horizontal" else "y")

    plt.tight_layout()
    return fig, ax
