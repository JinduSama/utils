"""Distribution plotting functions for ds_utils.

This module provides enhanced distribution plots including histograms,
density plots, box plots, and violin plots.
"""

from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy import stats

from ds_utils.config import get_colors
from ds_utils.plotting.core import apply_corporate_style, create_figure

from ds_utils.config.logging_config import get_logger

logger = get_logger("plotting.distributions")


def plot_histogram(
    data: pd.DataFrame | pd.Series | np.ndarray,
    column: str | None = None,
    bins: int | str | list[float] = "auto",
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str = "Frequency",
    figsize: str = "medium",
    color: str | None = None,
    add_kde: bool = True,
    add_normal: bool = False,
    show_stats: bool = True,
    ax: Axes | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Create an enhanced histogram with optional KDE and normal overlay.

    Args:
        data: Data to plot. Can be DataFrame, Series, or array.
        column: Column name if data is DataFrame.
        bins: Number of bins, or 'auto', 'fd', 'sturges', etc., or explicit bin edges.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        figsize: Size preset.
        color: Bar color. If None, uses primary corporate color.
        add_kde: Whether to overlay kernel density estimate.
        add_normal: Whether to overlay normal distribution fit.
        show_stats: Whether to show mean and std in legend.
        ax: Existing axes to plot on.
        **kwargs: Additional arguments passed to histogram.

    Returns:
        Tuple of (figure, axes).

    Example:
        >>> data = pd.DataFrame({'values': np.random.randn(1000)})
        >>> fig, ax = plot_histogram(data, column='values', add_normal=True)
    """
    apply_corporate_style()

    if ax is None:
        fig, ax = create_figure(size=figsize)
    else:
        fig = ax.figure

    # Extract data array
    if isinstance(data, pd.DataFrame):
        if column is None:
            raise ValueError("column must be specified when data is a DataFrame")
        values = data[column].dropna().values
        if xlabel is None:
            xlabel = column
    elif isinstance(data, pd.Series):
        values = data.dropna().values
        if xlabel is None:
            xlabel = data.name or "Value"
    else:
        values = np.array(data).flatten()
        values = values[~np.isnan(values)]
        if xlabel is None:
            xlabel = "Value"

    colors = get_colors("categorical")
    if color is None:
        color = colors[0]

    # Plot histogram
    stat = "density" if (add_kde or add_normal) else "count"
    hist_kwargs: dict[str, Any] = {"color": color, "alpha": 0.7, "edgecolor": "white"}
    hist_kwargs.update(kwargs)

    sns.histplot(
        values,
        bins=bins,
        stat=stat,
        ax=ax,
        **hist_kwargs,
    )

    # Add KDE
    if add_kde:
        sns.kdeplot(values, ax=ax, color=colors[1] if len(colors) > 1 else "red",
                    linewidth=2, label="KDE")

    # Add normal distribution fit
    if add_normal:
        mu, std = values.mean(), values.std()
        x = np.linspace(values.min(), values.max(), 100)
        y = stats.norm.pdf(x, mu, std)
        ax.plot(x, y, color=colors[2] if len(colors) > 2 else "green",
                linewidth=2, linestyle="--", label=f"Normal (μ={mu:.2f}, σ={std:.2f})")

    # Add statistics annotation
    if show_stats:
        stats_text = f"n = {len(values)}\nmean = {values.mean():.2f}\nstd = {values.std():.2f}"
        ax.text(
            0.95, 0.95, stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    # Apply labels
    if title:
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel if stat == "count" else "Density")

    if add_kde or add_normal:
        ax.legend()

    ax.grid(True, linestyle="--", alpha=0.3, axis="y")
    plt.tight_layout()

    return fig, ax


def plot_density(
    data: pd.DataFrame,
    columns: str | list[str],
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str = "Density",
    figsize: str = "medium",
    colors: list[str] | None = None,
    fill: bool = True,
    alpha: float = 0.3,
    add_rug: bool = False,
    ax: Axes | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Create a kernel density plot for one or more distributions.

    Args:
        data: DataFrame containing the data.
        columns: Column name(s) to plot.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        figsize: Size preset.
        colors: Custom color palette.
        fill: Whether to fill under the curves.
        alpha: Fill transparency.
        add_rug: Whether to add rug plot.
        ax: Existing axes to plot on.
        **kwargs: Additional arguments passed to kdeplot.

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

    # Handle single column
    if isinstance(columns, str):
        columns = [columns]

    # Plot each distribution
    for i, col in enumerate(columns):
        color = colors[i % len(colors)]
        sns.kdeplot(
            data=data,
            x=col,
            ax=ax,
            color=color,
            fill=fill,
            alpha=alpha if fill else 1.0,
            label=col,
            **kwargs,
        )

        if add_rug:
            sns.rugplot(data=data, x=col, ax=ax, color=color, alpha=0.5)

    # Apply labels
    if title:
        ax.set_title(title)
    ax.set_xlabel(xlabel if xlabel else (columns[0] if len(columns) == 1 else "Value"))
    ax.set_ylabel(ylabel)

    if len(columns) > 1:
        ax.legend()

    ax.grid(True, linestyle="--", alpha=0.3, axis="y")
    plt.tight_layout()

    return fig, ax


def plot_boxplot(
    data: pd.DataFrame,
    x: str | None = None,
    y: str | None = None,
    hue: str | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    figsize: str = "medium",
    colors: list[str] | None = None,
    orientation: Literal["vertical", "horizontal"] = "vertical",
    show_outliers: bool = True,
    show_means: bool = False,
    ax: Axes | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Create an enhanced box plot.

    Args:
        data: DataFrame containing the data.
        x: Column for x-axis (categories for vertical, values for horizontal).
        y: Column for y-axis (values for vertical, categories for horizontal).
        hue: Optional column for color grouping.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        figsize: Size preset.
        colors: Custom color palette.
        orientation: 'vertical' or 'horizontal'.
        show_outliers: Whether to show outlier points.
        show_means: Whether to show mean markers.
        ax: Existing axes to plot on.
        **kwargs: Additional arguments passed to boxplot.

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

    # Build plot kwargs
    plot_kwargs: dict[str, Any] = {
        "showfliers": show_outliers,
        "showmeans": show_means,
    }
    if hue:
        plot_kwargs["palette"] = colors
    plot_kwargs.update(kwargs)

    if orientation == "horizontal":
        sns.boxplot(data=data, x=y, y=x, hue=hue, ax=ax, orient="h", **plot_kwargs)
    else:
        sns.boxplot(data=data, x=x, y=y, hue=hue, ax=ax, **plot_kwargs)

    # Apply labels
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    ax.grid(True, linestyle="--", alpha=0.3,
            axis="x" if orientation == "horizontal" else "y")
    plt.tight_layout()

    return fig, ax


def plot_violin(
    data: pd.DataFrame,
    x: str | None = None,
    y: str | None = None,
    hue: str | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    figsize: str = "medium",
    colors: list[str] | None = None,
    split: bool = False,
    inner: Literal["box", "quart", "point", "stick", None] = "box",
    ax: Axes | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Create an enhanced violin plot.

    Args:
        data: DataFrame containing the data.
        x: Column for x-axis (categories).
        y: Column for y-axis (values).
        hue: Optional column for color grouping.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        figsize: Size preset.
        colors: Custom color palette.
        split: Whether to split violins when using hue (requires exactly 2 hue levels).
        inner: What to show inside violins ('box', 'quart', 'point', 'stick', or None).
        ax: Existing axes to plot on.
        **kwargs: Additional arguments passed to violinplot.

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

    plot_kwargs: dict[str, Any] = {"inner": inner}
    if hue:
        plot_kwargs["palette"] = colors
        plot_kwargs["split"] = split
    plot_kwargs.update(kwargs)

    sns.violinplot(data=data, x=x, y=y, hue=hue, ax=ax, **plot_kwargs)

    # Apply labels
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    ax.grid(True, linestyle="--", alpha=0.3, axis="y")
    plt.tight_layout()

    return fig, ax


def plot_qq(
    data: pd.DataFrame | pd.Series | np.ndarray,
    column: str | None = None,
    title: str = "Q-Q Plot",
    figsize: str = "square",
    color: str | None = None,
    line: Literal["45", "s", "r", "q"] = "45",
    ax: Axes | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Create a Q-Q (quantile-quantile) plot for normality assessment.

    Args:
        data: Data to plot.
        column: Column name if data is DataFrame.
        title: Plot title.
        figsize: Size preset.
        color: Point color.
        line: Reference line type. '45' for 45-degree line, 's' for standardized,
            'r' for regression, 'q' for quartiles.
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

    # Extract data array
    if isinstance(data, pd.DataFrame):
        if column is None:
            raise ValueError("column must be specified when data is a DataFrame")
        values = data[column].dropna().values
    elif isinstance(data, pd.Series):
        values = data.dropna().values
    else:
        values = np.array(data).flatten()
        values = values[~np.isnan(values)]

    colors = get_colors("categorical")
    if color is None:
        color = colors[0]

    # Create Q-Q plot
    stats.probplot(values, dist="norm", plot=ax)

    # Customize appearance
    ax.get_lines()[0].set_markerfacecolor(color)
    ax.get_lines()[0].set_markeredgecolor(color)
    ax.get_lines()[0].set_alpha(0.7)

    if title:
        ax.set_title(title)
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Sample Quantiles")

    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    return fig, ax
