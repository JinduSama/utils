"""Standard plotting functions for ds_utils.

This module provides enhanced line, scatter, and bar plots with
corporate styling and localized formatting.
"""

from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ds_utils.config import get_colors
from ds_utils.plotting.core import apply_corporate_style, create_figure, format_number

from ds_utils.config.logging_config import get_logger

logger = get_logger("plotting.standard")


def plot_line(
    data: pd.DataFrame,
    x: str,
    y: str | list[str],
    hue: str | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    figsize: str = "medium",
    colors: list[str] | None = None,
    markers: bool = False,
    legend: bool = True,
    legend_loc: str = "best",
    grid: bool = True,
    ax: Axes | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Create an enhanced line plot with corporate styling.

    Args:
        data: DataFrame containing the data.
        x: Column name for x-axis.
        y: Column name(s) for y-axis. Can be a single column or list of columns.
        hue: Optional column for color grouping.
        title: Plot title.
        xlabel: X-axis label. If None, uses column name.
        ylabel: Y-axis label. If None, uses column name.
        figsize: Size preset ('small', 'medium', 'large', 'presentation', 'square').
        colors: Custom color palette. If None, uses corporate colors.
        markers: Whether to show markers on data points.
        legend: Whether to show legend.
        legend_loc: Legend location.
        grid: Whether to show grid.
        ax: Existing axes to plot on. If None, creates new figure.
        **kwargs: Additional arguments passed to matplotlib plot.

    Returns:
        Tuple of (figure, axes).

    Example:
        >>> df = pd.DataFrame({'date': pd.date_range('2024-01-01', periods=30),
        ...                    'sales': np.random.randn(30).cumsum()})
        >>> fig, ax = plot_line(df, x='date', y='sales', title='Sales Trend')
    """
    apply_corporate_style()

    if ax is None:
        fig, ax = create_figure(size=figsize)
    else:
        fig = ax.figure

    if colors is None:
        colors = get_colors("categorical")

    # Handle multiple y columns
    if isinstance(y, str):
        y_cols = [y]
    else:
        y_cols = list(y)

    if hue is not None:
        # Use seaborn for hue-based plotting
        sns.lineplot(
            data=data,
            x=x,
            y=y_cols[0],
            hue=hue,
            palette=colors,
            marker="o" if markers else None,
            ax=ax,
            **kwargs,
        )
    else:
        # Plot each y column
        for i, col in enumerate(y_cols):
            color = colors[i % len(colors)]
            ax.plot(
                data[x],
                data[col],
                color=color,
                marker="o" if markers else None,
                label=col,
                **kwargs,
            )

    # Apply labels and styling
    if title:
        ax.set_title(title)
    ax.set_xlabel(xlabel if xlabel else x)
    ax.set_ylabel(ylabel if ylabel else (y_cols[0] if len(y_cols) == 1 else "Value"))

    if grid:
        ax.grid(True, linestyle="--", alpha=0.3)

    if legend and (len(y_cols) > 1 or hue is not None):
        ax.legend(loc=legend_loc)

    # Format dates if x-axis is datetime
    if pd.api.types.is_datetime64_any_dtype(data[x]):
        fig.autofmt_xdate()

    plt.tight_layout()
    return fig, ax


def plot_scatter(
    data: pd.DataFrame,
    x: str,
    y: str,
    hue: str | None = None,
    size: str | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    figsize: str = "medium",
    colors: list[str] | None = None,
    alpha: float = 0.7,
    add_regression: bool = False,
    add_marginals: bool = False,
    ax: Axes | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Create an enhanced scatter plot with optional regression line.

    Args:
        data: DataFrame containing the data.
        x: Column name for x-axis.
        y: Column name for y-axis.
        hue: Optional column for color grouping.
        size: Optional column for point size mapping.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        figsize: Size preset.
        colors: Custom color palette.
        alpha: Point transparency.
        add_regression: Whether to add a regression line.
        add_marginals: Whether to add marginal distributions (creates JointGrid).
        ax: Existing axes to plot on.
        **kwargs: Additional arguments passed to scatter plot.

    Returns:
        Tuple of (figure, axes).
    """
    apply_corporate_style()

    if colors is None:
        colors = get_colors("categorical")

    if add_marginals:
        # Use seaborn JointGrid for marginal distributions
        from ds_utils.config import get_figure_size

        figsize_tuple = get_figure_size(figsize)
        g = sns.jointplot(
            data=data,
            x=x,
            y=y,
            hue=hue,
            palette=colors if hue else None,
            alpha=alpha,
            height=figsize_tuple[0],
            **kwargs,
        )
        if title:
            g.figure.suptitle(title, y=1.02)
        return g.figure, g.ax_joint

    if ax is None:
        fig, ax = create_figure(size=figsize)
    else:
        fig = ax.figure

    # Create scatter plot
    scatter_kwargs: dict[str, Any] = {"alpha": alpha}
    scatter_kwargs.update(kwargs)

    if hue is not None or size is not None:
        sns.scatterplot(
            data=data,
            x=x,
            y=y,
            hue=hue,
            size=size,
            palette=colors if hue else None,
            ax=ax,
            **scatter_kwargs,
        )
    else:
        ax.scatter(
            data[x],
            data[y],
            color=colors[0],
            **scatter_kwargs,
        )

    # Add regression line
    if add_regression:
        z = np.polyfit(data[x].dropna(), data[y].dropna(), 1)
        p = np.poly1d(z)
        x_line = np.linspace(data[x].min(), data[x].max(), 100)
        ax.plot(x_line, p(x_line), color=colors[1] if len(colors) > 1 else "red",
                linestyle="--", linewidth=2, label="Regression")
        ax.legend()

    # Apply labels
    if title:
        ax.set_title(title)
    ax.set_xlabel(xlabel if xlabel else x)
    ax.set_ylabel(ylabel if ylabel else y)

    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    return fig, ax


def plot_bar(
    data: pd.DataFrame,
    x: str,
    y: str,
    hue: str | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    figsize: str = "medium",
    colors: list[str] | None = None,
    orientation: Literal["vertical", "horizontal"] = "vertical",
    show_values: bool = True,
    value_format: str = ".1f",
    stacked: bool = False,
    error_bars: str | None = None,
    ax: Axes | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Create an enhanced bar plot with value labels.

    Args:
        data: DataFrame containing the data.
        x: Column name for categories (x-axis for vertical, y-axis for horizontal).
        y: Column name for values.
        hue: Optional column for grouping.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        figsize: Size preset.
        colors: Custom color palette.
        orientation: 'vertical' or 'horizontal'.
        show_values: Whether to show value labels on bars.
        value_format: Format string for value labels.
        stacked: Whether to create stacked bars (requires hue).
        error_bars: Column name for error bar values.
        ax: Existing axes to plot on.
        **kwargs: Additional arguments passed to bar plot.

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

    # Prepare plot arguments
    plot_kwargs: dict[str, Any] = {"palette": colors}
    plot_kwargs.update(kwargs)

    if orientation == "horizontal":
        if hue:
            sns.barplot(data=data, y=x, x=y, hue=hue, ax=ax, orient="h", **plot_kwargs)
        else:
            sns.barplot(data=data, y=x, x=y, ax=ax, orient="h", color=colors[0], **plot_kwargs)
    else:
        if hue:
            sns.barplot(data=data, x=x, y=y, hue=hue, ax=ax, **plot_kwargs)
        else:
            sns.barplot(data=data, x=x, y=y, ax=ax, color=colors[0], **plot_kwargs)

    # Add value labels
    if show_values:
        for container in ax.containers:
            ax.bar_label(container, fmt=f"%{value_format}", padding=3)

    # Apply labels
    if title:
        ax.set_title(title)

    if orientation == "horizontal":
        ax.set_xlabel(ylabel if ylabel else y)
        ax.set_ylabel(xlabel if xlabel else x)
    else:
        ax.set_xlabel(xlabel if xlabel else x)
        ax.set_ylabel(ylabel if ylabel else y)

    ax.grid(True, linestyle="--", alpha=0.3, axis="y" if orientation == "vertical" else "x")
    plt.tight_layout()

    return fig, ax


def plot_time_series(
    data: pd.DataFrame,
    y: str | list[str],
    date_col: str | None = None,
    title: str | None = None,
    xlabel: str = "Date",
    ylabel: str | None = None,
    figsize: str = "large",
    colors: list[str] | None = None,
    show_trend: bool = False,
    resample: str | None = None,
    ax: Axes | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Create a time series plot with optional trend line.

    Args:
        data: DataFrame with datetime index or date column.
        y: Column name(s) for y-axis.
        date_col: Column name for dates. If None, uses index.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        figsize: Size preset.
        colors: Custom color palette.
        show_trend: Whether to show a trend line.
        resample: Resample frequency (e.g., 'W', 'M', 'Q').
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

    # Prepare data
    plot_data = data.copy()
    if date_col is not None:
        plot_data = plot_data.set_index(date_col)

    # Ensure datetime index
    if not isinstance(plot_data.index, pd.DatetimeIndex):
        plot_data.index = pd.to_datetime(plot_data.index)

    # Resample if requested
    if resample:
        plot_data = plot_data.resample(resample).mean()

    # Handle multiple y columns
    if isinstance(y, str):
        y_cols = [y]
    else:
        y_cols = list(y)

    # Plot each series
    for i, col in enumerate(y_cols):
        color = colors[i % len(colors)]
        ax.plot(plot_data.index, plot_data[col], color=color, label=col, **kwargs)

        # Add trend line
        if show_trend:
            x_numeric = np.arange(len(plot_data))
            z = np.polyfit(x_numeric, plot_data[col].fillna(method="ffill"), 1)
            p = np.poly1d(z)
            ax.plot(
                plot_data.index,
                p(x_numeric),
                color=color,
                linestyle="--",
                alpha=0.7,
                label=f"{col} (trend)",
            )

    # Apply labels
    if title:
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel if ylabel else (y_cols[0] if len(y_cols) == 1 else "Value"))

    if len(y_cols) > 1 or show_trend:
        ax.legend()

    ax.grid(True, linestyle="--", alpha=0.3)
    fig.autofmt_xdate()
    plt.tight_layout()

    return fig, ax
