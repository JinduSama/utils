# Plotting Module

The plotting module provides utilities for creating publication-quality visualizations with corporate styling and German locale support.

## Overview

The plotting module is organized into several submodules:

- **core** - Core utilities for styling, locale setup, and figure management
- **standard** - Standard plot types (line, scatter, bar, time series)
- **distributions** - Distribution visualizations (histogram, density, boxplot, violin, Q-Q)
- **styles** - Color palettes and styling utilities

## Getting Started

```python
from ds_utils.plotting import (
    apply_corporate_style,
    plot_line,
    plot_scatter,
    plot_histogram,
    save_plot,
)

# Apply corporate styling to all plots
apply_corporate_style()
```

## Core Functions

### `apply_corporate_style()`

Applies corporate styling to matplotlib. This should be called once at the beginning of your script or notebook.

```python
from ds_utils.plotting import apply_corporate_style

apply_corporate_style()
```

### `setup_locale(locale_name="de_DE.UTF-8")`

Sets up locale for number formatting. German locale uses comma as decimal separator and period as thousands separator.

```python
from ds_utils.plotting import setup_locale

setup_locale("de_DE.UTF-8")
```

### `format_number(value, decimals=2)`

Formats a number according to the configured locale.

```python
from ds_utils.plotting import format_number

formatted = format_number(1234567.89)  # "1.234.567,89"
```

### `save_plot(fig, filename, formats=None, dpi=300)`

Saves a figure to disk in one or more formats.

```python
from ds_utils.plotting import save_plot

fig, ax = plot_line(x, y)
save_plot(fig, "my_plot", formats=["png", "pdf", "svg"], dpi=300)
```

### `add_corporate_logo(fig, logo_path=None, position="lower right")`

Adds a corporate logo to a figure.

```python
from ds_utils.plotting import add_corporate_logo

add_corporate_logo(fig, logo_path="path/to/logo.png", position="lower right")
```

## Standard Plots

### `plot_line(x, y, **kwargs)`

Creates a line plot with corporate styling.

```python
import numpy as np
from ds_utils.plotting import plot_line

x = np.linspace(0, 10, 100)
y = np.sin(x)

fig, ax = plot_line(
    x, y,
    title="Sine Wave",
    xlabel="X-axis",
    ylabel="Y-axis",
    color=None,  # Uses default color palette
    linewidth=2,
    linestyle="-",
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `x` | array-like | required | X-axis data |
| `y` | array-like | required | Y-axis data |
| `title` | str | None | Plot title |
| `xlabel` | str | None | X-axis label |
| `ylabel` | str | None | Y-axis label |
| `color` | str | None | Line color |
| `linewidth` | float | 2 | Line width |
| `linestyle` | str | "-" | Line style |
| `figsize` | tuple | None | Figure size |
| `ax` | Axes | None | Existing axes to plot on |

### `plot_scatter(x, y, **kwargs)`

Creates a scatter plot with corporate styling.

```python
from ds_utils.plotting import plot_scatter

fig, ax = plot_scatter(
    x, y,
    title="Scatter Plot",
    xlabel="X-axis",
    ylabel="Y-axis",
    color=None,
    size=50,
    alpha=0.7,
)
```

### `plot_bar(categories, values, **kwargs)`

Creates a bar chart with corporate styling.

```python
from ds_utils.plotting import plot_bar

categories = ["A", "B", "C", "D"]
values = [10, 25, 15, 30]

fig, ax = plot_bar(
    categories, values,
    title="Bar Chart",
    xlabel="Category",
    ylabel="Value",
    horizontal=False,
    color=None,
)
```

### `plot_time_series(dates, values, **kwargs)`

Creates a time series plot with corporate styling and proper date formatting.

```python
import pandas as pd
from ds_utils.plotting import plot_time_series

dates = pd.date_range("2024-01-01", periods=100, freq="D")
values = np.cumsum(np.random.randn(100))

fig, ax = plot_time_series(
    dates, values,
    title="Time Series",
    xlabel="Date",
    ylabel="Value",
    date_format="%Y-%m-%d",
)
```

## Distribution Plots

### `plot_histogram(data, **kwargs)`

Creates a histogram with corporate styling.

```python
from ds_utils.plotting import plot_histogram

data = np.random.normal(0, 1, 1000)

fig, ax = plot_histogram(
    data,
    title="Distribution",
    xlabel="Value",
    ylabel="Frequency",
    bins=30,
    density=False,
    show_kde=True,
)
```

### `plot_density(data, **kwargs)`

Creates a kernel density estimate plot.

```python
from ds_utils.plotting import plot_density

fig, ax = plot_density(
    data,
    title="Density Plot",
    xlabel="Value",
    ylabel="Density",
    fill=True,
)
```

### `plot_boxplot(data, **kwargs)`

Creates a boxplot with corporate styling.

```python
from ds_utils.plotting import plot_boxplot

data = [np.random.normal(0, 1, 100) for _ in range(5)]

fig, ax = plot_boxplot(
    data,
    labels=["A", "B", "C", "D", "E"],
    title="Boxplot",
    xlabel="Group",
    ylabel="Value",
)
```

### `plot_violin(data, **kwargs)`

Creates a violin plot with corporate styling.

```python
from ds_utils.plotting import plot_violin

fig, ax = plot_violin(
    data,
    labels=["A", "B", "C", "D", "E"],
    title="Violin Plot",
    xlabel="Group",
    ylabel="Value",
)
```

### `plot_qq(data, **kwargs)`

Creates a Q-Q plot for normality assessment.

```python
from ds_utils.plotting import plot_qq

fig, ax = plot_qq(
    data,
    title="Q-Q Plot",
    distribution="norm",
)
```

## Color Palettes

### `get_color_palette(palette_name="categorical", n_colors=None)`

Returns a color palette from the corporate style configuration.

```python
from ds_utils.plotting import get_color_palette

# Get categorical palette
colors = get_color_palette("categorical")

# Get sequential palette with 10 colors
colors = get_color_palette("sequential", n_colors=10)

# Get diverging palette
colors = get_color_palette("diverging")
```

### `create_colormap(palette_name="sequential", n_colors=256)`

Creates a matplotlib colormap from a palette.

```python
from ds_utils.plotting import create_colormap

cmap = create_colormap("sequential")
```

## Context Managers

### `plot_context(**kwargs)`

A context manager for temporarily changing plot settings.

```python
from ds_utils.plotting import plot_context

with plot_context(figsize=(12, 8), dpi=150):
    fig, ax = plot_line(x, y)
```

### `style_context(style="dark")`

A context manager for temporarily changing the plot style.

```python
from ds_utils.plotting import style_context

with style_context("dark"):
    fig, ax = plot_line(x, y)
```

## Configuration

The plotting module reads its configuration from `plot_styles.yaml`. You can customize:

- Color palettes (categorical, sequential, diverging)
- Fonts and sizes
- Figure dimensions
- Grid and spine styles
- Logo settings
- Save settings
- Locale settings

See the [Configuration Guide](configuration.md) for details.
