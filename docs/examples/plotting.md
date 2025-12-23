# Plotting Examples

This page provides examples of using the plotting module.

## Setup

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ds_utils.plotting import (
    apply_corporate_style,
    plot_line,
    plot_scatter,
    plot_bar,
    plot_time_series,
    plot_histogram,
    plot_density,
    plot_boxplot,
    plot_violin,
    plot_qq,
    get_color_palette,
    save_plot,
)

# Apply corporate styling
apply_corporate_style()
```

## Line Plots

### Basic Line Plot

```python
x = np.linspace(0, 10, 100)
y = np.sin(x)

fig, ax = plot_line(x, y, title="Sine Wave", xlabel="X", ylabel="Y")
plt.show()
```

### Multiple Lines

```python
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.sin(x) * np.cos(x)

fig, ax = plt.subplots(figsize=(10, 6))
colors = get_color_palette("categorical")

ax.plot(x, y1, label="sin(x)", color=colors[0])
ax.plot(x, y2, label="cos(x)", color=colors[1])
ax.plot(x, y3, label="sin(x)cos(x)", color=colors[2])

ax.set_title("Trigonometric Functions")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.legend()
plt.show()
```

## Scatter Plots

### Basic Scatter Plot

```python
np.random.seed(42)
x = np.random.randn(100)
y = x + np.random.randn(100) * 0.5

fig, ax = plot_scatter(x, y, title="Scatter Plot", xlabel="X", ylabel="Y")
plt.show()
```

### Colored by Category

```python
np.random.seed(42)
n = 100
x = np.random.randn(n)
y = np.random.randn(n)
categories = np.random.choice(["A", "B", "C"], n)

fig, ax = plt.subplots(figsize=(10, 6))
colors = get_color_palette("categorical")

for i, cat in enumerate(["A", "B", "C"]):
    mask = categories == cat
    ax.scatter(x[mask], y[mask], label=cat, color=colors[i], alpha=0.7)

ax.set_title("Scatter by Category")
ax.legend()
plt.show()
```

## Bar Charts

### Vertical Bar Chart

```python
categories = ["Q1", "Q2", "Q3", "Q4"]
values = [150, 200, 180, 220]

fig, ax = plot_bar(categories, values, title="Quarterly Sales", ylabel="Sales ($K)")
plt.show()
```

### Horizontal Bar Chart

```python
products = ["Product A", "Product B", "Product C", "Product D", "Product E"]
sales = [45, 30, 25, 20, 15]

fig, ax = plot_bar(products, sales, horizontal=True, title="Sales by Product")
plt.show()
```

## Time Series

### Basic Time Series

```python
dates = pd.date_range("2024-01-01", periods=365, freq="D")
values = np.cumsum(np.random.randn(365)) + 100

fig, ax = plot_time_series(dates, values, title="Daily Values", ylabel="Value")
plt.show()
```

## Distribution Plots

### Histogram

```python
data = np.random.normal(100, 15, 1000)

fig, ax = plot_histogram(data, bins=30, title="Value Distribution", show_kde=True)
plt.show()
```

### Density Plot

```python
data = np.random.normal(100, 15, 1000)

fig, ax = plot_density(data, title="Density Plot", fill=True)
plt.show()
```

### Boxplot

```python
data = [np.random.normal(i * 10, 5, 100) for i in range(5)]

fig, ax = plot_boxplot(data, labels=["A", "B", "C", "D", "E"], title="Boxplot")
plt.show()
```

### Violin Plot

```python
data = [np.random.normal(i * 10, 5, 100) for i in range(5)]

fig, ax = plot_violin(data, labels=["A", "B", "C", "D", "E"], title="Violin Plot")
plt.show()
```

### Q-Q Plot

```python
data = np.random.normal(0, 1, 1000)

fig, ax = plot_qq(data, title="Q-Q Plot")
plt.show()
```

## Saving Plots

```python
fig, ax = plot_line(x, y, title="My Plot")

# Save in multiple formats
save_plot(fig, "my_plot", formats=["png", "pdf", "svg"], dpi=300)
```

## Color Palettes

```python
# Get different palettes
categorical = get_color_palette("categorical")
sequential = get_color_palette("sequential", n_colors=9)
diverging = get_color_palette("diverging", n_colors=9)

# Display palettes
fig, axes = plt.subplots(3, 1, figsize=(10, 4))

for ax, (name, colors) in zip(axes, [
    ("Categorical", categorical),
    ("Sequential", sequential),
    ("Diverging", diverging),
]):
    for i, color in enumerate(colors):
        ax.bar(i, 1, color=color)
    ax.set_title(name)
    ax.set_xlim(-0.5, len(colors) - 0.5)

plt.tight_layout()
plt.show()
```
