# Quick Start

Get up and running with DS Utils in minutes.

## Basic Setup

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import ds_utils modules
from ds_utils.plotting import apply_corporate_style, plot_line, plot_bar, save_plot
from ds_utils.ml_eval import classification_summary, plot_confusion_matrix
from ds_utils.cleaning import clean_german_numbers, validate_dataframe

# Apply corporate styling (do this once at the start)
apply_corporate_style()
```

## Creating Your First Plot

```python
# Generate sample data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create a line plot
fig, ax = plot_line(
    x, y,
    title="Sine Wave",
    xlabel="X-axis",
    ylabel="Y-axis",
)
plt.show()
```

## Working with German Data

```python
# Sample data with German formatting
df = pd.DataFrame({
    "product": ["Widget A", "Widget B", "Widget C"],
    "price": ["1.234,56", "2.345,67", "3.456,78"],
    "date": ["15.03.2024", "16.03.2024", "17.03.2024"],
})

# Clean German numbers
from ds_utils.cleaning import clean_german_numbers, parse_german_dates

df["price_clean"] = clean_german_numbers(df["price"])
df["date_clean"] = parse_german_dates(df["date"])

print(df)
```

## Evaluating a Model

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Generate data
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Get predictions
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

# Get classification summary
from ds_utils.ml_eval import classification_summary, plot_confusion_matrix, plot_roc_curve

metrics = classification_summary(y_test, y_pred, y_proba)
print(metrics)

# Visualize
plot_confusion_matrix(y_test, y_pred)
plt.show()

plot_roc_curve(y_test, y_proba)
plt.show()
```

## Data Validation

```python
import pandera as pa
from ds_utils.cleaning import validate_dataframe

# Create validation schema
schema = pa.DataFrameSchema({
    "product": pa.Column(str, pa.Check.str_length(min_value=1)),
    "price_clean": pa.Column(float, pa.Check.gt(0)),
})

# Validate
is_valid, errors = validate_dataframe(df, schema)
if is_valid:
    print("Data is valid!")
else:
    print(f"Errors: {errors}")
```

## Saving Plots

```python
fig, ax = plot_bar(
    ["A", "B", "C", "D"],
    [10, 25, 15, 30],
    title="Sales by Category",
)

# Save in multiple formats
save_plot(fig, "sales_chart", formats=["png", "pdf", "svg"], dpi=300)
```

## Next Steps

- Read the [Plotting Guide](plotting.md) for comprehensive plotting documentation
- Explore the [ML Evaluation Guide](ml_evaluation.md) for model evaluation tools
- Check out the [Data Cleaning Guide](cleaning.md) for data cleaning utilities
- See the [Configuration Guide](configuration.md) to customize DS Utils
