# DS Utils

**A comprehensive Data Science utilities library for Python**

DS Utils provides a collection of high-quality utilities for data science and machine learning workflows, with special support for German locale and corporate styling.

## Features

### 📊 Plotting Utilities
- Corporate-styled visualizations with consistent branding
- German locale support for number formatting
- Standard plots: line, scatter, bar, time series
- Distribution plots: histogram, density, boxplot, violin, Q-Q
- Easy logo integration and figure saving

### 📈 ML Evaluation Utilities
- Classification metrics and visualizations (confusion matrix, ROC, PR curves)
- Regression diagnostics (residual plots, prediction error)
- Feature importance visualization (built-in, permutation, SHAP)
- Learning and validation curves

### 🧹 Data Cleaning Utilities
- German number and date parsing
- Column name standardization
- Outlier detection and removal (IQR, z-score)
- Missing value handling with multiple strategies
- Data validation with Pandera schemas

## Quick Start

```python
import pandas as pd
import numpy as np
from ds_utils.plotting import plot_line, apply_corporate_style
from ds_utils.ml_eval import classification_summary, plot_confusion_matrix
from ds_utils.cleaning import clean_german_numbers, validate_dataframe

# Apply corporate styling
apply_corporate_style()

# Create a simple plot
x = np.linspace(0, 10, 100)
y = np.sin(x)
fig, ax = plot_line(x, y, title="Sine Wave", xlabel="X", ylabel="Y")

# Clean German-formatted numbers
df = pd.DataFrame({"price": ["1.234,56", "2.345,67"]})
df["price_clean"] = clean_german_numbers(df["price"])
```

## Installation

```bash
# Using pip
pip install ds-utils

# Using uv
uv add ds-utils

# Development installation
git clone https://github.com/your-org/ds-utils.git
cd ds-utils
uv sync
```

## Documentation

For detailed documentation, see:

- [Installation Guide](installation.md) - Detailed installation instructions
- [Quick Start](quickstart.md) - Get up and running quickly
- [Plotting Guide](plotting.md) - Comprehensive plotting documentation
- [ML Evaluation Guide](ml_evaluation.md) - Machine learning evaluation tools
- [Data Cleaning Guide](cleaning.md) - Data cleaning and validation
- [Configuration Guide](configuration.md) - Customizing DS Utils

## Requirements

- Python >= 3.10
- matplotlib >= 3.7
- seaborn >= 0.12
- pandas >= 2.0
- numpy >= 1.24
- scikit-learn >= 1.3
- pyjanitor >= 0.26
- pandera >= 0.17

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/your-org/ds-utils/blob/main/LICENSE) file for details.
