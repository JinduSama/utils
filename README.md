# DS Utils

A Python utilities library for data science and machine learning, focusing on plotting, ML performance evaluation, and data cleaning.

## Features

- **📊 Plotting Utilities**: Beautiful, publication-ready plots with corporate branding support
  - Line, scatter, bar plots with localized formatting
  - Distribution plots (histograms, density, box plots, violin plots)
  - Configurable color schemes and styles

- **🤖 ML Evaluation**: Comprehensive model evaluation tools
  - Classification metrics and visualizations (confusion matrix, ROC, PR curves)
  - Regression metrics and residual analysis
  - Feature importance visualization (supports SHAP, permutation importance)
  - Learning and validation curves

- **🧹 Data Cleaning**: Enhanced data cleaning with validation
  - Pyjanitor integration with custom extensions
  - Pandera schema validation helpers
  - Locale-specific data cleaning (German number formats, dates)

## Installation

### Using uv (Recommended)

```bash
# Clone the repository
git clone https://github.com/your-org/ds-utils.git
cd ds-utils

# Install with uv
uv pip install -e .

# Install with development dependencies
uv pip install -e ".[dev]"
```

### Using pip

```bash
pip install -e .
```

## Quick Start

### Plotting

```python
from ds_utils.plotting import plot_line, plot_scatter, apply_corporate_style
import pandas as pd

# Apply corporate styling globally
apply_corporate_style()

# Create a simple line plot
data = pd.DataFrame({
    'date': pd.date_range('2024-01-01', periods=30),
    'value': range(30)
})
fig, ax = plot_line(data, x='date', y='value', title='Sales Trend')
```

### ML Evaluation

```python
from ds_utils.ml_eval import classification_summary, plot_confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# After training your model...
y_pred = model.predict(X_test)

# Get classification metrics
metrics_df = classification_summary(y_test, y_pred)
print(metrics_df)

# Plot confusion matrix
fig, ax = plot_confusion_matrix(y_test, y_pred)
```

### Data Cleaning

```python
from ds_utils.cleaning import clean_german_numbers, validate_dataframe
import pandas as pd

# Clean German-formatted numbers (comma as decimal separator)
df = pd.DataFrame({'price': ['1.234,56', '789,12']})
df = clean_german_numbers(df, 'price')

# Validate with a schema
from ds_utils.cleaning import create_numeric_schema
schema = create_numeric_schema('price', min_value=0)
validated_df = validate_dataframe(df, schema)
```

## Configuration

The library supports customization through configuration files:

- **Color Schemes**: Define your corporate colors in `config/plot_styles.yaml`
- **Localization**: Default German locale with configurable alternatives
- **Logo Integration**: Add your corporate logo to plots

See the [Configuration Guide](docs/configuration.md) for details.

## Documentation

Full documentation is available at [https://JinduSama.github.io/utils](https://JinduSama.github.io/utils)

## Development

### Setup Development Environment

```bash
# Clone and install
git clone https://github.com/your-org/ds-utils.git
cd ds-utils
uv pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Run linting
ruff check src/
black --check src/
mypy src/
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=ds_utils --cov-report=html

# Run specific test file
pytest tests/test_plotting/test_standard.py
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please ensure:
- All tests pass
- Code follows the style guide (enforced by pre-commit hooks)
- New features include tests and documentation

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Matplotlib](https://matplotlib.org/) - Core plotting library
- [Seaborn](https://seaborn.pydata.org/) - Statistical visualization
- [scikit-learn](https://scikit-learn.org/) - Machine learning library
- [pyjanitor](https://pyjanitor-devs.github.io/pyjanitor/) - Data cleaning
- [Pandera](https://pandera.readthedocs.io/) - Data validation
