# Installation

## Requirements

- Python 3.10 or higher
- pip or uv package manager

## Quick Installation

### Using pip

```bash
pip install ds-utils
```

### Using uv (recommended)

```bash
uv add ds-utils
```

## Development Installation

For development or contributing, clone the repository and install with development dependencies:

```bash
# Clone the repository
git clone https://github.com/your-org/ds-utils.git
cd ds-utils

# Create virtual environment and install dependencies
uv venv
uv sync

# Or with pip
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Dependencies

### Core Dependencies

| Package | Minimum Version | Description |
|---------|----------------|-------------|
| matplotlib | ≥3.7 | Plotting library |
| seaborn | ≥0.12 | Statistical visualization |
| pandas | ≥2.0 | Data manipulation |
| numpy | ≥1.24 | Numerical computing |
| scikit-learn | ≥1.3 | Machine learning |
| pyjanitor | ≥0.26 | Data cleaning extensions |
| pandera | ≥0.17 | Data validation |
| pyyaml | ≥6.0 | YAML configuration |

### Development Dependencies

| Package | Description |
|---------|-------------|
| jupyter | Jupyter notebooks |
| pytest | Testing framework |
| pytest-cov | Coverage reporting |
| black | Code formatting |
| ruff | Linting |
| mypy | Type checking |
| pre-commit | Git hooks |
| mkdocs-material | Documentation |
| mkdocstrings[python] | API documentation |

## Verifying Installation

After installation, verify that DS Utils is working correctly:

```python
import ds_utils

print(ds_utils.__version__)
```

You should see the version number (e.g., `0.1.0`).

## Optional Dependencies

### SHAP Integration

For SHAP-based feature importance visualizations:

```bash
pip install shap
```

### Extended Plotting

For additional plot types:

```bash
pip install plotly
```

## Troubleshooting

### Locale Issues

If you encounter locale-related errors on Linux/macOS:

```bash
# Generate German locale
sudo locale-gen de_DE.UTF-8
sudo update-locale
```

On Windows, the German locale should be available by default.

### Font Issues

If corporate fonts are not displaying:

1. Install the required fonts on your system
2. Clear matplotlib's font cache:

```python
import matplotlib.font_manager
matplotlib.font_manager._rebuild()
```

### Import Errors

If you encounter import errors, ensure all dependencies are installed:

```bash
pip install -e ".[dev]" --force-reinstall
```
