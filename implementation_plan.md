# Comprehensive Implementation Plan for Data Science Utils Library

## Project Overview
A Python utilities library for data science and machine learning, focusing on plotting, ML performance evaluation, and data cleaning, managed with `uv`.

## Phase 0: Project Setup & Configuration

### 0.1 Core Configuration Files

**pyproject.toml**
- Project metadata (name, version, description)
- Dependencies:
  - Core: `matplotlib`, `seaborn`, `pandas`, `numpy`
  - ML: `scikit-learn`
  - Cleaning: `pyjanitor`
  - Validation: `pandera`
  - Dev: `jupyter`, `pytest`, `black`, `ruff`, `mypy`, `pre-commit`, `mkdocs-material`, `mkdocstrings[python]`
- Python version requirement (>=3.10)
- Build system configuration
- Tool configurations (ruff, pytest, black, mypy)

**.gitignore**
- Python artifacts (`__pycache__`, `*.pyc`, `.pytest_cache`)
- Virtual environments (`venv/`, `.venv/`, `uv.lock`)
- Jupyter notebooks checkpoints (`.ipynb_checkpoints/`)
- Distribution artifacts (`dist/`, `build/`, `*.egg-info/`, `site/`)
- IDE files (`.vscode/`, `.idea/`)
- Data files (`*.csv`, `*.parquet`, `data/` - with exceptions for examples)

**README.md**
- Project description and goals
- Installation instructions using `uv`
- Quick start examples
- Link to full documentation
- Contributing guidelines
- License information

### 0.2 Project Structure
```
utils/
├── pyproject.toml
├── README.md
├── .gitignore
├── LICENSE
├── .pre-commit-config.yaml
├── .github/
│   └── workflows/
│       ├── test.yml
│       ├── lint.yml
│       └── docs.yml
├── src/
│   └── ds_utils/
│       ├── __init__.py
│       ├── plotting/
│       │   ├── __init__.py
│       │   ├── core.py           # Base plotting utilities
│       │   ├── standard.py       # Line, scatter, bar plots
│       │   ├── distributions.py  # Histograms, density plots
│       │   └── styles.py         # Style management
│       ├── ml_eval/
│       │   ├── __init__.py
│       │   ├── metrics.py        # Summary metrics
│       │   ├── classification.py # Confusion matrix, ROC, PR curves
│       │   ├── regression.py     # Regression-specific plots
│       │   ├── feature_importance.py
│       │   └── learning_curves.py
│       ├── cleaning/
│       │   ├── __init__.py
│       │   ├── extensions.py     # pyjanitor extensions
│       │   └── validation.py     # pandera schemas
│       └── config/
│           ├── __init__.py
│           ├── plot_styles.yaml  # Color schemes, styles
│           ├── logging_config.py # Logging configuration
│           └── defaults.py       # Default configurations
├── tests/
│   ├── __init__.py
│   ├── test_plotting/
│   ├── test_ml_eval/
│   └── test_cleaning/
├── examples/
│   ├── plotting_examples.ipynb
│   ├── ml_evaluation_examples.ipynb
│   └── cleaning_examples.ipynb
├── docs/
│   ├── index.md
│   ├── plotting.md
│   ├── ml_evaluation.md
│   ├── cleaning.md
│   ├── mkdocs.yml
│   └── assets/
│       └── corporate_logo.png
└── assets/
    └── corporate_logo.png        # Logo for plots
```

### 0.3 CI/CD & Quality Assurance
**GitHub Actions Workflows**
- `test.yml`: Run pytest on push/PR across Python versions (3.10, 3.11, 3.12)
- `lint.yml`: Run ruff, black, and mypy
- `docs.yml`: Build and deploy documentation to GitHub Pages

**Pre-commit Hooks**
- Standard hooks (trailing-whitespace, end-of-file-fixer)
- Ruff linting and formatting
- Black formatting
- Mypy type checking

## Phase 1: Configuration & Styling System

### 1.1 Configuration Module (`config/`)

**plot_styles.yaml**
- Define color palettes (primary, secondary, categorical)
- Font configurations
- Figure sizes (small, medium, large, presentation)
- Grid styles
- Corporate color scheme

**defaults.py**
- Load YAML configuration
- Provide functions to get/set configurations
- Locale settings (default to German 'de_DE.UTF-8', but configurable)
- Logo positioning defaults

**logging_config.py**
- Standardized logging configuration
- Console and file handlers
- Log formatting

### 1.2 Questions for Clarification:
- What's your corporate color scheme? (Primary and secondary colors)
- Do you have a specific logo file? (PNG with transparency preferred)
- What size should the logo be on plots? (Small watermark or prominent?)
- Any specific font preferences? (DejaVu Sans is standard for German characters)

## Phase 2: Plotting Utilities (`plotting/`)

### 2.1 Core Module (`core.py`)
**Base functionality:**
- `setup_locale()`: Configure matplotlib for specific locale (default German)
- `add_corporate_logo()`: Add logo to any matplotlib figure
- `apply_corporate_style()`: Apply standard styling to plots
- `save_plot()`: Save with consistent settings (DPI, format, tight_layout)

### 2.2 Standard Plots (`standard.py`)
**Functions to implement:**
- `plot_line()`: Enhanced line plots with localized formatting
  - Multiple series support
  - Automatic legend positioning
  - Localized date formatting on x-axis
  - Corporate styling applied
  
- `plot_scatter()`: Scatter plots with customization
  - Size and color mapping support
  - Regression line option
  - Marginal distributions option
  
- `plot_bar()`: Bar plots (vertical/horizontal)
  - Grouped and stacked options
  - Localized number formatting on axes
  - Error bars support
  - Value labels on bars

### 2.3 Distribution Plots (`distributions.py`)
**Functions to implement:**
- `plot_histogram()`: Enhanced histograms
  - Automatic binning or custom bins
  - Overlay normal distribution option
  - KDE overlay option
  
- `plot_density()`: Density plots
  - Multiple distributions comparison
  - Fill under curve option
  - Rug plots option
  
- `plot_boxplot()`: Box plots with customization
- `plot_violin()`: Violin plots

### 2.4 Style Management (`styles.py`)
- `get_color_palette()`: Retrieve defined color schemes
- `set_plot_context()`: Context manager for temporary style changes
- `create_figure()`: Factory function with standard configuration

## Phase 3: ML Evaluation Utilities (`ml_eval/`)

### 3.1 Metrics Module (`metrics.py`)
**Classification metrics:**
- `classification_summary()`: DataFrame with accuracy, precision, recall, F1
- Support for multi-class and binary classification
- Per-class metrics

**Regression metrics:**
- `regression_summary()`: DataFrame with MAE, RMSE, R², MAPE
- Residual statistics
- Optional grouped analysis

### 3.2 Classification Visualizations (`classification.py`)
**Functions to implement:**
- `plot_confusion_matrix()`:
  - Normalized and absolute values
  - Corporate color scheme
  - Localized number formatting
  - Annotation customization
  
- `plot_roc_curve()`:
  - Multi-class support (OvR or OvO)
  - AUC scores in legend
  - Diagonal reference line
  - Corporate styling
  
- `plot_precision_recall_curve()`:
  - Multi-class support
  - Average precision scores
  - Baseline reference
  
- `plot_calibration_curve()`:
  - Model probability calibration
  - Perfect calibration reference

### 3.3 Regression Visualizations (`regression.py`)
- `plot_residuals()`: Residual vs predicted
- `plot_prediction_error()`: Actual vs predicted
- `plot_residual_distribution()`: Histogram + QQ plot

### 3.4 Feature Importance (`feature_importance.py`)
**Functions supporting:**
- Tree-based models (native importance)
- SHAP values integration
- Permutation importance
- Horizontal bar plots with localized formatting

### 3.5 Learning Curves (`learning_curves.py`)
- `plot_learning_curve()`: Performance vs training size
- `plot_validation_curve()`: Performance vs hyperparameter
- Cross-validation visualization
- Training/validation score comparison

## Phase 4: Data Cleaning & Validation (`cleaning/`)

### 4.1 Pyjanitor Extensions (`extensions.py`)
**Custom chain methods:**
- Locale-specific cleaners (replace commas in numbers, date parsing)
- Common data quality checks
- Outlier detection and handling
- Missing value analysis and visualization
- Data type inference and conversion

### 4.2 Data Validation (`validation.py`)
- `pandera` schema generation helpers
- Common validation checks (email, phone, IDs)
- Data quality reporting

## Phase 5: Documentation

### 5.1 Code Documentation
- Comprehensive docstrings (Google or NumPy style)
- Type hints for all public functions (checked by mypy)
- Examples in docstrings

### 5.2 User Documentation (`docs/`)
**Tools:**
- `MkDocs` with `mkdocs-material` theme
- `mkdocstrings` for automatic API reference

**Structure:**
- **index.md**: Overview, installation, quick start
- **plotting.md**: Complete plotting guide with examples
- **ml_evaluation.md**: ML evaluation utilities guide
- **cleaning.md**: Data cleaning utilities guide
- **api_reference.md**: Auto-generated API docs
- Configuration guide (how to customize styles, colors, logo)

### 5.3 Example Notebooks
**plotting_examples.ipynb:**
- All plot types with sample data
- Styling customization examples
- Logo and corporate branding demonstration

**ml_evaluation_examples.ipynb:**
- Classification example (with scikit-learn model)
- Regression example
- Feature importance examples
- Learning curves

**cleaning_examples.ipynb:**
- Pyjanitor workflow examples
- Custom cleaning operations
- Data quality reports

## Phase 6: Testing

### 6.1 Test Strategy
- Unit tests for all utility functions
- Fixture data for consistent testing
- Visual regression tests (compare generated plots)
- Integration tests for complete workflows
- Type checking with `mypy`

### 6.2 Test Coverage Goals
- Minimum 80% code coverage
- All public API functions tested
- Edge cases covered (empty data, single values, etc.)

## Phase 7: Release & Maintenance
- Semantic versioning strategy
- Changelog management (Keep a Changelog)
- Release process (GitHub Releases, PyPI publishing if applicable)

## Implementation Order Recommendation

### Sprint 1: Foundation (Week 1)
1. Create project structure and CI/CD workflows
2. Set up pyproject.toml with dependencies (including dev tools)
3. Create .gitignore, .pre-commit-config.yaml, and basic README
4. Implement configuration system (plot_styles.yaml, defaults.py, logging)
5. Create core plotting utilities (Locale setup, logo addition)

### Sprint 2: Basic Plotting (Week 2)
1. Implement standard plots (line, scatter, bar)
2. Implement distribution plots (histogram, density)
3. Create plotting_examples.ipynb
4. Write tests for plotting utilities

### Sprint 3: ML Evaluation - Classification (Week 3)
1. Implement metrics module
2. Implement classification visualizations
3. Create classification examples in notebook
4. Write tests for classification utilities

### Sprint 4: ML Evaluation - Advanced (Week 4)
1. Implement regression visualizations
2. Implement feature importance utilities
3. Implement learning curves
4. Complete ml_evaluation_examples.ipynb
5. Write tests

### Sprint 5: Data Cleaning & Validation (Week 5)
1. Set up pyjanitor integration
2. Implement custom cleaning extensions
3. Implement pandera validation helpers
4. Create cleaning_examples.ipynb
5. Write tests

### Sprint 6: Documentation & Polish (Week 6)
1. Set up MkDocs and build documentation site
2. Review and improve all docstrings
3. Final testing, type checking, and bug fixes
4. Create comprehensive README

## Key Technical Decisions

### Localization (German Focus)
Use `locale.setlocale` and matplotlib's localization features for:
- Decimal separator: comma (,)
- Thousands separator: period (.)
- Date format: DD.MM.YYYY
*Note: Make this configurable to allow other locales.*

### Logo Integration
- Store logo in assets folder
- Use `matplotlib.offsetbox.OffsetImage` for positioning
- Provide configurable opacity and size
- Default position: bottom-right corner

### Style Management
- Use matplotlib stylesheets as base
- Override with corporate colors
- Provide context managers for temporary changes
- Make all styling optional (easy to disable)

### Dependency Management with UV
- Use `uv pip install` for development
- Pin major versions, allow minor/patch updates
- Separate dev dependencies from runtime
