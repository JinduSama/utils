# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial release of DS Utils library
- Plotting module with corporate styling and German locale support
- ML evaluation module for classification and regression metrics
- Data cleaning module with German data format support
- Validation module using Pandera schemas
- Example Jupyter notebooks
- Comprehensive documentation
- GitHub Actions CI/CD workflows
- Pre-commit hooks configuration

## [0.1.0] - 2024-XX-XX

### Added

#### Plotting Module
- `apply_corporate_style()` - Apply corporate styling to all plots
- `setup_locale()` - Configure locale for number formatting
- `format_number()` - Format numbers according to locale
- `save_plot()` - Save figures in multiple formats
- `add_corporate_logo()` - Add logo to figures
- `create_figure()` - Create styled figures
- `plot_context()` - Context manager for plot settings
- `plot_line()` - Line plots
- `plot_scatter()` - Scatter plots
- `plot_bar()` - Bar charts
- `plot_time_series()` - Time series plots
- `plot_histogram()` - Histograms
- `plot_density()` - Density plots
- `plot_boxplot()` - Boxplots
- `plot_violin()` - Violin plots
- `plot_qq()` - Q-Q plots
- `get_color_palette()` - Get color palettes
- `create_colormap()` - Create colormaps

#### ML Evaluation Module
- `classification_summary()` - Classification metrics summary
- `regression_summary()` - Regression metrics summary
- `cross_validation_summary()` - CV results summary
- `calculate_lift()` - Lift calculation
- `plot_confusion_matrix()` - Confusion matrix visualization
- `plot_roc_curve()` - ROC curve visualization
- `plot_precision_recall_curve()` - PR curve visualization
- `plot_calibration_curve()` - Calibration curve
- `plot_classification_report()` - Classification report heatmap
- `plot_residuals()` - Residual plots
- `plot_prediction_error()` - Actual vs predicted
- `plot_residual_distribution()` - Residual distribution analysis
- `plot_residuals_vs_feature()` - Residuals vs feature
- `plot_feature_importance()` - Feature importance bar chart
- `plot_permutation_importance()` - Permutation importance
- `plot_shap_summary()` - SHAP summary plot
- `plot_feature_importance_comparison()` - Compare importances
- `plot_learning_curve()` - Learning curves
- `plot_validation_curve()` - Validation curves
- `plot_cv_results()` - CV results visualization
- `plot_cv_comparison()` - Model comparison

#### Cleaning Module
- `clean_german_numbers()` - Parse German number format
- `parse_german_dates()` - Parse German date format
- `clean_column_names()` - Standardize column names
- `detect_outliers()` - Detect outliers (IQR, z-score, MAD)
- `remove_outliers()` - Remove outliers
- `missing_value_report()` - Missing value analysis
- `fill_missing_values()` - Fill missing values with strategies
- `convert_dtypes()` - Convert data types
- `normalize_text()` - Normalize text values
- `register_janitor_methods()` - Register pyjanitor extensions
- `validate_dataframe()` - Validate with Pandera schema
- `create_schema()` - Create validation schema
- `create_numeric_schema()` - Numeric column schema
- `create_string_schema()` - String column schema
- `create_datetime_schema()` - Datetime column schema
- `create_email_schema()` - Email validation schema
- `create_german_phone_schema()` - German phone validation
- `create_german_postal_code_schema()` - German postal code validation
- `generate_validation_report()` - Detailed validation report
- `data_quality_report()` - Data quality analysis

#### Configuration
- YAML-based configuration
- Customizable color palettes
- Font configuration
- Locale settings
- Logo integration
- Output settings

#### Infrastructure
- GitHub Actions workflows for CI/CD
- Pre-commit hooks
- pytest test suite
- MkDocs documentation
- Type hints throughout

[Unreleased]: https://github.com/your-org/ds-utils/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/your-org/ds-utils/releases/tag/v0.1.0
