"""DS Utils - Data Science Utilities Library.

A Python utilities library for data science and machine learning,
focusing on plotting, ML performance evaluation, and data cleaning.

Modules:
    - plotting: Enhanced plotting utilities with corporate styling
    - ml_eval: Machine learning model evaluation and visualization
    - cleaning: Data cleaning and validation utilities
    - config: Configuration management and logging

Example:
    >>> from ds_utils.plotting import plot_line, apply_corporate_style
    >>> from ds_utils.ml_eval import classification_summary, plot_confusion_matrix
    >>> from ds_utils.cleaning import clean_german_numbers, validate_dataframe
"""

__version__ = "0.1.0"
__author__ = "Data Science Team"

# Import main modules for convenience
from ds_utils import cleaning, config, ml_eval, plotting

__all__ = [
    "plotting",
    "ml_eval",
    "cleaning",
    "config",
    "__version__",
]
