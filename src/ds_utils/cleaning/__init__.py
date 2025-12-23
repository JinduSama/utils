"""Data cleaning module for ds_utils.

This module provides data cleaning utilities including pyjanitor extensions
and pandera-based validation helpers.
"""

from ds_utils.cleaning.extensions import (
    clean_column_names,
    clean_german_numbers,
    convert_dtypes,
    detect_outliers,
    drop_duplicate_columns,
    fill_missing_values,
    missing_value_report,
    normalize_text,
    parse_german_dates,
    register_janitor_methods,
    remove_outliers,
)
from ds_utils.cleaning.validation import (
    create_datetime_schema,
    create_email_schema,
    create_german_phone_schema,
    create_german_postal_code_schema,
    create_numeric_schema,
    create_schema,
    create_string_schema,
    data_quality_report,
    generate_validation_report,
    validate_dataframe,
)

__all__ = [
    # Extensions
    "clean_german_numbers",
    "parse_german_dates",
    "clean_column_names",
    "detect_outliers",
    "remove_outliers",
    "missing_value_report",
    "fill_missing_values",
    "convert_dtypes",
    "drop_duplicate_columns",
    "normalize_text",
    "register_janitor_methods",
    # Validation
    "validate_dataframe",
    "create_schema",
    "create_numeric_schema",
    "create_string_schema",
    "create_datetime_schema",
    "create_email_schema",
    "create_german_phone_schema",
    "create_german_postal_code_schema",
    "generate_validation_report",
    "data_quality_report",
]
