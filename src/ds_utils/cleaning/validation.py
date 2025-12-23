"""Data validation utilities for ds_utils.

This module provides helpers for data validation using pandera schemas.
"""

from typing import Any, Callable, Type

import numpy as np
import pandas as pd

try:
    import pandera as pa
    from pandera import Column, Check, DataFrameSchema
    from pandera.errors import SchemaError

    PANDERA_AVAILABLE = True
except ImportError:
    PANDERA_AVAILABLE = False

from ds_utils.config.logging_config import get_logger

logger = get_logger("cleaning.validation")


def validate_dataframe(
    df: pd.DataFrame,
    schema: Any,
    lazy: bool = True,
) -> pd.DataFrame:
    """Validate a DataFrame against a pandera schema.

    Args:
        df: DataFrame to validate.
        schema: Pandera DataFrameSchema or callable that returns one.
        lazy: If True, collects all validation errors. If False, fails on first error.

    Returns:
        Validated DataFrame.

    Raises:
        SchemaError: If validation fails.
        ImportError: If pandera is not installed.

    Example:
        >>> schema = create_schema({'age': 'int', 'name': 'str'})
        >>> validated_df = validate_dataframe(df, schema)
    """
    if not PANDERA_AVAILABLE:
        raise ImportError("pandera is required for validation. Install with: pip install pandera")

    if callable(schema) and not isinstance(schema, DataFrameSchema):
        schema = schema()

    try:
        return schema.validate(df, lazy=lazy)
    except SchemaError as e:
        logger.error(f"Schema validation failed: {e}")
        raise


def create_schema(
    columns: dict[str, Any],
    strict: bool = False,
    coerce: bool = True,
) -> Any:
    """Create a pandera DataFrameSchema from a column specification.

    Args:
        columns: Dictionary mapping column names to their specifications.
            Values can be:
            - String dtype: 'int', 'float', 'str', 'bool', 'datetime'
            - Dict with 'dtype' and optional 'checks', 'nullable', etc.
        strict: If True, raise error if extra columns are present.
        coerce: If True, coerce values to specified dtype.

    Returns:
        Pandera DataFrameSchema.

    Example:
        >>> schema = create_schema({
        ...     'age': {'dtype': 'int', 'checks': [Check.ge(0), Check.le(120)]},
        ...     'name': 'str',
        ...     'score': {'dtype': 'float', 'nullable': True},
        ... })
    """
    if not PANDERA_AVAILABLE:
        raise ImportError("pandera is required for validation. Install with: pip install pandera")

    dtype_map = {
        "int": int,
        "float": float,
        "str": str,
        "string": str,
        "bool": bool,
        "boolean": bool,
        "datetime": "datetime64[ns]",
        "date": "datetime64[ns]",
    }

    schema_columns = {}

    for col_name, spec in columns.items():
        if isinstance(spec, str):
            # Simple dtype specification
            dtype = dtype_map.get(spec.lower(), spec)
            schema_columns[col_name] = Column(dtype, coerce=coerce)
        elif isinstance(spec, dict):
            # Detailed specification
            dtype = spec.get("dtype", "object")
            if isinstance(dtype, str):
                dtype = dtype_map.get(dtype.lower(), dtype)

            col_kwargs = {
                "dtype": dtype,
                "nullable": spec.get("nullable", False),
                "coerce": spec.get("coerce", coerce),
            }

            # Add checks if specified
            if "checks" in spec:
                col_kwargs["checks"] = spec["checks"]

            # Add unique constraint
            if spec.get("unique", False):
                col_kwargs["unique"] = True

            schema_columns[col_name] = Column(**col_kwargs)
        else:
            # Assume it's already a Column
            schema_columns[col_name] = spec

    return DataFrameSchema(schema_columns, strict=strict, coerce=coerce)


def create_numeric_schema(
    column: str,
    min_value: float | None = None,
    max_value: float | None = None,
    nullable: bool = False,
    dtype: str = "float",
) -> Any:
    """Create a schema for a numeric column with range validation.

    Args:
        column: Column name.
        min_value: Minimum allowed value.
        max_value: Maximum allowed value.
        nullable: Whether null values are allowed.
        dtype: Data type ('int' or 'float').

    Returns:
        Pandera DataFrameSchema.

    Example:
        >>> schema = create_numeric_schema('age', min_value=0, max_value=120)
    """
    if not PANDERA_AVAILABLE:
        raise ImportError("pandera is required for validation. Install with: pip install pandera")

    checks = []
    if min_value is not None:
        checks.append(Check.ge(min_value))
    if max_value is not None:
        checks.append(Check.le(max_value))

    col_dtype = int if dtype == "int" else float

    return DataFrameSchema({
        column: Column(col_dtype, checks=checks, nullable=nullable, coerce=True)
    })


def create_string_schema(
    column: str,
    pattern: str | None = None,
    min_length: int | None = None,
    max_length: int | None = None,
    nullable: bool = False,
    allowed_values: list[str] | None = None,
) -> Any:
    """Create a schema for a string column with pattern validation.

    Args:
        column: Column name.
        pattern: Regex pattern the values must match.
        min_length: Minimum string length.
        max_length: Maximum string length.
        nullable: Whether null values are allowed.
        allowed_values: List of allowed values (categorical).

    Returns:
        Pandera DataFrameSchema.

    Example:
        >>> # Email validation
        >>> schema = create_string_schema('email', pattern=r'^[\w\.-]+@[\w\.-]+\.\w+$')

        >>> # Categorical
        >>> schema = create_string_schema('status', allowed_values=['active', 'inactive'])
    """
    if not PANDERA_AVAILABLE:
        raise ImportError("pandera is required for validation. Install with: pip install pandera")

    checks = []

    if pattern is not None:
        checks.append(Check.str_matches(pattern))

    if min_length is not None:
        checks.append(Check.str_length(min_value=min_length))

    if max_length is not None:
        checks.append(Check.str_length(max_value=max_length))

    if allowed_values is not None:
        checks.append(Check.isin(allowed_values))

    return DataFrameSchema({
        column: Column(str, checks=checks, nullable=nullable, coerce=True)
    })


def create_datetime_schema(
    column: str,
    min_date: str | pd.Timestamp | None = None,
    max_date: str | pd.Timestamp | None = None,
    nullable: bool = False,
) -> Any:
    """Create a schema for a datetime column with range validation.

    Args:
        column: Column name.
        min_date: Minimum allowed date.
        max_date: Maximum allowed date.
        nullable: Whether null values are allowed.

    Returns:
        Pandera DataFrameSchema.

    Example:
        >>> schema = create_datetime_schema('created_at', min_date='2020-01-01')
    """
    if not PANDERA_AVAILABLE:
        raise ImportError("pandera is required for validation. Install with: pip install pandera")

    checks = []

    if min_date is not None:
        min_ts = pd.Timestamp(min_date)
        checks.append(Check(lambda x: x >= min_ts, element_wise=True))

    if max_date is not None:
        max_ts = pd.Timestamp(max_date)
        checks.append(Check(lambda x: x <= max_ts, element_wise=True))

    return DataFrameSchema({
        column: Column("datetime64[ns]", checks=checks, nullable=nullable, coerce=True)
    })


# Common validation patterns
EMAIL_PATTERN = r"^[\w\.-]+@[\w\.-]+\.\w+$"
PHONE_PATTERN_DE = r"^(\+49|0)[1-9]\d{1,14}$"
IBAN_PATTERN_DE = r"^DE\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{2}$"
POSTAL_CODE_PATTERN_DE = r"^\d{5}$"


def create_email_schema(column: str, nullable: bool = False) -> Any:
    """Create a schema for email validation.

    Args:
        column: Column name.
        nullable: Whether null values are allowed.

    Returns:
        Pandera DataFrameSchema.
    """
    return create_string_schema(column, pattern=EMAIL_PATTERN, nullable=nullable)


def create_german_phone_schema(column: str, nullable: bool = False) -> Any:
    """Create a schema for German phone number validation.

    Args:
        column: Column name.
        nullable: Whether null values are allowed.

    Returns:
        Pandera DataFrameSchema.
    """
    return create_string_schema(column, pattern=PHONE_PATTERN_DE, nullable=nullable)


def create_german_postal_code_schema(column: str, nullable: bool = False) -> Any:
    """Create a schema for German postal code validation (5 digits).

    Args:
        column: Column name.
        nullable: Whether null values are allowed.

    Returns:
        Pandera DataFrameSchema.
    """
    return create_string_schema(column, pattern=POSTAL_CODE_PATTERN_DE, nullable=nullable)


def generate_validation_report(
    df: pd.DataFrame,
    schema: Any,
) -> pd.DataFrame:
    """Generate a detailed validation report.

    Args:
        df: DataFrame to validate.
        schema: Pandera schema to validate against.

    Returns:
        DataFrame with validation errors.

    Example:
        >>> report = generate_validation_report(df, schema)
        >>> print(report[report['valid'] == False])
    """
    if not PANDERA_AVAILABLE:
        raise ImportError("pandera is required for validation. Install with: pip install pandera")

    errors = []

    try:
        schema.validate(df, lazy=True)
    except SchemaError as e:
        for failure in e.failure_cases.itertuples():
            errors.append({
                "column": failure.column if hasattr(failure, "column") else None,
                "index": failure.index if hasattr(failure, "index") else None,
                "check": failure.check if hasattr(failure, "check") else None,
                "failure_case": failure.failure_case if hasattr(failure, "failure_case") else None,
            })

    if errors:
        return pd.DataFrame(errors)
    else:
        return pd.DataFrame(columns=["column", "index", "check", "failure_case"])


def data_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Generate a comprehensive data quality report.

    Args:
        df: DataFrame to analyze.

    Returns:
        DataFrame with quality metrics for each column.
    """
    report_data = []

    for col in df.columns:
        col_data = df[col]
        n_total = len(col_data)
        n_missing = col_data.isna().sum()
        n_unique = col_data.nunique()

        col_report = {
            "column": col,
            "dtype": str(col_data.dtype),
            "non_null_count": n_total - n_missing,
            "null_count": n_missing,
            "null_percent": (n_missing / n_total) * 100 if n_total > 0 else 0,
            "unique_count": n_unique,
            "unique_percent": (n_unique / n_total) * 100 if n_total > 0 else 0,
        }

        # Numeric-specific stats
        if pd.api.types.is_numeric_dtype(col_data):
            col_report["min"] = col_data.min()
            col_report["max"] = col_data.max()
            col_report["mean"] = col_data.mean()
            col_report["std"] = col_data.std()
        else:
            col_report["min"] = None
            col_report["max"] = None
            col_report["mean"] = None
            col_report["std"] = None

        # Check for potential issues
        issues = []
        if n_missing > 0:
            issues.append(f"{n_missing} missing")
        if n_unique == 1:
            issues.append("constant")
        if n_unique == n_total and n_total > 1:
            issues.append("all unique")

        col_report["issues"] = "; ".join(issues) if issues else "none"

        report_data.append(col_report)

    return pd.DataFrame(report_data)
