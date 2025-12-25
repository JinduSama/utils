"""Data cleaning extensions for ds_utils.

This module provides enhanced data cleaning functions with pyjanitor integration
and locale-specific cleaners for German data formats.
"""

import re
from typing import Any

import numpy as np
import pandas as pd

from ds_utils.config import get_decimal_separator, get_thousands_separator
from ds_utils.config.logging_config import get_logger

logger = get_logger("cleaning.extensions")


def clean_german_numbers(
    data: pd.DataFrame | pd.Series,
    columns: str | list[str] | None = None,
    thousands_sep: str | None = None,
    decimal_sep: str | None = None,
    errors: str = "coerce",
) -> pd.DataFrame | pd.Series:
    """Convert German-formatted numbers to standard floats.

    German number format uses comma as decimal separator and period as thousands
    separator.
    Example: "1.234,56" -> 1234.56

    Args:
        data: DataFrame or Series to clean.
        columns: Column name(s) to convert (only if data is a DataFrame).
        thousands_sep: Thousands separator to replace. If None, uses configured default.
        decimal_sep: Decimal separator to replace. If None, uses configured default.
        errors: How to handle conversion errors ('coerce', 'raise', 'ignore').

    Returns:
        DataFrame or Series with converted values.

    Example:
        >>> df = pd.DataFrame({'price': ['1.234,56', '789,12']})
        >>> df = clean_german_numbers(df, 'price')
        >>> s = pd.Series(['1.234,56', '789,12'])
        >>> s_clean = clean_german_numbers(s)
    """
    if thousands_sep is None:
        thousands_sep = get_thousands_separator()
    if decimal_sep is None:
        decimal_sep = get_decimal_separator()

    if isinstance(data, pd.Series):
        # Convert to string for processing
        s = data.astype(str)
        # Remove thousands separator
        s = s.str.replace(thousands_sep, "", regex=False)
        # Replace decimal separator with period
        s = s.str.replace(decimal_sep, ".", regex=False)
        # Convert to numeric
        return pd.to_numeric(s, errors=errors)

    # Handle DataFrame
    df = data.copy()

    if columns is None:
        logger.warning("No columns specified for clean_german_numbers on DataFrame")
        return df

    if isinstance(columns, str):
        columns = [columns]

    for col in columns:
        if col not in df.columns:
            logger.warning(f"Column '{col}' not found in DataFrame")
            continue

        # Convert to string for processing
        df[col] = df[col].astype(str)

        # Remove thousands separator
        df[col] = df[col].str.replace(thousands_sep, "", regex=False)

        # Replace decimal separator with period
        df[col] = df[col].str.replace(decimal_sep, ".", regex=False)

        # Convert to numeric
        df[col] = pd.to_numeric(df[col], errors=errors)

    return df


def parse_german_dates(
    data: pd.DataFrame | pd.Series,
    columns: str | list[str] | None = None,
    date_format: str = "%d.%m.%Y",
    errors: str = "coerce",
) -> pd.DataFrame | pd.Series:
    """Parse German-formatted dates.

    Args:
        data: DataFrame or Series to clean.
        columns: Column name(s) to convert (only if data is a DataFrame).
        date_format: Expected date format (default: DD.MM.YYYY).
        errors: How to handle conversion errors.

    Returns:
        DataFrame or Series with parsed date values.

    Example:
        >>> df = pd.DataFrame({'date': ['25.12.2024', '01.01.2025']})
        >>> df = parse_german_dates(df, 'date')
        >>> s = pd.Series(['25.12.2024', '01.01.2025'])
        >>> s_clean = parse_german_dates(s)
    """
    if isinstance(data, pd.Series):
        return pd.to_datetime(data, format=date_format, errors=errors)

    # Handle DataFrame
    df = data.copy()

    if columns is None:
        logger.warning("No columns specified for parse_german_dates on DataFrame")
        return df

    if isinstance(columns, str):
        columns = [columns]

    for col in columns:
        if col not in df.columns:
            logger.warning(f"Column '{col}' not found in DataFrame")
            continue

        df[col] = pd.to_datetime(df[col], format=date_format, errors=errors)

    return df


def clean_column_names(
    df: pd.DataFrame,
    case: str = "snake",
    strip_chars: str | None = None,
    replace_spaces: str = "_",
) -> pd.DataFrame:
    """Clean and standardize column names.

    Args:
        df: DataFrame to clean.
        case: Target case ('snake', 'lower', 'upper', 'title').
        strip_chars: Characters to strip from names.
        replace_spaces: Character to replace spaces with.

    Returns:
        DataFrame with cleaned column names.

    Example:
        >>> df = pd.DataFrame({'First Name': [1], 'Last Name': [2]})
        >>> df = clean_column_names(df)
        >>> print(df.columns.tolist())
        ['first_name', 'last_name']
    """
    df = df.copy()

    def clean_name(name: str) -> str:
        # Strip specified characters
        if strip_chars:
            name = name.strip(strip_chars)

        # Replace spaces
        name = name.replace(" ", replace_spaces)

        # Remove special characters except underscores
        name = re.sub(r"[^\w\s]", "", name)

        # Apply case transformation
        if case == "snake":
            # Convert CamelCase to snake_case
            name = re.sub(r"([a-z])([A-Z])", r"\1_\2", name)
            name = name.lower()
        elif case == "lower":
            name = name.lower()
        elif case == "upper":
            name = name.upper()
        elif case == "title":
            name = name.title()

        # Remove multiple underscores
        name = re.sub(r"_+", "_", name)

        return name.strip("_")

    df.columns = [clean_name(str(col)) for col in df.columns]
    return df


def detect_outliers(
    data: pd.DataFrame | pd.Series,
    columns: str | list[str] | None = None,
    method: str = "iqr",
    threshold: float = 1.5,
) -> pd.DataFrame | pd.Series:
    """Detect outliers in numeric columns.

    Args:
        data: DataFrame or Series to analyze.
        columns: Column name(s) to check (only if data is a DataFrame).
            If None, checks all numeric columns.
        method: Detection method ('iqr', 'zscore', 'modified_zscore').
        threshold: Threshold for outlier detection.
            For IQR: multiplier (default 1.5)
            For Z-score: number of standard deviations (default 3)

    Returns:
        DataFrame or Series with boolean values indicating outliers.

    Example:
        >>> df = pd.DataFrame({'values': [1, 2, 3, 100, 4, 5]})
        >>> outliers = detect_outliers(df, 'values')
        >>> s = pd.Series([1, 2, 3, 100, 4, 5])
        >>> s_outliers = detect_outliers(s)
    """

    def _get_outlier_mask(s: pd.Series) -> pd.Series:
        values = s.dropna()
        if values.empty:
            return pd.Series(False, index=s.index)

        if method == "iqr":
            q1 = values.quantile(0.25)
            q3 = values.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - threshold * iqr
            upper = q3 + threshold * iqr
            return (s < lower) | (s > upper)

        elif method == "zscore":
            z_scores = np.abs((s - values.mean()) / values.std())
            return z_scores > threshold

        elif method == "modified_zscore":
            median = values.median()
            mad = np.median(np.abs(values - median))
            if mad == 0:
                return pd.Series(False, index=s.index)
            modified_z = 0.6745 * (s - median) / mad
            return np.abs(modified_z) > threshold

        return pd.Series(False, index=s.index)

    if isinstance(data, pd.Series):
        return _get_outlier_mask(data)

    # Handle DataFrame
    df = data
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    elif isinstance(columns, str):
        columns = [columns]

    result = pd.DataFrame(index=df.index)
    for col in columns:
        if col not in df.columns:
            continue
        result[f"{col}_outlier"] = _get_outlier_mask(df[col])

    return result


def remove_outliers(
    data: pd.DataFrame | pd.Series,
    columns: str | list[str] | None = None,
    method: str = "iqr",
    threshold: float = 1.5,
) -> pd.DataFrame | pd.Series:
    """Remove rows with outliers.

    Args:
        data: DataFrame or Series to clean.
        columns: Column name(s) to check (only if data is a DataFrame).
        method: Detection method ('iqr', 'zscore', 'modified_zscore').
        threshold: Threshold for outlier detection.

    Returns:
        DataFrame or Series with outlier rows removed.
    """
    outliers = detect_outliers(data, columns, method, threshold)

    if isinstance(data, pd.Series):
        mask = ~outliers
        removed_count = (~mask).sum()
        if removed_count > 0:
            logger.info(f"Removed {removed_count} outliers from Series")
        return data[mask].copy()

    # Handle DataFrame
    mask = ~outliers.any(axis=1)
    removed_count = (~mask).sum()

    if removed_count > 0:
        logger.info(f"Removed {removed_count} rows with outliers")

    return data[mask].copy()


def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Generate a report on missing values.

    Args:
        df: DataFrame to analyze.

    Returns:
        DataFrame with missing value statistics per column.

    Example:
        >>> report = missing_value_report(df)
        >>> print(report)
    """
    total = len(df)
    missing = df.isnull().sum()
    missing_pct = (missing / total) * 100

    report = pd.DataFrame(
        {
            "column": df.columns,
            "missing_count": missing.values,
            "missing_percent": missing_pct.values,
            "dtype": df.dtypes.values,
            "non_null_count": total - missing.values,
        }
    )

    report = report.sort_values("missing_percent", ascending=False)
    report = report.reset_index(drop=True)

    return report


def fill_missing_values(
    df: pd.DataFrame,
    strategy: str | dict[str, str] = "mean",
    fill_value: Any = None,
    strategies: dict[str, str] | None = None,
    fill_values: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Fill missing values with specified strategy.

    Args:
        df: DataFrame to fill.
        strategy: Fill strategy. Can be:
            - String: 'mean', 'median', 'mode', 'ffill', 'bfill', 'constant'
            - Dict: {column: strategy} for column-specific strategies
        fill_value: Value to use when strategy is 'constant'.
        strategies: Alias for strategy (for compatibility).
        fill_values: Dictionary of column-specific fill values for 'constant' strategy.

    Returns:
        DataFrame with filled missing values.
    """
    df = df.copy()

    # Use strategies if provided
    if strategies is not None:
        strategy = strategies

    if isinstance(strategy, dict):
        for col, strat in strategy.items():
            if col not in df.columns:
                continue

            # Get specific fill value for this column if available
            col_fill_value = fill_value
            if fill_values and col in fill_values:
                col_fill_value = fill_values[col]

            df[col] = _fill_column(df[col], strat, col_fill_value)
    else:
        for col in df.columns:
            df[col] = _fill_column(df[col], strategy, fill_value)

    return df


def _fill_column(
    series: pd.Series,
    strategy: str,
    fill_value: Any = None,
) -> pd.Series:
    """Fill missing values in a single column."""
    if strategy == "mean":
        if pd.api.types.is_numeric_dtype(series):
            return series.fillna(series.mean())
    elif strategy == "median":
        if pd.api.types.is_numeric_dtype(series):
            return series.fillna(series.median())
    elif strategy == "mode":
        mode_val = series.mode()
        if len(mode_val) > 0:
            return series.fillna(mode_val[0])
    elif strategy == "ffill":
        return series.ffill()
    elif strategy == "bfill":
        return series.bfill()
    elif strategy == "constant":
        return series.fillna(fill_value)

    return series


def convert_dtypes(
    df: pd.DataFrame,
    dtype_map: dict[str, str] | None = None,
    infer: bool = True,
) -> pd.DataFrame:
    """Convert column data types.

    Args:
        df: DataFrame to convert.
        dtype_map: Dictionary mapping column names to target dtypes.
        infer: Whether to use pandas' convert_dtypes for type inference.

    Returns:
        DataFrame with converted types.

    Example:
        >>> df = convert_dtypes(df, {'age': 'int', 'price': 'float'})
    """
    df = df.copy()

    if infer:
        df = df.convert_dtypes()

    if dtype_map:
        for col, dtype in dtype_map.items():
            if col in df.columns:
                try:
                    df[col] = df[col].astype(dtype)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Could not convert {col} to {dtype}: {e}")

    return df


def drop_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate columns (columns with identical values).

    Args:
        df: DataFrame to clean.

    Returns:
        DataFrame with duplicate columns removed.
    """
    # Transpose, drop duplicates, transpose back
    df_t = df.T
    df_t = df_t.drop_duplicates()
    df_clean = df_t.T

    dropped = set(df.columns) - set(df_clean.columns)
    if dropped:
        logger.info(f"Dropped duplicate columns: {dropped}")

    return df_clean


def normalize_text(
    data: pd.DataFrame | pd.Series,
    columns: str | list[str] | None = None,
    case: str | None = None,
    lowercase: bool = True,
    strip: bool = True,
    remove_extra_spaces: bool = True,
) -> pd.DataFrame | pd.Series:
    """Normalize text in string columns.

    Args:
        data: DataFrame or Series to clean.
        columns: Column name(s) to normalize (only if data is a DataFrame).
        case: Target case ('lower', 'upper', 'title', 'snake'). If provided, overrides
            lowercase.
        lowercase: Whether to convert to lowercase (default: True).
        strip: Whether to strip leading/trailing whitespace (default: True).
        remove_extra_spaces: Whether to replace multiple spaces with single space
            (default: True).

    Returns:
        DataFrame or Series with normalized text.
    """

    def _normalize_series(s: pd.Series) -> pd.Series:
        if s.dtype != "object" and not pd.api.types.is_string_dtype(s):
            return s

        # Apply case transformation
        if case == "lower":
            s = s.str.lower()
        elif case == "upper":
            s = s.str.upper()
        elif case == "title":
            s = s.str.title()
        elif case == "snake":
            s = (
                s.str.replace(r"([a-z])([A-Z])", r"\1_\2", regex=True)
                .str.lower()
                .str.replace(r"\s+", "_", regex=True)
            )
        elif lowercase:
            s = s.str.lower()

        if strip:
            s = s.str.strip()
        if remove_extra_spaces:
            s = s.str.replace(r"\s+", " ", regex=True)

        return s

    if isinstance(data, pd.Series):
        return _normalize_series(data)

    # Handle DataFrame
    df = data.copy()

    if columns is None:
        logger.warning("No columns specified for normalize_text on DataFrame")
        return df

    if isinstance(columns, str):
        columns = [columns]

    for col in columns:
        if col not in df.columns:
            continue
        df[col] = _normalize_series(df[col])

    return df


# Pyjanitor-style method chaining
def register_janitor_methods() -> None:
    """Register custom methods for pandas DataFrames (pyjanitor style).

    This adds methods like df.clean_german_numbers() for method chaining.
    """
    try:
        import janitor  # noqa: F401

        # Register methods
        pd.DataFrame.clean_german_numbers = clean_german_numbers
        pd.DataFrame.parse_german_dates = parse_german_dates
        pd.DataFrame.clean_column_names = clean_column_names
        pd.DataFrame.detect_outliers = detect_outliers
        pd.DataFrame.remove_outliers = remove_outliers
        pd.DataFrame.missing_value_report = missing_value_report
        pd.DataFrame.fill_missing_values = fill_missing_values
        pd.DataFrame.normalize_text = normalize_text

        logger.info("Registered custom janitor methods on DataFrame")

    except ImportError:
        logger.debug("pyjanitor not installed. Skipping method registration.")
