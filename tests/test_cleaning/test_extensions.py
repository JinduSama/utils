"""Tests for data cleaning extensions."""

import pandas as pd
import pytest

from ds_utils.cleaning.extensions import (
    clean_column_names,
    clean_german_numbers,
    detect_outliers,
    fill_missing_values,
    missing_value_report,
    normalize_text,
    parse_german_dates,
    remove_outliers,
)


class TestCleanGermanNumbers:
    """Tests for clean_german_numbers function."""

    def test_basic_conversion(self, german_formatted_dataframe):
        """Test basic German number conversion."""
        result = clean_german_numbers(german_formatted_dataframe, "price")

        assert result["price"].iloc[0] == pytest.approx(1234.56)
        assert result["price"].iloc[1] == pytest.approx(789.12)

    def test_large_numbers(self, german_formatted_dataframe):
        """Test conversion of large numbers with multiple thousand separators."""
        result = clean_german_numbers(german_formatted_dataframe, "price")

        assert result["price"].iloc[2] == pytest.approx(2345678.90)

    def test_original_not_modified(self, german_formatted_dataframe):
        """Test that original DataFrame is not modified."""
        original_value = german_formatted_dataframe["price"].iloc[0]
        clean_german_numbers(german_formatted_dataframe, "price")

        assert german_formatted_dataframe["price"].iloc[0] == original_value


class TestParseGermanDates:
    """Tests for parse_german_dates function."""

    def test_basic_date_parsing(self, german_formatted_dataframe):
        """Test basic German date parsing."""
        result = parse_german_dates(german_formatted_dataframe, "date")

        assert pd.api.types.is_datetime64_any_dtype(result["date"])
        assert result["date"].iloc[0].day == 25
        assert result["date"].iloc[0].month == 12
        assert result["date"].iloc[0].year == 2024


class TestCleanColumnNames:
    """Tests for clean_column_names function."""

    def test_snake_case_conversion(self):
        """Test conversion to snake_case."""
        df = pd.DataFrame({"First Name": [1], "Last Name": [2]})
        result = clean_column_names(df)

        assert "first_name" in result.columns
        assert "last_name" in result.columns

    def test_camel_case_conversion(self):
        """Test conversion of CamelCase to snake_case."""
        df = pd.DataFrame({"FirstName": [1], "LastName": [2]})
        result = clean_column_names(df)

        assert "first_name" in result.columns


class TestDetectOutliers:
    """Tests for detect_outliers function."""

    def test_iqr_method(self, dataframe_with_outliers):
        """Test outlier detection with IQR method."""
        result = detect_outliers(dataframe_with_outliers, method="iqr")

        assert "with_outlier_outlier" in result.columns
        assert result["with_outlier_outlier"].iloc[-1]  # 100 is an outlier

    def test_zscore_method(self, dataframe_with_outliers):
        """Test outlier detection with Z-score method."""
        result = detect_outliers(dataframe_with_outliers, method="zscore", threshold=2)

        assert "with_outlier_outlier" in result.columns


class TestRemoveOutliers:
    """Tests for remove_outliers function."""

    def test_outlier_removal(self, dataframe_with_outliers):
        """Test that outliers are removed."""
        result = remove_outliers(dataframe_with_outliers, columns="with_outlier")

        assert len(result) < len(dataframe_with_outliers)
        assert 100 not in result["with_outlier"].values


class TestMissingValueReport:
    """Tests for missing_value_report function."""

    def test_report_structure(self, dataframe_with_missing):
        """Test missing value report structure."""
        result = missing_value_report(dataframe_with_missing)

        assert "column" in result.columns
        assert "missing_count" in result.columns
        assert "missing_percent" in result.columns

    def test_correct_counts(self, dataframe_with_missing):
        """Test correct missing value counts."""
        result = missing_value_report(dataframe_with_missing)

        all_missing_row = result[result["column"] == "all_missing"]
        assert all_missing_row["missing_count"].values[0] == 5


class TestFillMissingValues:
    """Tests for fill_missing_values function."""

    def test_mean_fill(self, dataframe_with_missing):
        """Test filling with mean."""
        result = fill_missing_values(dataframe_with_missing, strategy="mean")

        assert not result["with_missing"].isna().any()

    def test_constant_fill(self, dataframe_with_missing):
        """Test filling with constant value."""
        result = fill_missing_values(
            dataframe_with_missing, strategy="constant", fill_value=0
        )

        assert not result["with_missing"].isna().any()


class TestNormalizeText:
    """Tests for normalize_text function."""

    def test_lowercase_normalization(self):
        """Test lowercase normalization."""
        df = pd.DataFrame({"text": ["HELLO", "World", "TEST"]})
        result = normalize_text(df, "text")

        assert result["text"].iloc[0] == "hello"
        assert result["text"].iloc[1] == "world"

    def test_strip_whitespace(self):
        """Test whitespace stripping."""
        df = pd.DataFrame({"text": ["  hello  ", "world  "]})
        result = normalize_text(df, "text", strip=True)

        assert result["text"].iloc[0] == "hello"
        assert result["text"].iloc[1] == "world"
