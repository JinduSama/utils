"""Tests for data validation utilities."""

import pandas as pd
import pytest

from ds_utils.cleaning.validation import (
    create_numeric_schema,
    create_schema,
    create_string_schema,
    data_quality_report,
)

# Skip tests if pandera is not installed
pytest.importorskip("pandera")


class TestCreateSchema:
    """Tests for create_schema function."""

    def test_simple_schema(self):
        """Test creating a simple schema."""
        schema = create_schema(
            {
                "age": "int",
                "name": "str",
            }
        )

        assert schema is not None
        assert "age" in schema.columns
        assert "name" in schema.columns

    def test_schema_validation_pass(self):
        """Test that valid data passes schema validation."""
        schema = create_schema(
            {
                "age": "int",
                "name": "str",
            }
        )

        df = pd.DataFrame({"age": [25, 30], "name": ["Alice", "Bob"]})
        result = schema.validate(df)

        assert len(result) == 2


class TestCreateNumericSchema:
    """Tests for create_numeric_schema function."""

    def test_range_validation(self):
        """Test numeric range validation."""
        schema = create_numeric_schema("age", min_value=0, max_value=120)

        df_valid = pd.DataFrame({"age": [25, 50, 75]})
        result = schema.validate(df_valid)

        assert len(result) == 3

    def test_range_validation_fails(self):
        """Test that out-of-range values fail validation."""
        from pandera.errors import SchemaError

        schema = create_numeric_schema("age", min_value=0, max_value=120)

        df_invalid = pd.DataFrame({"age": [25, 150]})

        with pytest.raises(SchemaError):
            schema.validate(df_invalid)


class TestCreateStringSchema:
    """Tests for create_string_schema function."""

    def test_pattern_validation(self):
        """Test string pattern validation."""
        schema = create_string_schema("code", pattern=r"^[A-Z]{3}$")

        df_valid = pd.DataFrame({"code": ["ABC", "XYZ"]})
        result = schema.validate(df_valid)

        assert len(result) == 2

    def test_allowed_values_validation(self):
        """Test allowed values validation."""
        schema = create_string_schema("status", allowed_values=["active", "inactive"])

        df_valid = pd.DataFrame({"status": ["active", "inactive"]})
        result = schema.validate(df_valid)

        assert len(result) == 2


class TestDataQualityReport:
    """Tests for data_quality_report function."""

    def test_report_structure(self, dataframe_with_missing):
        """Test data quality report structure."""
        result = data_quality_report(dataframe_with_missing)

        assert "column" in result.columns
        assert "dtype" in result.columns
        assert "null_count" in result.columns
        assert "null_percent" in result.columns
        assert "unique_count" in result.columns
        assert "issues" in result.columns

    def test_identifies_issues(self, dataframe_with_missing):
        """Test that issues are identified."""
        result = data_quality_report(dataframe_with_missing)

        all_missing_row = result[result["column"] == "all_missing"]
        assert "missing" in all_missing_row["issues"].values[0]
