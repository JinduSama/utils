# Data Cleaning Module

The cleaning module provides utilities for data cleaning, transformation, and validation with special support for German data formats.

## Overview

The cleaning module is organized into two main submodules:

- **extensions** - pandas/pyjanitor extensions for data cleaning
- **validation** - Data validation with Pandera schemas

## Getting Started

```python
from ds_utils.cleaning import (
    # Data cleaning
    clean_german_numbers,
    parse_german_dates,
    clean_column_names,
    detect_outliers,
    remove_outliers,
    missing_value_report,
    fill_missing_values,
    convert_dtypes,
    normalize_text,
    register_janitor_methods,
    # Validation
    validate_dataframe,
    create_schema,
    data_quality_report,
)

# Register custom janitor methods
register_janitor_methods()
```

## German Data Cleaning

### `clean_german_numbers(series)`

Converts German-formatted numbers to float.

German format uses:
- Comma (`,`) as decimal separator
- Period (`.`) as thousands separator

```python
import pandas as pd
from ds_utils.cleaning import clean_german_numbers

df = pd.DataFrame({
    "price": ["1.234,56", "2.345,67", "12.345,00"],
    "quantity": ["1.000", "2.500", "10.000"],
})

df["price_clean"] = clean_german_numbers(df["price"])
df["quantity_clean"] = clean_german_numbers(df["quantity"])

# Result:
# price_clean: [1234.56, 2345.67, 12345.0]
# quantity_clean: [1000.0, 2500.0, 10000.0]
```

### `parse_german_dates(series, format="%d.%m.%Y")`

Parses German-formatted dates.

German date format: `DD.MM.YYYY`

```python
from ds_utils.cleaning import parse_german_dates

df = pd.DataFrame({
    "date": ["15.03.2024", "01.12.2023", "22.06.2024"],
})

df["date_clean"] = parse_german_dates(df["date"])
# Result: datetime64 column
```

**Custom formats:**

```python
# With time
df["datetime"] = parse_german_dates(
    df["datetime_str"], 
    format="%d.%m.%Y %H:%M:%S"
)

# Different separator
df["date"] = parse_german_dates(
    df["date_str"], 
    format="%d-%m-%Y"
)
```

## Column Cleaning

### `clean_column_names(df, case="snake")`

Standardizes column names.

```python
from ds_utils.cleaning import clean_column_names

df = pd.DataFrame({
    "Product Name  ": [1, 2],
    " Sales (USD)": [100, 200],
    "Date Of Purchase": ["2024-01-01", "2024-01-02"],
})

df_clean = clean_column_names(df)
# Columns: ["product_name", "sales_usd", "date_of_purchase"]

# Alternative cases
df_clean = clean_column_names(df, case="camel")
# Columns: ["productName", "salesUsd", "dateOfPurchase"]

df_clean = clean_column_names(df, case="pascal")
# Columns: ["ProductName", "SalesUsd", "DateOfPurchase"]
```

**Cleaning operations:**

- Strips leading/trailing whitespace
- Replaces spaces with underscores
- Removes special characters
- Converts to specified case
- Handles duplicate names

## Outlier Detection

### `detect_outliers(df, columns, method="iqr", threshold=1.5)`

Detects outliers in numeric columns.

```python
from ds_utils.cleaning import detect_outliers

# Using IQR method (default)
outlier_mask = detect_outliers(df, columns=["value"], method="iqr", threshold=1.5)

# Using z-score method
outlier_mask = detect_outliers(df, columns=["value"], method="zscore", threshold=3)

# View outlier rows
outliers = df[outlier_mask]
```

**Methods:**

| Method | Description | Threshold |
|--------|-------------|-----------|
| `iqr` | Interquartile Range | Multiplier for IQR (default: 1.5) |
| `zscore` | Z-score | Number of standard deviations (default: 3) |
| `mad` | Median Absolute Deviation | Multiplier for MAD |

### `remove_outliers(df, columns, method="iqr", threshold=1.5)`

Removes outliers from a DataFrame.

```python
from ds_utils.cleaning import remove_outliers

df_clean = remove_outliers(df, columns=["value"], method="iqr")

# Multiple columns
df_clean = remove_outliers(
    df, 
    columns=["value_a", "value_b"], 
    method="zscore", 
    threshold=3
)
```

## Missing Value Handling

### `missing_value_report(df)`

Generates a report of missing values.

```python
from ds_utils.cleaning import missing_value_report

report = missing_value_report(df)
print(report)
```

**Report contains:**

| Column | Description |
|--------|-------------|
| `column` | Column name |
| `dtype` | Data type |
| `missing_count` | Number of missing values |
| `missing_pct` | Percentage missing |
| `non_missing_count` | Number of non-missing values |

### `fill_missing_values(df, strategies, fill_values=None)`

Fills missing values using specified strategies.

```python
from ds_utils.cleaning import fill_missing_values

df_filled = fill_missing_values(
    df,
    strategies={
        "name": "constant",
        "age": "median",
        "salary": "mean",
        "department": "mode",
    },
    fill_values={
        "name": "Unknown",
    }
)
```

**Strategies:**

| Strategy | Description |
|----------|-------------|
| `mean` | Fill with column mean |
| `median` | Fill with column median |
| `mode` | Fill with most frequent value |
| `constant` | Fill with specified value |
| `ffill` | Forward fill |
| `bfill` | Backward fill |
| `interpolate` | Linear interpolation |

## Data Type Conversion

### `convert_dtypes(df, type_map)`

Converts column data types.

```python
from ds_utils.cleaning import convert_dtypes

df_converted = convert_dtypes(
    df,
    type_map={
        "id": "int",
        "amount": "float",
        "is_active": "bool",
        "date": "datetime",
        "category": "category",
    }
)
```

**Supported types:**

- `int`, `int32`, `int64`
- `float`, `float32`, `float64`
- `bool`
- `str`, `string`
- `datetime`
- `category`

## Text Normalization

### `normalize_text(series, case="lower")`

Normalizes text in a series.

```python
from ds_utils.cleaning import normalize_text

df["name"] = normalize_text(df["name"], case="title")
df["code"] = normalize_text(df["code"], case="upper")
df["email"] = normalize_text(df["email"], case="lower")
```

**Operations:**

- Strips leading/trailing whitespace
- Collapses multiple spaces
- Converts case

**Case options:**

- `lower` - lowercase
- `upper` - UPPERCASE
- `title` - Title Case
- `sentence` - Sentence case

## Data Validation

### `validate_dataframe(df, schema)`

Validates a DataFrame against a Pandera schema.

```python
import pandera as pa
from ds_utils.cleaning import validate_dataframe

schema = pa.DataFrameSchema({
    "id": pa.Column(int, pa.Check.gt(0)),
    "name": pa.Column(str, pa.Check.str_length(min_value=1)),
    "age": pa.Column(int, pa.Check.in_range(0, 120)),
})

is_valid, errors = validate_dataframe(df, schema)

if not is_valid:
    print(f"Validation errors: {errors}")
```

### Schema Creation Helpers

#### `create_schema(columns)`

Creates a Pandera schema from column definitions.

```python
from ds_utils.cleaning import create_schema

schema = create_schema({
    "id": {"dtype": int, "nullable": False, "checks": [pa.Check.gt(0)]},
    "name": {"dtype": str, "nullable": False},
    "age": {"dtype": int, "nullable": True},
})
```

#### `create_numeric_schema(min_value=None, max_value=None, nullable=True)`

Creates a schema for numeric columns.

```python
from ds_utils.cleaning import create_numeric_schema

age_schema = create_numeric_schema(min_value=0, max_value=120, nullable=False)
price_schema = create_numeric_schema(min_value=0, nullable=False)
```

#### `create_string_schema(min_length=None, max_length=None, pattern=None, nullable=True)`

Creates a schema for string columns.

```python
from ds_utils.cleaning import create_string_schema

name_schema = create_string_schema(min_length=1, max_length=100)
code_schema = create_string_schema(pattern=r"^[A-Z]{3}[0-9]{4}$")
```

#### `create_datetime_schema(min_date=None, max_date=None, nullable=True)`

Creates a schema for datetime columns.

```python
from ds_utils.cleaning import create_datetime_schema
from datetime import datetime

date_schema = create_datetime_schema(
    min_date=datetime(2020, 1, 1),
    max_date=datetime(2025, 12, 31),
)
```

#### `create_email_schema(nullable=True)`

Creates a schema for email validation.

```python
from ds_utils.cleaning import create_email_schema

email_schema = create_email_schema(nullable=False)
```

#### `create_german_phone_schema(nullable=True)`

Creates a schema for German phone numbers.

```python
from ds_utils.cleaning import create_german_phone_schema

phone_schema = create_german_phone_schema()
# Validates: +49123456789, 0049123456789, 0123456789
```

#### `create_german_postal_code_schema(nullable=True)`

Creates a schema for German postal codes.

```python
from ds_utils.cleaning import create_german_postal_code_schema

postal_schema = create_german_postal_code_schema()
# Validates: 5-digit codes like 12345, 01234
```

### `generate_validation_report(df, schema)`

Generates a detailed validation report.

```python
from ds_utils.cleaning import generate_validation_report

report = generate_validation_report(df, schema)
print(report)
```

### `data_quality_report(df)`

Generates a comprehensive data quality report.

```python
from ds_utils.cleaning import data_quality_report

report = data_quality_report(df)
print(report)
```

**Report contains:**

- Column statistics (dtype, unique values, missing)
- Numeric column statistics (min, max, mean, std)
- String column statistics (min/max length, patterns)
- Data quality score

## pyjanitor Integration

### `register_janitor_methods()`

Registers custom methods as pyjanitor extensions.

```python
from ds_utils.cleaning import register_janitor_methods

# Register methods
register_janitor_methods()

# Now you can use them in method chains
df_clean = (
    df
    .clean_names()
    .clean_german_numbers("price")
    .parse_german_dates("date")
    .remove_outliers(["value"])
    .fill_missing(strategies={"name": "constant"}, fill_values={"name": "Unknown"})
)
```

## Examples

### Complete Data Cleaning Pipeline

```python
import pandas as pd
from ds_utils.cleaning import (
    clean_column_names,
    clean_german_numbers,
    parse_german_dates,
    remove_outliers,
    fill_missing_values,
    normalize_text,
    validate_dataframe,
    create_schema,
)
import pandera as pa

# Load raw data
df_raw = pd.read_csv("data.csv")

# 1. Clean column names
df = clean_column_names(df_raw)

# 2. Clean German numbers
df["price"] = clean_german_numbers(df["price"])
df["amount"] = clean_german_numbers(df["amount"])

# 3. Parse German dates
df["order_date"] = parse_german_dates(df["order_date"])

# 4. Normalize text
df["customer_name"] = normalize_text(df["customer_name"], case="title")
df["email"] = normalize_text(df["email"], case="lower")

# 5. Handle outliers
df = remove_outliers(df, columns=["price", "amount"], method="iqr")

# 6. Fill missing values
df = fill_missing_values(
    df,
    strategies={
        "customer_name": "constant",
        "price": "median",
    },
    fill_values={"customer_name": "Unknown"},
)

# 7. Validate
schema = pa.DataFrameSchema({
    "price": pa.Column(float, pa.Check.gt(0)),
    "customer_name": pa.Column(str, pa.Check.str_length(min_value=1)),
    "email": pa.Column(str, pa.Check.str_matches(r"^[^@]+@[^@]+\.[^@]+$")),
})

is_valid, errors = validate_dataframe(df, schema)
if is_valid:
    print("Data is valid!")
else:
    print(f"Validation errors: {errors}")
```

### German Invoice Data Processing

```python
import pandas as pd
from ds_utils.cleaning import (
    clean_german_numbers,
    parse_german_dates,
    clean_column_names,
)

# Sample German invoice data
df = pd.DataFrame({
    "Rechnungsnummer": ["R-001", "R-002", "R-003"],
    "Datum": ["15.03.2024", "16.03.2024", "17.03.2024"],
    "Betrag (EUR)": ["1.234,56", "2.345,67", "3.456,78"],
    "MwSt (EUR)": ["234,57", "445,68", "656,79"],
    "Gesamt (EUR)": ["1.469,13", "2.791,35", "4.113,57"],
})

# Clean column names
df = clean_column_names(df)

# Parse dates
df["datum"] = parse_german_dates(df["datum"])

# Convert numbers
for col in ["betrag_eur", "mwst_eur", "gesamt_eur"]:
    df[col] = clean_german_numbers(df[col])

print(df)
print(df.dtypes)
```
