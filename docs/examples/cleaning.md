# Data Cleaning Examples

This page provides examples of using the data cleaning module.

## Setup

```python
import pandas as pd
import numpy as np

from ds_utils.cleaning import (
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
    validate_dataframe,
    create_numeric_schema,
    create_email_schema,
    create_german_postal_code_schema,
    data_quality_report,
)

# Register custom janitor methods
register_janitor_methods()
```

## German Number Cleaning

```python
# Create sample data with German formatting
df = pd.DataFrame({
    "price": ["1.234,56", "2.345,67", "3.456,78"],
    "quantity": ["1.000", "2.500", "10.000"],
    "revenue": ["1.234.560,00", "5.862.500,00", "34.567.800,00"],
})

# Clean German numbers
df["price_clean"] = clean_german_numbers(df["price"])
df["quantity_clean"] = clean_german_numbers(df["quantity"])
df["revenue_clean"] = clean_german_numbers(df["revenue"])

print(df)
```

## German Date Parsing

```python
df = pd.DataFrame({
    "date": ["15.03.2024", "01.12.2023", "22.06.2024"],
    "datetime": ["15.03.2024 14:30:00", "01.12.2023 09:15:00", "22.06.2024 18:45:00"],
})

# Parse German dates
df["date_clean"] = parse_german_dates(df["date"])
df["datetime_clean"] = parse_german_dates(df["datetime"], format="%d.%m.%Y %H:%M:%S")

print(df.dtypes)
```

## Column Name Cleaning

```python
df = pd.DataFrame({
    "Product Name  ": [1, 2, 3],
    " Sales (USD)": [100, 200, 300],
    "Date Of Purchase": ["2024-01-01", "2024-01-02", "2024-01-03"],
    "Customer ID#": ["C001", "C002", "C003"],
})

# Clean column names
df_clean = clean_column_names(df)
print(df_clean.columns.tolist())
# ['product_name', 'sales_usd', 'date_of_purchase', 'customer_id']
```

## Outlier Detection and Removal

```python
# Create data with outliers
np.random.seed(42)
df = pd.DataFrame({
    "value": np.concatenate([np.random.normal(50, 10, 95), [150, 200, -50, 180, 250]]),
})

# Detect outliers
outlier_mask = detect_outliers(df, columns=["value"], method="iqr")
print(f"Outliers detected: {outlier_mask.sum()}")
print(df[outlier_mask])

# Remove outliers
df_clean = remove_outliers(df, columns=["value"], method="iqr")
print(f"Original rows: {len(df)}, After removal: {len(df_clean)}")
```

### Different Methods

```python
# IQR method (default)
mask_iqr = detect_outliers(df, columns=["value"], method="iqr", threshold=1.5)

# Z-score method
mask_zscore = detect_outliers(df, columns=["value"], method="zscore", threshold=3)

print(f"IQR outliers: {mask_iqr.sum()}")
print(f"Z-score outliers: {mask_zscore.sum()}")
```

## Missing Value Handling

```python
# Create data with missing values
df = pd.DataFrame({
    "name": ["Alice", "Bob", None, "David", "Eve", None],
    "age": [25, np.nan, 35, 40, np.nan, 28],
    "salary": [50000, 60000, np.nan, 75000, 55000, np.nan],
    "department": ["Sales", "IT", "HR", None, "Sales", "IT"],
})

# Generate missing value report
report = missing_value_report(df)
print(report)

# Fill missing values
df_filled = fill_missing_values(
    df,
    strategies={
        "name": "constant",
        "age": "median",
        "salary": "mean",
        "department": "mode",
    },
    fill_values={"name": "Unknown"},
)
print(df_filled)
```

## Text Normalization

```python
df = pd.DataFrame({
    "name": ["  JOHN DOE  ", "jane   smith", "BOB JOHNSON"],
    "email": ["JOHN@EMAIL.COM", "jane@email.com", "BOB@EMAIL.COM"],
})

# Normalize text
df["name_clean"] = normalize_text(df["name"], case="title")
df["email_clean"] = normalize_text(df["email"], case="lower")

print(df)
```

## Data Type Conversion

```python
df = pd.DataFrame({
    "id": ["1", "2", "3"],
    "amount": ["100.50", "200.75", "300.25"],
    "is_active": ["True", "False", "True"],
    "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
    "category": ["A", "B", "A"],
})

# Convert types
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

print(df_converted.dtypes)
```

## Data Validation

```python
import pandera as pa

# Create schema
schema = pa.DataFrameSchema({
    "name": pa.Column(str, pa.Check.str_length(min_value=1, max_value=100)),
    "age": create_numeric_schema(min_value=0, max_value=120, nullable=False),
    "email": create_email_schema(nullable=False),
    "postal_code": create_german_postal_code_schema(nullable=True),
})

# Valid data
df_valid = pd.DataFrame({
    "name": ["Alice", "Bob"],
    "age": [25, 30],
    "email": ["alice@example.com", "bob@example.com"],
    "postal_code": ["12345", "67890"],
})

is_valid, errors = validate_dataframe(df_valid, schema)
print(f"Valid: {is_valid}")

# Invalid data
df_invalid = pd.DataFrame({
    "name": ["Alice", ""],
    "age": [25, -5],
    "email": ["alice@example.com", "not-an-email"],
    "postal_code": ["12345", "ABCDE"],
})

is_valid, errors = validate_dataframe(df_invalid, schema)
print(f"Valid: {is_valid}")
print(f"Errors: {errors}")
```

## Data Quality Report

```python
np.random.seed(42)
df = pd.DataFrame({
    "id": range(1, 101),
    "name": ["Person " + str(i) if i % 10 != 0 else None for i in range(1, 101)],
    "age": [np.random.randint(18, 80) if i % 7 != 0 else np.nan for i in range(1, 101)],
    "salary": [np.random.uniform(30000, 100000) if i % 5 != 0 else np.nan for i in range(1, 101)],
})

report = data_quality_report(df)
print(report)
```

## Complete Pipeline Example

```python
# Complete data cleaning pipeline
df_raw = pd.DataFrame({
    "Produkt Name": ["Widget A", "Widget B", "Widget C"],
    "Preis (EUR)": ["1.234,56", "2.345,67", "3.456,78"],
    "Datum": ["15.03.2024", "16.03.2024", "17.03.2024"],
    "Menge": ["100", "200", None],
})

# 1. Clean column names
df = clean_column_names(df_raw)

# 2. Clean German numbers
df["preis_eur"] = clean_german_numbers(df["preis_eur"])

# 3. Parse German dates
df["datum"] = parse_german_dates(df["datum"])

# 4. Fill missing values
df = fill_missing_values(
    df,
    strategies={"menge": "median"},
)

# 5. Convert types
df = convert_dtypes(df, type_map={"menge": "float"})

print(df)
print(df.dtypes)
```
