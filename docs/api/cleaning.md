# Cleaning Module API Reference

::: ds_utils.cleaning.extensions
    options:
      show_source: true
      members:
        - clean_german_numbers
        - parse_german_dates
        - clean_column_names
        - detect_outliers
        - remove_outliers
        - missing_value_report
        - fill_missing_values
        - convert_dtypes
        - normalize_text
        - register_janitor_methods

---

::: ds_utils.cleaning.validation
    options:
      show_source: true
      members:
        - validate_dataframe
        - create_schema
        - create_numeric_schema
        - create_string_schema
        - create_datetime_schema
        - create_email_schema
        - create_german_phone_schema
        - create_german_postal_code_schema
        - generate_validation_report
        - data_quality_report
