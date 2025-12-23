"""Configuration module for ds_utils.

This module provides access to configuration settings, color schemes,
and logging utilities.
"""

from ds_utils.config.defaults import (
    get_colors,
    get_config,
    get_date_format,
    get_decimal_separator,
    get_figure_size,
    get_font_config,
    get_grid_config,
    get_locale_config,
    get_locale_name,
    get_logo_config,
    get_save_config,
    get_thousands_separator,
    load_config,
    reset_config,
    set_config_value,
)
from ds_utils.config.logging_config import (
    disable_logging,
    enable_logging,
    get_logger,
    set_log_level,
    setup_logging,
)

__all__ = [
    # Configuration
    "load_config",
    "get_config",
    "get_colors",
    "get_figure_size",
    "get_font_config",
    "get_grid_config",
    "get_logo_config",
    "get_locale_config",
    "get_save_config",
    "set_config_value",
    "reset_config",
    # Locale utilities
    "get_locale_name",
    "get_decimal_separator",
    "get_thousands_separator",
    "get_date_format",
    # Logging
    "setup_logging",
    "get_logger",
    "set_log_level",
    "disable_logging",
    "enable_logging",
]
