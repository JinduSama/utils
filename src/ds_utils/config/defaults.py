"""Default configuration management for ds_utils.

This module provides functions to load, access, and modify configuration settings
for the library, including color schemes, fonts, figure sizes, and locale settings.
"""

from pathlib import Path
from typing import Any

import yaml

# Path to the configuration file
CONFIG_DIR = Path(__file__).parent
CONFIG_FILE = CONFIG_DIR / "plot_styles.yaml"

# Global configuration cache
_config: dict[str, Any] | None = None


def load_config(config_path: Path | str | None = None) -> dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Optional path to a custom configuration file.
            If None, loads the default configuration.

    Returns:
        Dictionary containing all configuration settings.

    Raises:
        FileNotFoundError: If the configuration file doesn't exist.
        yaml.YAMLError: If the configuration file is invalid YAML.
    """
    global _config

    if config_path is None:
        config_path = CONFIG_FILE
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        _config = yaml.safe_load(f)

    return _config


def get_config() -> dict[str, Any]:
    """Get the current configuration.

    Returns:
        Dictionary containing all configuration settings.
        Loads from default file if not already loaded.
    """
    global _config
    if _config is None:
        load_config()
    return _config  # type: ignore


def get_colors(palette: str = "categorical") -> list[str]:
    """Get a color palette.

    Args:
        palette: Name of the palette. Options: 'categorical', 'sequential',
            'primary', 'secondary', 'semantic'.

    Returns:
        List of color hex codes, or dict for named palettes.

    Raises:
        KeyError: If the palette name is not found.
    """
    config = get_config()
    colors = config.get("colors", {})

    if palette in colors:
        return colors[palette]
    raise KeyError(f"Unknown color palette: {palette}")


def get_figure_size(size: str = "medium") -> tuple[float, float]:
    """Get predefined figure dimensions.

    Args:
        size: Size preset name. Options: 'small', 'medium', 'large',
            'presentation', 'square'.

    Returns:
        Tuple of (width, height) in inches.

    Raises:
        KeyError: If the size preset is not found.
    """
    config = get_config()
    sizes = config.get("figure_sizes", {})

    if size in sizes:
        return (sizes[size]["width"], sizes[size]["height"])
    raise KeyError(f"Unknown figure size: {size}")


def get_font_config() -> dict[str, Any]:
    """Get font configuration settings.

    Returns:
        Dictionary with font settings (family, sizes).
    """
    config = get_config()
    return config.get("fonts", {})


def get_grid_config() -> dict[str, Any]:
    """Get grid configuration settings.

    Returns:
        Dictionary with grid settings (show, style, alpha, color).
    """
    config = get_config()
    return config.get("grid", {})


def get_logo_config() -> dict[str, Any]:
    """Get logo configuration settings.

    Returns:
        Dictionary with logo settings (path, position, size, alpha, padding).
    """
    config = get_config()
    return config.get("logo", {})


def get_locale_config() -> dict[str, Any]:
    """Get locale configuration settings.

    Returns:
        Dictionary with locale settings (name, separators, formats).
    """
    config = get_config()
    return config.get("locale", {})


def get_save_config() -> dict[str, Any]:
    """Get save configuration settings.

    Returns:
        Dictionary with save settings (dpi, format, etc.).
    """
    config = get_config()
    return config.get("save", {})


def set_config_value(key: str, value: Any) -> None:
    """Set a configuration value at runtime.

    Args:
        key: Dot-separated key path (e.g., 'colors.primary.main').
        value: Value to set.

    Note:
        Changes are not persisted to file. They only affect the current session.
    """
    config = get_config()
    keys = key.split(".")

    # Navigate to the parent of the target key
    current = config
    for k in keys[:-1]:
        if k not in current:
            current[k] = {}
        current = current[k]

    # Set the value
    current[keys[-1]] = value


def reset_config() -> None:
    """Reset configuration to default values from file."""
    global _config
    _config = None
    load_config()


# Locale utilities
def get_locale_name() -> str:
    """Get the configured locale name.

    Returns:
        Locale string (e.g., 'de_DE.UTF-8').
    """
    locale_config = get_locale_config()
    return locale_config.get("name", "de_DE.UTF-8")


def get_decimal_separator() -> str:
    """Get the decimal separator for the configured locale.

    Returns:
        Decimal separator character.
    """
    locale_config = get_locale_config()
    return locale_config.get("decimal_separator", ",")


def get_thousands_separator() -> str:
    """Get the thousands separator for the configured locale.

    Returns:
        Thousands separator character.
    """
    locale_config = get_locale_config()
    return locale_config.get("thousands_separator", ".")


def get_date_format() -> str:
    """Get the date format string.

    Returns:
        Date format string for strftime.
    """
    locale_config = get_locale_config()
    return locale_config.get("date_format", "%d.%m.%Y")
