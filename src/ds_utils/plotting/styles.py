"""Style management for ds_utils plotting.

This module provides functions for managing color palettes, creating figures
with consistent styling, and temporary style contexts.
"""

from contextlib import contextmanager
from typing import Any, Generator, Literal

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure

from ds_utils.config import get_colors, get_config, get_figure_size, get_font_config

from ds_utils.config.logging_config import get_logger

logger = get_logger("plotting.styles")


def get_color_palette(
    palette: str = "categorical",
    n_colors: int | None = None,
) -> list[str]:
    """Get a color palette from configuration.

    Args:
        palette: Name of the palette. Options:
            - 'categorical': For discrete categories
            - 'sequential': For ordered data
            - 'primary': Primary corporate colors
            - 'secondary': Secondary corporate colors
        n_colors: Number of colors to return. If None, returns all colors in palette.

    Returns:
        List of color hex codes.

    Example:
        >>> colors = get_color_palette('categorical', n_colors=3)
        >>> ['#1f77b4', '#ff7f0e', '#2ca02c']
    """
    colors = get_colors(palette)

    if isinstance(colors, dict):
        # For palettes like 'primary', 'secondary' that are dicts
        color_list = list(colors.values())
    else:
        color_list = list(colors)

    if n_colors is not None:
        # Cycle colors if we need more than available
        if n_colors > len(color_list):
            color_list = (color_list * ((n_colors // len(color_list)) + 1))[:n_colors]
        else:
            color_list = color_list[:n_colors]

    return color_list


def set_seaborn_palette(palette: str = "categorical") -> None:
    """Set seaborn's default color palette.

    Args:
        palette: Name of the palette to use.
    """
    colors = get_color_palette(palette)
    sns.set_palette(colors)
    logger.debug(f"Set seaborn palette to {palette}")


def create_styled_figure(
    size: str = "medium",
    nrows: int = 1,
    ncols: int = 1,
    style: str = "corporate",
    **kwargs: Any,
) -> tuple[Figure, Any]:
    """Create a figure with predefined styling.

    Args:
        size: Size preset ('small', 'medium', 'large', 'presentation', 'square').
        nrows: Number of subplot rows.
        ncols: Number of subplot columns.
        style: Style to apply ('corporate', 'minimal', 'publication').
        **kwargs: Additional arguments passed to plt.subplots().

    Returns:
        Tuple of (figure, axes).
    """
    # Apply style
    style_params = _get_style_params(style)
    plt.rcParams.update(style_params)

    # Get figure size
    figsize = get_figure_size(size)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, **kwargs)

    return fig, axes


def _get_style_params(style: str) -> dict[str, Any]:
    """Get matplotlib parameters for a named style.

    Args:
        style: Style name ('corporate', 'minimal', 'publication').

    Returns:
        Dictionary of matplotlib rcParams.
    """
    font_config = get_font_config()

    base_params = {
        "font.family": font_config.get("family", "DejaVu Sans"),
        "font.size": font_config.get("label_size", 12),
        "axes.titlesize": font_config.get("title_size", 14),
        "axes.labelsize": font_config.get("label_size", 12),
        "xtick.labelsize": font_config.get("tick_size", 10),
        "ytick.labelsize": font_config.get("tick_size", 10),
        "legend.fontsize": font_config.get("legend_size", 10),
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }

    if style == "corporate":
        return {
            **base_params,
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    elif style == "minimal":
        return {
            **base_params,
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.spines.bottom": True,
        }
    elif style == "publication":
        return {
            **base_params,
            "axes.grid": True,
            "grid.linestyle": ":",
            "grid.alpha": 0.5,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.linewidth": 1.5,
            "font.size": 14,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
        }
    else:
        logger.warning(f"Unknown style '{style}'. Using corporate style.")
        return _get_style_params("corporate")


@contextmanager
def style_context(
    style: str = "corporate",
    palette: str = "categorical",
) -> Generator[None, None, None]:
    """Context manager for temporary style changes.

    Args:
        style: Style name to apply within context.
        palette: Color palette to use.

    Yields:
        None

    Example:
        >>> with style_context('publication'):
        ...     fig, ax = plt.subplots()
        ...     ax.plot([1, 2, 3], [1, 4, 9])
    """
    # Save current state
    original_rcparams = plt.rcParams.copy()

    try:
        style_params = _get_style_params(style)
        plt.rcParams.update(style_params)
        set_seaborn_palette(palette)
        yield
    finally:
        # Restore original state
        plt.rcParams.update(original_rcparams)


@contextmanager
def temporary_palette(colors: list[str]) -> Generator[None, None, None]:
    """Context manager for temporary color palette.

    Args:
        colors: List of color hex codes to use.

    Yields:
        None
    """
    original_palette = sns.color_palette()

    try:
        sns.set_palette(colors)
        yield
    finally:
        sns.set_palette(original_palette)


def reset_style() -> None:
    """Reset matplotlib and seaborn to default styles."""
    plt.rcdefaults()
    sns.reset_defaults()
    logger.debug("Reset to default styles")


def apply_style(
    style: Literal["corporate", "minimal", "publication"] = "corporate",
) -> None:
    """Apply a named style globally.

    Args:
        style: Style name to apply.
    """
    style_params = _get_style_params(style)
    plt.rcParams.update(style_params)
    set_seaborn_palette("categorical")
    logger.debug(f"Applied {style} style globally")


def get_contrasting_color(background_color: str) -> str:
    """Get a contrasting text color (black or white) for a given background.

    Args:
        background_color: Hex color code for background.

    Returns:
        '#000000' for dark text or '#ffffff' for light text.
    """
    # Remove # if present
    hex_color = background_color.lstrip("#")

    # Convert to RGB
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)

    # Calculate luminance
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255

    return "#000000" if luminance > 0.5 else "#ffffff"


def create_colormap(
    colors: list[str] | None = None,
    name: str = "custom",
    n_colors: int = 256,
) -> Any:
    """Create a matplotlib colormap from a list of colors.

    Args:
        colors: List of hex color codes. If None, uses sequential palette.
        name: Name for the colormap.
        n_colors: Number of colors in the resulting colormap.

    Returns:
        matplotlib LinearSegmentedColormap.
    """
    from matplotlib.colors import LinearSegmentedColormap

    if colors is None:
        colors = get_colors("sequential")

    return LinearSegmentedColormap.from_list(name, colors, N=n_colors)
