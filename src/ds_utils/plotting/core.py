"""Core plotting utilities for ds_utils.

This module provides base functionality for all plotting operations, including
locale setup, corporate branding, and save utilities.
"""

import locale
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

from ds_utils.config import (
    get_figure_size,
    get_font_config,
    get_grid_config,
    get_locale_config,
    get_logo_config,
    get_save_config,
)
from ds_utils.config.logging_config import get_logger

logger = get_logger("plotting.core")


def setup_locale(locale_name: str | None = None) -> bool:
    """Configure matplotlib and system locale for proper number and date formatting.

    Args:
        locale_name: Locale string (e.g., 'de_DE.UTF-8'). If None, uses the
            configured default locale.

    Returns:
        True if locale was set successfully, False otherwise.

    Note:
        On Windows, locale names may differ (e.g., 'German_Germany.1252').
        The function attempts common variants if the primary locale fails.
    """
    if locale_name is None:
        locale_config = get_locale_config()
        locale_name = locale_config.get("name", "de_DE.UTF-8")

    # Try different locale name variants
    locale_variants = [
        locale_name,
        locale_name.replace(".UTF-8", ""),
        locale_name.split("_")[0],
        "German_Germany.1252",  # Windows variant
        "",  # System default
    ]

    for loc in locale_variants:
        try:
            locale.setlocale(locale.LC_ALL, loc)
            logger.debug(f"Locale set to: {loc}")
            return True
        except locale.Error:
            continue

    warnings.warn(
        f"Could not set locale to {locale_name}. Using system default.",
        UserWarning,
        stacklevel=2,
    )
    return False


def add_corporate_logo(
    fig: Figure,
    logo_path: str | Path | None = None,
    position: str | None = None,
    size: float | None = None,
    alpha: float | None = None,
    padding: int | None = None,
) -> Figure:
    """Add corporate logo to a matplotlib figure.

    Args:
        fig: Matplotlib figure to add logo to.
        logo_path: Path to logo image file. If None, uses configured default.
        position: Logo position. Options: 'top-left', 'top-right', 'bottom-left',
            'bottom-right'. If None, uses configured default.
        size: Relative size of logo (0.1 = 10% of figure width). If None, uses default.
        alpha: Logo transparency (0-1). If None, uses default.
        padding: Padding from edges in pixels. If None, uses default.

    Returns:
        The figure with logo added.

    Raises:
        FileNotFoundError: If the logo file doesn't exist.
    """
    logo_config = get_logo_config()

    # Apply defaults from config if not specified
    if logo_path is None:
        logo_path = logo_config.get("path", "assets/corporate_logo.png")
    if position is None:
        position = logo_config.get("position", "bottom-right")
    if size is None:
        size = logo_config.get("size", 0.1)
    if alpha is None:
        alpha = logo_config.get("alpha", 0.7)
    if padding is None:
        padding = logo_config.get("padding", 10)

    logo_path = Path(logo_path)

    if not logo_path.exists():
        logger.warning(f"Logo file not found: {logo_path}")
        return fig

    try:
        logo_img = plt.imread(logo_path)
    except Exception as e:
        logger.warning(f"Could not load logo: {e}")
        return fig

    # Calculate logo dimensions
    fig_width_px = fig.get_figwidth() * fig.dpi
    logo_width_px = int(fig_width_px * size)

    # Calculate position
    positions = {
        "top-left": (padding / fig_width_px, 1 - padding / (fig.get_figheight() * fig.dpi)),
        "top-right": (1 - padding / fig_width_px, 1 - padding / (fig.get_figheight() * fig.dpi)),
        "bottom-left": (padding / fig_width_px, padding / (fig.get_figheight() * fig.dpi)),
        "bottom-right": (1 - padding / fig_width_px, padding / (fig.get_figheight() * fig.dpi)),
    }

    if position not in positions:
        logger.warning(f"Unknown position: {position}. Using 'bottom-right'.")
        position = "bottom-right"

    x, y = positions[position]

    # Add logo using OffsetImage
    zoom = logo_width_px / logo_img.shape[1]
    imagebox = OffsetImage(logo_img, zoom=zoom, alpha=alpha)

    ab = AnnotationBbox(
        imagebox,
        (x, y),
        xycoords="figure fraction",
        frameon=False,
        box_alignment=(0.5, 0.5),
    )

    # Add to all axes or create a temporary one
    axes = fig.get_axes()
    if axes:
        axes[0].add_artist(ab)
    else:
        ax = fig.add_axes([0, 0, 1, 1], frame_on=False)
        ax.set_axis_off()
        ax.add_artist(ab)

    return fig


def apply_corporate_style(
    apply_globally: bool = True,
    font_family: str | None = None,
    use_grid: bool = True,
) -> dict[str, Any]:
    """Apply corporate styling to matplotlib plots.

    Args:
        apply_globally: If True, applies style to all subsequent plots.
            If False, returns style dict for manual application.
        font_family: Font family to use. If None, uses configured default.
        use_grid: Whether to show grid lines.

    Returns:
        Dictionary of style parameters applied.
    """
    font_config = get_font_config()
    grid_config = get_grid_config()

    if font_family is None:
        font_family = font_config.get("family", "DejaVu Sans")

    style_params = {
        # Font settings
        "font.family": font_family,
        "font.size": font_config.get("label_size", 12),
        "axes.titlesize": font_config.get("title_size", 14),
        "axes.labelsize": font_config.get("label_size", 12),
        "xtick.labelsize": font_config.get("tick_size", 10),
        "ytick.labelsize": font_config.get("tick_size", 10),
        "legend.fontsize": font_config.get("legend_size", 10),
        # Grid settings
        "axes.grid": use_grid and grid_config.get("show", True),
        "grid.linestyle": grid_config.get("style", "--"),
        "grid.alpha": grid_config.get("alpha", 0.3),
        "grid.color": grid_config.get("color", "#cccccc"),
        # Spine settings
        "axes.spines.top": False,
        "axes.spines.right": False,
        # Figure settings
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }

    if apply_globally:
        plt.rcParams.update(style_params)
        logger.debug("Applied corporate style globally")

    return style_params


def save_plot(
    fig: Figure,
    filepath: str | Path,
    dpi: int | None = None,
    file_format: str | None = None,
    transparent: bool | None = None,
    add_logo: bool = True,
    **kwargs: Any,
) -> Path:
    """Save a matplotlib figure with consistent settings.

    Args:
        fig: Figure to save.
        filepath: Output file path.
        dpi: Resolution in dots per inch. If None, uses configured default.
        file_format: Output format (e.g., 'png', 'pdf'). If None, inferred from filepath.
        transparent: Whether background should be transparent. If None, uses default.
        add_logo: Whether to add corporate logo before saving.
        **kwargs: Additional arguments passed to fig.savefig().

    Returns:
        Path to the saved file.
    """
    save_config = get_save_config()

    filepath = Path(filepath)

    # Apply defaults
    if dpi is None:
        dpi = save_config.get("dpi", 150)
    if file_format is None:
        file_format = filepath.suffix.lstrip(".") or save_config.get("format", "png")
    if transparent is None:
        transparent = save_config.get("transparent", False)

    # Add logo if requested
    if add_logo:
        add_corporate_logo(fig)

    # Ensure directory exists
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Ensure correct extension
    if not filepath.suffix:
        filepath = filepath.with_suffix(f".{file_format}")

    # Merge save settings
    save_kwargs = {
        "dpi": dpi,
        "format": file_format,
        "transparent": transparent,
        "bbox_inches": save_config.get("bbox_inches", "tight"),
        "pad_inches": save_config.get("pad_inches", 0.1),
    }
    save_kwargs.update(kwargs)

    fig.savefig(filepath, **save_kwargs)
    logger.info(f"Saved plot to: {filepath}")

    return filepath


def create_figure(
    size: str = "medium",
    nrows: int = 1,
    ncols: int = 1,
    apply_style: bool = True,
    **kwargs: Any,
) -> tuple[Figure, Any]:
    """Create a figure with standard configuration.

    Args:
        size: Size preset name ('small', 'medium', 'large', 'presentation', 'square').
        nrows: Number of subplot rows.
        ncols: Number of subplot columns.
        apply_style: Whether to apply corporate style.
        **kwargs: Additional arguments passed to plt.subplots().

    Returns:
        Tuple of (figure, axes).
    """
    if apply_style:
        apply_corporate_style()

    figsize = get_figure_size(size)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, **kwargs)

    return fig, axes


@contextmanager
def plot_context(
    locale_name: str | None = None,
    apply_style: bool = True,
) -> Generator[None, None, None]:
    """Context manager for temporary plot configuration.

    Args:
        locale_name: Locale to use within the context.
        apply_style: Whether to apply corporate style.

    Yields:
        None

    Example:
        >>> with plot_context(locale_name='en_US.UTF-8'):
        ...     fig, ax = plt.subplots()
        ...     ax.plot(data)
    """
    # Save current state
    original_rcparams = plt.rcParams.copy()

    try:
        if locale_name:
            setup_locale(locale_name)
        if apply_style:
            apply_corporate_style()
        yield
    finally:
        # Restore original state
        plt.rcParams.update(original_rcparams)


def format_number(
    value: float,
    decimal_places: int = 2,
    use_locale: bool = True,
) -> str:
    """Format a number according to locale settings.

    Args:
        value: Number to format.
        decimal_places: Number of decimal places.
        use_locale: Whether to use locale-specific formatting.

    Returns:
        Formatted number string.
    """
    if use_locale:
        return locale.format_string(f"%.{decimal_places}f", value, grouping=True)
    return f"{value:.{decimal_places}f}"
