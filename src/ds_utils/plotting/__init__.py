"""Plotting module for ds_utils.

This module provides enhanced plotting utilities with corporate styling,
localization support, and consistent visual formatting.
"""

from ds_utils.plotting.core import (
    add_corporate_logo,
    apply_corporate_style,
    create_figure,
    format_number,
    plot_context,
    save_plot,
    setup_locale,
)
from ds_utils.plotting.distributions import (
    plot_boxplot,
    plot_density,
    plot_histogram,
    plot_qq,
    plot_violin,
)
from ds_utils.plotting.standard import (
    plot_bar,
    plot_line,
    plot_scatter,
    plot_time_series,
)
from ds_utils.plotting.styles import (
    apply_style,
    create_colormap,
    create_styled_figure,
    get_color_palette,
    get_contrasting_color,
    reset_style,
    set_seaborn_palette,
    style_context,
    temporary_palette,
)

__all__ = [
    # Core utilities
    "setup_locale",
    "add_corporate_logo",
    "apply_corporate_style",
    "save_plot",
    "create_figure",
    "plot_context",
    "format_number",
    # Standard plots
    "plot_line",
    "plot_scatter",
    "plot_bar",
    "plot_time_series",
    # Distribution plots
    "plot_histogram",
    "plot_density",
    "plot_boxplot",
    "plot_violin",
    "plot_qq",
    # Style management
    "get_color_palette",
    "set_seaborn_palette",
    "create_styled_figure",
    "style_context",
    "temporary_palette",
    "reset_style",
    "apply_style",
    "get_contrasting_color",
    "create_colormap",
]
