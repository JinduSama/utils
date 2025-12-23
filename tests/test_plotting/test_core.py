"""Tests for core plotting utilities."""

import matplotlib.pyplot as plt
import pytest
from pathlib import Path

from ds_utils.plotting.core import (
    apply_corporate_style,
    create_figure,
    format_number,
    plot_context,
    setup_locale,
)


@pytest.fixture(autouse=True)
def close_figures():
    """Close all figures after each test."""
    yield
    plt.close("all")


class TestSetupLocale:
    """Tests for setup_locale function."""

    def test_setup_locale_returns_bool(self):
        """Test that setup_locale returns a boolean."""
        result = setup_locale()
        assert isinstance(result, bool)

    def test_setup_locale_with_custom_locale(self):
        """Test setup_locale with a custom locale string."""
        # This may or may not succeed depending on system
        result = setup_locale("C")
        assert isinstance(result, bool)


class TestApplyCorporateStyle:
    """Tests for apply_corporate_style function."""

    def test_apply_style_globally(self):
        """Test applying style globally."""
        params = apply_corporate_style(apply_globally=True)

        assert isinstance(params, dict)
        assert "font.family" in params
        assert "axes.grid" in params

    def test_apply_style_returns_params(self):
        """Test that apply_style returns style parameters."""
        params = apply_corporate_style(apply_globally=False)

        assert "font.family" in params
        assert "axes.titlesize" in params


class TestCreateFigure:
    """Tests for create_figure function."""

    def test_create_figure_default(self):
        """Test creating a figure with default settings."""
        fig, ax = create_figure()

        assert fig is not None
        assert ax is not None

    def test_create_figure_with_size(self):
        """Test creating a figure with specific size preset."""
        fig, ax = create_figure(size="large")

        assert fig is not None
        assert fig.get_figwidth() > 10

    def test_create_figure_with_subplots(self):
        """Test creating a figure with multiple subplots."""
        fig, axes = create_figure(nrows=2, ncols=2)

        assert fig is not None
        assert axes.shape == (2, 2)


class TestPlotContext:
    """Tests for plot_context context manager."""

    def test_plot_context_restores_params(self):
        """Test that plot_context restores original parameters."""
        original_font_size = plt.rcParams["font.size"]

        with plot_context():
            plt.rcParams["font.size"] = 999

        # Should be restored (or at least different from 999)
        assert plt.rcParams["font.size"] != 999


class TestFormatNumber:
    """Tests for format_number function."""

    def test_format_number_default(self):
        """Test default number formatting."""
        result = format_number(1234.567, decimal_places=2, use_locale=False)
        assert result == "1234.57"

    def test_format_number_custom_decimals(self):
        """Test formatting with custom decimal places."""
        result = format_number(1234.5, decimal_places=0, use_locale=False)
        assert result == "1234"
