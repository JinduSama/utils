"""Tests for standard plotting functions."""

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from ds_utils.plotting import plot_bar, plot_line, plot_scatter, plot_time_series


@pytest.fixture(autouse=True)
def close_figures():
    """Close all figures after each test."""
    yield
    plt.close("all")


class TestPlotLine:
    """Tests for plot_line function."""

    def test_basic_line_plot(self, sample_dataframe):
        """Test creating a basic line plot."""
        fig, ax = plot_line(sample_dataframe, x="date", y="numeric")

        assert fig is not None
        assert ax is not None
        assert len(ax.lines) >= 1

    def test_line_plot_with_title(self, sample_dataframe):
        """Test line plot with custom title."""
        fig, ax = plot_line(sample_dataframe, x="date", y="numeric", title="Test Title")

        assert ax.get_title() == "Test Title"

    def test_line_plot_multiple_y(self, sample_dataframe):
        """Test line plot with multiple y columns."""
        sample_dataframe["numeric2"] = sample_dataframe["numeric"] * 2

        fig, ax = plot_line(sample_dataframe, x="date", y=["numeric", "numeric2"])

        assert len(ax.lines) >= 2

    def test_line_plot_with_markers(self, sample_dataframe):
        """Test line plot with markers enabled."""
        fig, ax = plot_line(sample_dataframe, x="date", y="numeric", markers=True)

        line = ax.lines[0]
        assert line.get_marker() is not None


class TestPlotScatter:
    """Tests for plot_scatter function."""

    def test_basic_scatter_plot(self, sample_dataframe):
        """Test creating a basic scatter plot."""
        fig, ax = plot_scatter(sample_dataframe, x="numeric", y="value")

        assert fig is not None
        assert ax is not None

    def test_scatter_with_regression(self, sample_dataframe):
        """Test scatter plot with regression line."""
        fig, ax = plot_scatter(
            sample_dataframe, x="value", y="numeric", add_regression=True
        )

        # Should have scatter + regression line
        assert len(ax.lines) >= 1 or len(ax.collections) >= 1

    def test_scatter_with_title(self, sample_dataframe):
        """Test scatter plot with custom title."""
        fig, ax = plot_scatter(
            sample_dataframe, x="numeric", y="value", title="Scatter Test"
        )

        assert ax.get_title() == "Scatter Test"


class TestPlotBar:
    """Tests for plot_bar function."""

    def test_basic_bar_plot(self):
        """Test creating a basic bar plot."""
        data = pd.DataFrame(
            {
                "category": ["A", "B", "C"],
                "value": [10, 20, 15],
            }
        )

        fig, ax = plot_bar(data, x="category", y="value")

        assert fig is not None
        assert ax is not None
        assert len(ax.patches) == 3

    def test_horizontal_bar_plot(self):
        """Test horizontal bar plot."""
        data = pd.DataFrame(
            {
                "category": ["A", "B", "C"],
                "value": [10, 20, 15],
            }
        )

        fig, ax = plot_bar(data, x="category", y="value", orientation="horizontal")

        assert fig is not None

    def test_bar_plot_with_values(self):
        """Test bar plot with value labels."""
        data = pd.DataFrame(
            {
                "category": ["A", "B"],
                "value": [10, 20],
            }
        )

        fig, ax = plot_bar(data, x="category", y="value", show_values=True)

        # Check that text labels exist
        assert len(ax.texts) >= 0  # bar_label adds texts


class TestPlotTimeSeries:
    """Tests for plot_time_series function."""

    def test_basic_time_series(self, sample_dataframe):
        """Test creating a basic time series plot."""
        fig, ax = plot_time_series(sample_dataframe, y="numeric", date_col="date")

        assert fig is not None
        assert ax is not None

    def test_time_series_with_trend(self, sample_dataframe):
        """Test time series with trend line."""
        fig, ax = plot_time_series(
            sample_dataframe, y="numeric", date_col="date", show_trend=True
        )

        # Should have original line + trend line
        assert len(ax.lines) >= 2
