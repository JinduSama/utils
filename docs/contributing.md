# Contributing

Thank you for your interest in contributing to DS Utils! This document provides guidelines and instructions for contributing.

## Development Setup

1. **Fork and clone the repository**

   ```bash
   git clone https://github.com/YOUR_USERNAME/ds-utils.git
   cd ds-utils
   ```

2. **Set up the development environment**

   ```bash
   # Using uv (recommended)
   uv venv
   uv sync

   # Or using pip
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -e ".[dev]"
   ```

3. **Install pre-commit hooks**

   ```bash
   pre-commit install
   ```

## Development Workflow

### Code Style

We use the following tools to maintain code quality:

- **Black** for code formatting
- **Ruff** for linting
- **mypy** for type checking

Run all checks before committing:

```bash
# Format code
black src tests

# Lint code
ruff check src tests --fix

# Type checking
mypy src
```

Or use pre-commit to run all checks automatically:

```bash
pre-commit run --all-files
```

### Testing

We use pytest for testing:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/ds_utils --cov-report=html

# Run specific test file
pytest tests/test_plotting/test_core.py

# Run tests matching a pattern
pytest -k "test_line"
```

### Documentation

We use MkDocs with the Material theme:

```bash
# Serve documentation locally
mkdocs serve

# Build documentation
mkdocs build
```

## Making Changes

### Branch Naming

Use descriptive branch names:

- `feature/add-new-plot-type`
- `fix/german-number-parsing`
- `docs/update-readme`
- `refactor/simplify-config`

### Commit Messages

Follow conventional commit format:

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

Examples:

```
feat(plotting): add violin plot function

fix(cleaning): handle empty strings in German number parser

docs(readme): update installation instructions
```

### Pull Requests

1. Create a feature branch from `main`
2. Make your changes
3. Run tests and linting
4. Push to your fork
5. Create a pull request

PR template:

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Refactoring

## Checklist
- [ ] Tests pass locally
- [ ] Code is formatted with Black
- [ ] Linting passes with Ruff
- [ ] Type hints added for new code
- [ ] Documentation updated
```

## Adding New Features

### Adding a New Plot Type

1. Add the function to the appropriate module:

   ```python
   # src/ds_utils/plotting/standard.py

   def plot_new_type(
       data: ArrayLike,
       *,
       title: str | None = None,
       xlabel: str | None = None,
       ylabel: str | None = None,
       figsize: tuple[float, float] | None = None,
       ax: Axes | None = None,
       **kwargs: Any,
   ) -> tuple[Figure, Axes]:
       """Create a new plot type.

       Parameters
       ----------
       data : array-like
           Input data for the plot.
       title : str, optional
           Plot title.
       xlabel : str, optional
           X-axis label.
       ylabel : str, optional
           Y-axis label.
       figsize : tuple, optional
           Figure size (width, height).
       ax : Axes, optional
           Existing axes to plot on.
       **kwargs
           Additional keyword arguments passed to the plotting function.

       Returns
       -------
       tuple[Figure, Axes]
           The figure and axes objects.

       Examples
       --------
       >>> fig, ax = plot_new_type([1, 2, 3, 4], title="My Plot")
       """
       fig, ax = create_figure(figsize=figsize, ax=ax)
       # Implementation here
       return fig, ax
   ```

2. Export in `__init__.py`:

   ```python
   from .standard import plot_new_type
   ```

3. Add tests:

   ```python
   # tests/test_plotting/test_standard.py

   def test_plot_new_type():
       data = [1, 2, 3, 4]
       fig, ax = plot_new_type(data, title="Test")
       assert fig is not None
       assert ax is not None
   ```

4. Add documentation

### Adding a New Cleaning Function

1. Add to extensions.py or validation.py
2. Add type hints
3. Add docstring with examples
4. Export in `__init__.py`
5. Add tests
6. Update documentation

## Code Guidelines

### Type Hints

All functions should have type hints:

```python
from typing import Any
import pandas as pd
import numpy as np
from numpy.typing import ArrayLike

def my_function(
    data: pd.DataFrame,
    columns: list[str],
    threshold: float = 0.5,
) -> pd.DataFrame:
    ...
```

### Docstrings

Use NumPy-style docstrings:

```python
def my_function(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Short description.

    Longer description if needed.

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame.
    columns : list of str
        Column names to process.

    Returns
    -------
    pd.DataFrame
        Processed DataFrame.

    Raises
    ------
    ValueError
        If columns not found in data.

    Examples
    --------
    >>> df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    >>> result = my_function(df, ["a"])
    """
```

### Error Handling

Use appropriate exceptions:

```python
def my_function(data: pd.DataFrame, column: str) -> pd.Series:
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    return data[column]
```

## Questions?

If you have questions, please:

1. Check existing issues
2. Read the documentation
3. Open a new issue

Thank you for contributing!
