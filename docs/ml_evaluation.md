# ML Evaluation Module

The ML evaluation module provides utilities for evaluating and visualizing machine learning model performance.

## Overview

The ML evaluation module is organized into several submodules:

- **metrics** - Metric calculations and summaries
- **classification** - Classification visualizations
- **regression** - Regression diagnostics
- **feature_importance** - Feature importance visualizations
- **learning_curves** - Learning and validation curves

## Getting Started

```python
from ds_utils.ml_eval import (
    classification_summary,
    regression_summary,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_residuals,
    plot_feature_importance,
    plot_learning_curve,
)
```

## Metrics Functions

### `classification_summary(y_true, y_pred, y_proba=None)`

Returns a comprehensive summary of classification metrics.

```python
from ds_utils.ml_eval import classification_summary

metrics = classification_summary(y_test, y_pred, y_proba)
print(metrics)
```

**Returns a dictionary containing:**

| Metric | Description |
|--------|-------------|
| `accuracy` | Overall accuracy |
| `precision` | Weighted precision |
| `recall` | Weighted recall |
| `f1` | Weighted F1 score |
| `roc_auc` | ROC AUC (if `y_proba` provided) |
| `log_loss` | Log loss (if `y_proba` provided) |
| `confusion_matrix` | Confusion matrix |

### `regression_summary(y_true, y_pred)`

Returns a comprehensive summary of regression metrics.

```python
from ds_utils.ml_eval import regression_summary

metrics = regression_summary(y_test, y_pred)
print(metrics)
```

**Returns a dictionary containing:**

| Metric | Description |
|--------|-------------|
| `r2` | R-squared |
| `mae` | Mean Absolute Error |
| `mse` | Mean Squared Error |
| `rmse` | Root Mean Squared Error |
| `mape` | Mean Absolute Percentage Error |
| `max_error` | Maximum error |

### `cross_validation_summary(cv_results)`

Summarizes cross-validation results.

```python
from sklearn.model_selection import cross_validate
from ds_utils.ml_eval import cross_validation_summary

cv_results = cross_validate(model, X, y, cv=5, scoring=['accuracy', 'f1'])
summary = cross_validation_summary(cv_results)
```

### `calculate_lift(y_true, y_proba, n_bins=10)`

Calculates lift values for a classifier.

```python
from ds_utils.ml_eval import calculate_lift

lift_df = calculate_lift(y_test, y_proba, n_bins=10)
```

## Classification Visualizations

### `plot_confusion_matrix(y_true, y_pred, **kwargs)`

Creates a confusion matrix visualization.

```python
from ds_utils.ml_eval import plot_confusion_matrix

fig, ax = plot_confusion_matrix(
    y_test, y_pred,
    labels=["Negative", "Positive"],
    normalize=None,  # "true", "pred", or "all"
    title="Confusion Matrix",
    cmap="Blues",
    show_values=True,
    show_percentages=False,
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `y_true` | array-like | required | True labels |
| `y_pred` | array-like | required | Predicted labels |
| `labels` | list | None | Class labels |
| `normalize` | str | None | Normalization mode |
| `title` | str | "Confusion Matrix" | Plot title |
| `cmap` | str | "Blues" | Colormap |
| `figsize` | tuple | (8, 6) | Figure size |
| `ax` | Axes | None | Existing axes |

### `plot_roc_curve(y_true, y_proba, **kwargs)`

Creates an ROC curve visualization.

```python
from ds_utils.ml_eval import plot_roc_curve

fig, ax = plot_roc_curve(
    y_test, y_proba,
    title="ROC Curve",
    show_auc=True,
    show_threshold=False,
)
```

### `plot_precision_recall_curve(y_true, y_proba, **kwargs)`

Creates a Precision-Recall curve visualization.

```python
from ds_utils.ml_eval import plot_precision_recall_curve

fig, ax = plot_precision_recall_curve(
    y_test, y_proba,
    title="Precision-Recall Curve",
    show_ap=True,  # Show Average Precision
)
```

### `plot_calibration_curve(y_true, y_proba, **kwargs)`

Creates a calibration curve visualization.

```python
from ds_utils.ml_eval import plot_calibration_curve

fig, ax = plot_calibration_curve(
    y_test, y_proba,
    title="Calibration Curve",
    n_bins=10,
    strategy="uniform",  # or "quantile"
)
```

### `plot_classification_report(y_true, y_pred, **kwargs)`

Creates a heatmap visualization of the classification report.

```python
from ds_utils.ml_eval import plot_classification_report

fig, ax = plot_classification_report(
    y_test, y_pred,
    labels=["Class A", "Class B", "Class C"],
    title="Classification Report",
)
```

## Regression Visualizations

### `plot_residuals(y_true, y_pred, **kwargs)`

Creates a residuals vs predicted values plot.

```python
from ds_utils.ml_eval import plot_residuals

fig, ax = plot_residuals(
    y_test, y_pred,
    title="Residuals vs Predicted",
    show_zero_line=True,
    show_lowess=True,
)
```

### `plot_prediction_error(y_true, y_pred, **kwargs)`

Creates an actual vs predicted values plot.

```python
from ds_utils.ml_eval import plot_prediction_error

fig, ax = plot_prediction_error(
    y_test, y_pred,
    title="Actual vs Predicted",
    show_identity=True,
    show_r2=True,
)
```

### `plot_residual_distribution(y_true, y_pred, **kwargs)`

Creates a residual distribution analysis plot.

```python
from ds_utils.ml_eval import plot_residual_distribution

fig, axes = plot_residual_distribution(
    y_test, y_pred,
    title="Residual Distribution",
    show_histogram=True,
    show_qq=True,
)
```

### `plot_residuals_vs_feature(y_true, y_pred, feature, **kwargs)`

Creates a residuals vs feature plot.

```python
from ds_utils.ml_eval import plot_residuals_vs_feature

fig, ax = plot_residuals_vs_feature(
    y_test, y_pred, X_test[:, 0],
    feature_name="Feature 1",
    title="Residuals vs Feature",
)
```

## Feature Importance

### `plot_feature_importance(importances, feature_names, **kwargs)`

Creates a feature importance bar chart.

```python
from ds_utils.ml_eval import plot_feature_importance

fig, ax = plot_feature_importance(
    model.feature_importances_,
    feature_names=["feat_1", "feat_2", "feat_3"],
    top_n=20,
    title="Feature Importance",
    horizontal=True,
    show_values=True,
)
```

### `plot_permutation_importance(model, X, y, **kwargs)`

Creates a permutation importance visualization.

```python
from ds_utils.ml_eval import plot_permutation_importance

fig, ax = plot_permutation_importance(
    model, X_test, y_test,
    feature_names=feature_names,
    n_repeats=10,
    top_n=20,
    title="Permutation Importance",
)
```

### `plot_shap_summary(shap_values, X, **kwargs)`

Creates a SHAP summary plot.

```python
from ds_utils.ml_eval import plot_shap_summary
import shap

# First, calculate SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Then plot
fig, ax = plot_shap_summary(
    shap_values, X_test,
    feature_names=feature_names,
    max_display=20,
)
```

### `plot_feature_importance_comparison(importances_dict, **kwargs)`

Compares feature importances from multiple sources.

```python
from ds_utils.ml_eval import plot_feature_importance_comparison

importances = {
    "Built-in": model.feature_importances_,
    "Permutation": perm_importance,
}

fig, ax = plot_feature_importance_comparison(
    importances,
    feature_names=feature_names,
    top_n=15,
    title="Feature Importance Comparison",
)
```

## Learning Curves

### `plot_learning_curve(estimator, X, y, **kwargs)`

Creates a learning curve visualization.

```python
from ds_utils.ml_eval import plot_learning_curve

fig, ax = plot_learning_curve(
    estimator, X, y,
    cv=5,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring="accuracy",
    title="Learning Curve",
    show_std=True,
)
```

### `plot_validation_curve(estimator, X, y, param_name, param_range, **kwargs)`

Creates a validation curve visualization.

```python
from ds_utils.ml_eval import plot_validation_curve

fig, ax = plot_validation_curve(
    estimator, X, y,
    param_name="max_depth",
    param_range=[1, 2, 3, 5, 7, 10, 15, 20],
    cv=5,
    scoring="accuracy",
    title="Validation Curve",
)
```

### `plot_cv_results(cv_results, **kwargs)`

Visualizes cross-validation results.

```python
from ds_utils.ml_eval import plot_cv_results

fig, ax = plot_cv_results(
    cv_results,
    metric="test_accuracy",
    title="Cross-Validation Results",
)
```

### `plot_cv_comparison(models_cv_results, **kwargs)`

Compares cross-validation results from multiple models.

```python
from ds_utils.ml_eval import plot_cv_comparison

cv_results = {
    "Random Forest": rf_cv_results,
    "Gradient Boosting": gb_cv_results,
    "Logistic Regression": lr_cv_results,
}

fig, ax = plot_cv_comparison(
    cv_results,
    metric="test_accuracy",
    title="Model Comparison",
)
```

## Examples

### Complete Classification Workflow

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from ds_utils.ml_eval import (
    classification_summary,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_feature_importance,
    plot_learning_curve,
)

# Generate data
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

# Metrics
metrics = classification_summary(y_test, y_pred, y_proba)

# Visualizations
plot_confusion_matrix(y_test, y_pred)
plot_roc_curve(y_test, y_proba)
plot_precision_recall_curve(y_test, y_proba)
plot_feature_importance(clf.feature_importances_, feature_names)
plot_learning_curve(clf, X, y, cv=5)
```

### Complete Regression Workflow

```python
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from ds_utils.ml_eval import (
    regression_summary,
    plot_residuals,
    plot_prediction_error,
    plot_residual_distribution,
    plot_feature_importance,
)

# Generate data
X, y = make_regression(n_samples=500, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
reg = RandomForestRegressor(n_estimators=100, random_state=42)
reg.fit(X_train, y_train)

# Predictions
y_pred = reg.predict(X_test)

# Metrics
metrics = regression_summary(y_test, y_pred)

# Visualizations
plot_residuals(y_test, y_pred)
plot_prediction_error(y_test, y_pred)
plot_residual_distribution(y_test, y_pred)
plot_feature_importance(reg.feature_importances_, feature_names)
```
