# ML Evaluation Examples

This page provides examples of using the ML evaluation module.

## Setup

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from ds_utils.ml_eval import (
    classification_summary,
    regression_summary,
    cross_validation_summary,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_calibration_curve,
    plot_residuals,
    plot_prediction_error,
    plot_residual_distribution,
    plot_feature_importance,
    plot_learning_curve,
    plot_validation_curve,
)

from ds_utils.plotting import apply_corporate_style
apply_corporate_style()
```

## Classification Examples

### Generate Data and Train Model

```python
# Generate classification data
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=10,
    n_redundant=5,
    n_classes=2,
    random_state=42
)

feature_names = [f"feature_{i}" for i in range(20)]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Get predictions
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]
```

### Classification Metrics

```python
# Get comprehensive metrics
metrics = classification_summary(y_test, y_pred, y_proba)
print(metrics)
```

### Confusion Matrix

```python
# Standard confusion matrix
fig, ax = plot_confusion_matrix(
    y_test, y_pred,
    labels=["Negative", "Positive"],
    title="Confusion Matrix",
)
plt.show()

# Normalized confusion matrix
fig, ax = plot_confusion_matrix(
    y_test, y_pred,
    labels=["Negative", "Positive"],
    normalize="true",
    title="Normalized Confusion Matrix",
)
plt.show()
```

### ROC Curve

```python
fig, ax = plot_roc_curve(
    y_test, y_proba,
    title="ROC Curve",
    show_auc=True,
)
plt.show()
```

### Precision-Recall Curve

```python
fig, ax = plot_precision_recall_curve(
    y_test, y_proba,
    title="Precision-Recall Curve",
    show_ap=True,
)
plt.show()
```

### Calibration Curve

```python
fig, ax = plot_calibration_curve(
    y_test, y_proba,
    title="Calibration Curve",
    n_bins=10,
)
plt.show()
```

## Regression Examples

### Generate Data and Train Model

```python
# Generate regression data
X_reg, y_reg = make_regression(
    n_samples=500,
    n_features=10,
    n_informative=5,
    noise=20,
    random_state=42
)

# Split data
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# Train regressor
reg = RandomForestRegressor(n_estimators=100, random_state=42)
reg.fit(X_train_reg, y_train_reg)

# Get predictions
y_pred_reg = reg.predict(X_test_reg)
```

### Regression Metrics

```python
metrics = regression_summary(y_test_reg, y_pred_reg)
print(metrics)
```

### Residual Plots

```python
# Residuals vs predicted
fig, ax = plot_residuals(
    y_test_reg, y_pred_reg,
    title="Residuals vs Predicted",
)
plt.show()

# Actual vs predicted
fig, ax = plot_prediction_error(
    y_test_reg, y_pred_reg,
    title="Actual vs Predicted",
    show_r2=True,
)
plt.show()

# Residual distribution
fig, axes = plot_residual_distribution(
    y_test_reg, y_pred_reg,
    title="Residual Distribution",
    show_qq=True,
)
plt.show()
```

## Feature Importance

```python
# From model
fig, ax = plot_feature_importance(
    clf.feature_importances_,
    feature_names=feature_names,
    top_n=15,
    title="Feature Importance",
)
plt.show()
```

## Learning Curves

```python
fig, ax = plot_learning_curve(
    RandomForestClassifier(n_estimators=50, random_state=42),
    X, y,
    cv=5,
    title="Learning Curve",
    scoring="accuracy",
)
plt.show()
```

## Validation Curves

```python
fig, ax = plot_validation_curve(
    RandomForestClassifier(random_state=42),
    X, y,
    param_name="n_estimators",
    param_range=[10, 25, 50, 75, 100],
    cv=3,
    title="Validation Curve - Number of Trees",
)
plt.show()
```

## Cross-Validation Summary

```python
cv_results = cross_validate(
    RandomForestClassifier(n_estimators=50, random_state=42),
    X, y,
    cv=5,
    scoring=["accuracy", "precision", "recall", "f1"],
    return_train_score=True,
)

summary = cross_validation_summary(cv_results)
print(summary)
```
