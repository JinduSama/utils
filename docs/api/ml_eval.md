# ML Evaluation Module API Reference

::: ds_utils.ml_eval.metrics
    options:
      show_source: true
      members:
        - classification_summary
        - regression_summary
        - cross_validation_summary
        - calculate_lift

---

::: ds_utils.ml_eval.classification
    options:
      show_source: true
      members:
        - plot_confusion_matrix
        - plot_roc_curve
        - plot_precision_recall_curve
        - plot_calibration_curve
        - plot_classification_report

---

::: ds_utils.ml_eval.regression
    options:
      show_source: true
      members:
        - plot_residuals
        - plot_prediction_error
        - plot_residual_distribution
        - plot_residuals_vs_feature

---

::: ds_utils.ml_eval.feature_importance
    options:
      show_source: true
      members:
        - plot_feature_importance
        - plot_permutation_importance
        - plot_shap_summary
        - plot_feature_importance_comparison

---

::: ds_utils.ml_eval.learning_curves
    options:
      show_source: true
      members:
        - plot_learning_curve
        - plot_validation_curve
        - plot_cv_results
        - plot_cv_comparison
