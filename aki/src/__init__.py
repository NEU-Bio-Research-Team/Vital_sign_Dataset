"""
AKI Prediction Package

A comprehensive machine learning package for predicting postoperative Acute Kidney Injury (AKI)
using vital signs and clinical data from VitalDB.

This package provides:
- Data loading and preprocessing utilities
- Model training and hyperparameter tuning
- Comprehensive model evaluation
- Visualization tools
- SHAP-based model interpretability

Modules:
--------
- utils: Data loading, preprocessing, and utility functions
- train: Model training and hyperparameter tuning
- evaluate: Model evaluation and metrics calculation
- visualization: Plotting and visualization functions
- shap_explainer: SHAP-based model interpretability
"""

from .utils import (
    setup_plotting,
    load_vitaldb_data,
    preprocess_data,
    prepare_train_test_data,
    save_predictions,
    load_predictions
)

from .train import (
    get_default_model_configs,
    hyperparameter_tuning,
    train_single_model,
    save_model,
    load_model,
    save_best_model
)

from .evaluate import (
    calculate_comprehensive_metrics,
    evaluate_models,
    evaluate_single_model,
    print_evaluation_summary,
    save_evaluation_results,
    load_evaluation_results
)

from .visualization import (
    plot_roc_curves,
    plot_pr_curves,
    plot_model_comparison,
    plot_confusion_matrices,
    save_plots
)

from .shap_explainer import (
    explain_model_with_shap,
    explain_best_model_with_shap,
    analyze_logistic_regression_coefficients,
    plot_feature_importance_coefficients,
    save_shap_values,
    load_shap_values
)

__version__ = "1.0.0"
__author__ = "NEU Bio Research Team"
__email__ = "contact@example.com"

__all__ = [
    # Utils
    'setup_plotting',
    'load_vitaldb_data',
    'preprocess_data',
    'prepare_train_test_data',
    'save_predictions',
    'load_predictions',
    
    # Training
    'get_default_model_configs',
    'hyperparameter_tuning',
    'train_single_model',
    'save_model',
    'load_model',
    'save_best_model',
    
    # Evaluation
    'calculate_comprehensive_metrics',
    'evaluate_models',
    'evaluate_single_model',
    'print_evaluation_summary',
    'save_evaluation_results',
    'load_evaluation_results',
    
    # Visualization
    'plot_roc_curves',
    'plot_pr_curves',
    'plot_model_comparison',
    'plot_confusion_matrices',
    'save_plots',
    
    # SHAP
    'explain_model_with_shap',
    'explain_best_model_with_shap',
    'analyze_logistic_regression_coefficients',
    'plot_feature_importance_coefficients',
    'save_shap_values',
    'load_shap_values'
]
