"""
SHAP explanation functions for AKI Prediction project.

This module contains functions for generating SHAP explanations
and model interpretability analysis.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


def explain_model_with_shap(model, X_test_data, feature_names, model_name="Model", max_display=10, figsize=(10, 8)):
    """
    Generate SHAP explanations for a trained model.
    
    Parameters:
    -----------
    model : trained model
        The trained model to explain
    X_test_data : array-like
        Test data for explanation
    feature_names : list
        Names of features
    model_name : str, default="Model"
        Name of the model for display
    max_display : int, default=10
        Maximum number of features to display
    figsize : tuple, default=(10, 8)
        Figure size for the plot
    """
    
    print(f"🔍 Generating SHAP explanation for {model_name}...")
    
    try:
        # Create explainer based on model type
        if hasattr(model, 'predict_proba') and 'XGB' in str(type(model)):
            # XGBoost model
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test_data)
        elif hasattr(model, 'predict_proba') and 'RandomForest' in str(type(model)):
            # Random Forest model
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test_data)
        elif 'Logistic' in str(type(model)):
            # Logistic Regression - use LinearExplainer for better accuracy
            print(f"   📊 Using LinearExplainer for Logistic Regression")
            print(f"   📊 Test data shape: {X_test_data.shape}")
            print(f"   📊 Number of features: {X_test_data.shape[1]}")
            explainer = shap.LinearExplainer(model, X_test_data)
            shap_values = explainer.shap_values(X_test_data)
            print(f"   📊 SHAP values shape: {shap_values.shape}")
        else:
            # For other models (SVM), use KernelExplainer with larger sample
            # Use a larger subset for better reliability
            background_data = X_test_data[:200]  # Larger background for better approximation
            explanation_data = X_test_data[:100]  # More samples for explanation
            explainer = shap.KernelExplainer(model.predict_proba, background_data)
            shap_values = explainer.shap_values(explanation_data)
        
        # Create summary plot
        plt.figure(figsize=figsize)
        # Use the appropriate data for plotting based on explainer type
        if 'Logistic' in str(type(model)):
            # For Logistic Regression, use full dataset
            plot_data = X_test_data
        elif 'explanation_data' in locals():
            # For other models using subsets
            plot_data = explanation_data
        else:
            # Default to full dataset
            plot_data = X_test_data
            
        shap.summary_plot(shap_values, plot_data, feature_names=feature_names, 
                         max_display=max_display, show=False)
        plt.title(f'SHAP Feature Importance - {model_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # Save plot
        plt.savefig(f'shap_{model_name.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
        print(f"✅ SHAP plot saved as: shap_{model_name.lower().replace(' ', '_')}.png")
        
        return explainer, shap_values
        
    except Exception as e:
        print(f"❌ Error generating SHAP explanation for {model_name}: {str(e)}")
        return None, None


def explain_best_model_with_shap(models_dict, X_test_dict, feature_names, model_data_mapping=None, max_display=10):
    """
    Generate SHAP explanation for the best performing model.
    
    Parameters:
    -----------
    models_dict : dict
        Dictionary with model names as keys and fitted models as values
    X_test_dict : dict
        Dictionary with test data: {'scaled': X_test_scaled, 'imputed': X_test_imputed}
    feature_names : list
        Names of features
    model_data_mapping : dict, optional
        Dictionary mapping model names to data type ('scaled' or 'imputed')
    max_display : int, default=10
        Maximum number of features to display
    """
    from .evaluate import evaluate_models
    
    print("🔍 Finding best model for SHAP explanation...")
    
    # Evaluate all models to find the best one
    results_df = evaluate_models(models_dict, X_test_dict, None, model_data_mapping)
    best_model_name = results_df.iloc[0]['Model']
    best_model = models_dict[best_model_name]
    
    print(f"🏆 Best model: {best_model_name} (ROC-AUC: {results_df.iloc[0]['ROC-AUC']:.4f})")
    
    # Determine which test data to use for the best model
    if model_data_mapping and best_model_name in model_data_mapping:
        data_type = model_data_mapping[best_model_name]
    else:
        # Auto-detect based on model type
        if 'SVM' in best_model_name or 'Logistic' in best_model_name:
            data_type = 'scaled'
        else:
            data_type = 'imputed'
    
    # Select appropriate test data
    if data_type == 'scaled':
        X_test_data = X_test_dict['scaled']
    else:
        X_test_data = X_test_dict['imputed']
    
    # Generate SHAP explanation
    explainer, shap_values = explain_model_with_shap(
        best_model, X_test_data, feature_names, best_model_name, max_display
    )
    
    return explainer, shap_values, best_model_name


def analyze_logistic_regression_coefficients(model, feature_names, model_name="LogisticRegression"):
    """
    Analyze Logistic Regression model coefficients to understand feature importance.
    
    Parameters:
    -----------
    model : LogisticRegression model
        Trained Logistic Regression model
    feature_names : list
        Names of features
    model_name : str, default="LogisticRegression"
        Name of the model
    """
    print(f"\n🔍 {model_name} Model Analysis:")
    print(f"   📊 Model coefficients shape: {model.coef_.shape}")
    print(f"   📊 Number of non-zero coefficients: {np.count_nonzero(model.coef_[0])}")
    print(f"   📊 Feature names with non-zero coefficients:")
    
    non_zero_indices = np.where(np.abs(model.coef_[0]) > 1e-6)[0]
    for i, idx in enumerate(non_zero_indices[:10]):  # Show first 10
        print(f"      {i+1:2d}. {feature_names[idx]:<20} coeff: {model.coef_[0][idx]:8.4f}")
    
    if len(non_zero_indices) > 10:
        print(f"      ... and {len(non_zero_indices) - 10} more features")
    
    return non_zero_indices


def plot_feature_importance_coefficients(model, feature_names, model_name="Model", top_n=15):
    """
    Plot feature importance based on model coefficients.
    
    Parameters:
    -----------
    model : trained model
        Model with coefficients or feature_importances_
    feature_names : list
        Names of features
    model_name : str, default="Model"
        Name of the model
    top_n : int, default=15
        Number of top features to display
    """
    plt.figure(figsize=(10, 8))
    
    # Get feature importance based on model type
    if hasattr(model, 'coef_'):
        # Linear models (Logistic Regression)
        importance = np.abs(model.coef_[0])
        title = f'{model_name} - Feature Importance (Coefficients)'
    elif hasattr(model, 'feature_importances_'):
        # Tree-based models (Random Forest, XGBoost)
        importance = model.feature_importances_
        title = f'{model_name} - Feature Importance'
    else:
        print(f"❌ Model {model_name} doesn't have coefficients or feature_importances_")
        return
    
    # Get top features
    top_indices = np.argsort(importance)[-top_n:][::-1]
    top_features = [feature_names[i] for i in top_indices]
    top_importance = importance[top_indices]
    
    # Create horizontal bar plot
    plt.barh(range(len(top_features)), top_importance)
    plt.yticks(range(len(top_features)), top_features)
    plt.xlabel('Importance')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    
    print(f"✅ Feature importance plot created for {model_name}")


def save_shap_values(shap_values, model_name, save_dir='shap_values'):
    """
    Save SHAP values to file for later analysis.
    
    Parameters:
    -----------
    shap_values : array-like
        SHAP values to save
    model_name : str
        Name of the model
    save_dir : str, default='shap_values'
        Directory to save SHAP values
    """
    import os
    import joblib
    
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{save_dir}/{model_name}_shap_values.joblib"
    joblib.dump(shap_values, filename)
    print(f"✅ SHAP values saved to: {filename}")


def load_shap_values(model_name, save_dir='shap_values'):
    """
    Load SHAP values from file.
    
    Parameters:
    -----------
    model_name : str
        Name of the model
    save_dir : str, default='shap_values'
        Directory containing SHAP values
    
    Returns:
    --------
    array-like or None
    """
    import joblib
    
    filename = f"{save_dir}/{model_name}_shap_values.joblib"
    try:
        shap_values = joblib.load(filename)
        print(f"✅ SHAP values loaded from: {filename}")
        return shap_values
    except FileNotFoundError:
        print(f"❌ SHAP values file not found: {filename}")
        return None
