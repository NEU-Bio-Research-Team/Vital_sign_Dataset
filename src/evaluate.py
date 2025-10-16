"""
Evaluation functions for AKI Prediction project.

This module contains functions for model evaluation, metrics calculation,
and performance analysis.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (roc_curve, roc_auc_score, precision_recall_curve, auc, 
                           accuracy_score, precision_score, recall_score, f1_score,
                           confusion_matrix, classification_report)


def calculate_comprehensive_metrics(y_true, y_pred, y_pred_proba, model_name="Model"):
    """
    Calculate comprehensive evaluation metrics including PPV, AUPRC, Precision, Recall.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_pred : array-like
        Predicted binary labels
    y_pred_proba : array-like
        Predicted probabilities for positive class
    model_name : str, default="Model"
        Name of the model for display
    
    Returns:
    --------
    dict : Dictionary containing all calculated metrics
    
    Usage:
    ------
    metrics = calculate_comprehensive_metrics(y_true, y_pred, y_pred_proba, "LogisticRegression")
    """
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # PPV is the same as precision
    ppv = precision_score(y_true, y_pred, zero_division=0)
    
    # AUC metrics
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_pred_proba)
    auprc = auc(recall_vals, precision_vals)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Additional metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    metrics = {
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'PPV': ppv,
        'Recall': recall,
        'Specificity': specificity,
        'NPV': npv,
        'F1-Score': f1,
        'ROC-AUC': roc_auc,
        'AUPRC': auprc,
        'True Positives': tp,
        'False Positives': fp,
        'True Negatives': tn,
        'False Negatives': fn
    }
    
    return metrics


def evaluate_models(models_dict, X_test_dict, y_test, model_data_mapping=None):
    """
    Modular function to evaluate multiple models and return comprehensive metrics.
    
    Parameters:
    -----------
    models_dict : dict
        Dictionary with model names as keys and fitted models as values
    X_test_dict : dict
        Dictionary with test data: {'scaled': X_test_scaled, 'imputed': X_test_imputed}
    y_test : array-like
        True test labels
    model_data_mapping : dict, optional
        Dictionary mapping model names to data type ('scaled' or 'imputed')
        If None, will try to determine automatically
    
    Returns:
    --------
    pandas.DataFrame : Results dataframe with all metrics
    """
    
    print("📊 Starting Comprehensive Model Evaluation...")
    print("=" * 70)
    
    results = []
    
    for model_name, model in models_dict.items():
        print(f"\n🔍 Evaluating {model_name}...")
        
        # Determine which test data to use
        if model_data_mapping and model_name in model_data_mapping:
            data_type = model_data_mapping[model_name]
        else:
            # Auto-detect based on model type
            if 'SVM' in model_name or 'Logistic' in model_name:
                data_type = 'scaled'
            else:
                data_type = 'imputed'
        
        # Select appropriate test data
        if data_type == 'scaled':
            X_test_data = X_test_dict['scaled']
        else:
            X_test_data = X_test_dict['imputed']
        
        try:
            # Make predictions
            y_pred = model.predict(X_test_data)
            y_pred_proba = model.predict_proba(X_test_data)[:, 1]
            
            # Calculate comprehensive metrics
            metrics = calculate_comprehensive_metrics(y_test, y_pred, y_pred_proba, model_name)
            results.append(metrics)
            
            print(f"   ✅ ROC-AUC: {metrics['ROC-AUC']:.4f}")
            print(f"   ✅ Accuracy: {metrics['Accuracy']:.4f}")
            print(f"   ✅ F1-Score: {metrics['F1-Score']:.4f}")
            
        except Exception as e:
            print(f"   ❌ Error evaluating {model_name}: {str(e)}")
            continue
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort by ROC-AUC (descending)
    results_df = results_df.sort_values('ROC-AUC', ascending=False).reset_index(drop=True)
    
    print(f"\n🎉 Evaluation completed for {len(results_df)} models!")
    print(f"🏆 Best model: {results_df.iloc[0]['Model']} (ROC-AUC: {results_df.iloc[0]['ROC-AUC']:.4f})")
    
    return results_df


def evaluate_single_model(model, model_name, X_test_data, y_test):
    """
    Evaluate a single model and return predictions and metrics.
    
    Parameters:
    -----------
    model : trained model
        Model to evaluate
    model_name : str
        Name of the model
    X_test_data : array-like
        Test features
    y_test : array-like
        True test labels
    
    Returns:
    --------
    tuple: (y_pred, y_pred_proba, metrics)
    """
    print(f"🔍 Evaluating {model_name}...")
    
    # Make predictions
    y_pred = model.predict(X_test_data)
    y_pred_proba = model.predict_proba(X_test_data)[:, 1]
    
    # Calculate metrics
    metrics = calculate_comprehensive_metrics(y_test, y_pred, y_pred_proba, model_name)
    
    print(f"   ✅ ROC-AUC: {metrics['ROC-AUC']:.4f}")
    print(f"   ✅ Accuracy: {metrics['Accuracy']:.4f}")
    print(f"   ✅ F1-Score: {metrics['F1-Score']:.4f}")
    
    return y_pred, y_pred_proba, metrics


def print_evaluation_summary(results_df):
    """
    Print a formatted summary of evaluation results.
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        Results dataframe from evaluate_models function
    """
    print(f"\n📈 Evaluation Results Summary:")
    print(f"{'Model':<20} {'ROC-AUC':<10} {'AUPRC':<10} {'Accuracy':<10} {'F1-Score':<10} {'PPV':<10}")
    print("-" * 80)

    for _, row in results_df.iterrows():
        print(f"{row['Model']:<20} {row['ROC-AUC']:<10.4f} {row['AUPRC']:<10.4f} {row['Accuracy']:<10.4f} {row['F1-Score']:<10.4f} {row['PPV']:<10.4f}")

    print(f"\n🏆 FINAL RANKINGS:")
    print(f"{'Rank':<5} {'Model':<20} {'ROC-AUC':<10} {'Accuracy':<10} {'F1-Score':<10}")
    print("-" * 70)

    for i, (_, row) in enumerate(results_df.iterrows(), 1):
        print(f"{i:<5} {row['Model']:<20} {row['ROC-AUC']:<10.4f} {row['Accuracy']:<10.4f} {row['F1-Score']:<10.4f}")


def save_evaluation_results(results_df, filename='evaluation_results.csv'):
    """
    Save evaluation results to CSV file.
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        Results dataframe to save
    filename : str, default='evaluation_results.csv'
        Filename to save results
    """
    results_df.to_csv(filename, index=False)
    print(f"✅ Evaluation results saved to: {filename}")


def load_evaluation_results(filename='evaluation_results.csv'):
    """
    Load evaluation results from CSV file.
    
    Parameters:
    -----------
    filename : str, default='evaluation_results.csv'
        Filename to load results from
    
    Returns:
    --------
    pandas.DataFrame or None
    """
    try:
        results_df = pd.read_csv(filename)
        print(f"✅ Evaluation results loaded from: {filename}")
        return results_df
    except FileNotFoundError:
        print(f"❌ Results file not found: {filename}")
        return None
