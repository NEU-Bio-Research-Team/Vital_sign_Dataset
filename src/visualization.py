"""
Visualization functions for AKI Prediction project.

This module contains functions for plotting ROC curves, PR curves,
model comparisons, and other visualizations.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc, confusion_matrix


def plot_roc_curves(models_dict, X_test_dict, y_test, model_data_mapping=None, figsize=(12, 8)):
    """
    Plot ROC curves for multiple models with enhanced multi-curve visualization.
    
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
    figsize : tuple, default=(12, 8)
        Figure size for the plot
    """
    
    plt.figure(figsize=figsize)
    
    # Plot diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier (AUC = 0.500)', alpha=0.8, linewidth=2)
    
    # Enhanced color palette for multiple models
    colors = plt.cm.tab10(np.linspace(0, 1, len(models_dict)))
    linestyles = ['-', '--', '-.', ':'] * (len(models_dict) // 4 + 1)
    
    successful_plots = 0
    auc_scores = []
    
    for i, (model_name, model) in enumerate(models_dict.items()):
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
            # Get prediction probabilities
            y_pred_proba = model.predict_proba(X_test_data)[:, 1]
            
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            # Plot ROC curve with enhanced styling
            plt.plot(fpr, tpr, 
                    color=colors[i], 
                    linestyle=linestyles[i],
                    lw=3, 
                    alpha=0.8,
                    label=f'{model_name} (AUC = {roc_auc:.3f})')
            
            auc_scores.append(roc_auc)
            successful_plots += 1
            
        except Exception as e:
            print(f"⚠️ Error plotting {model_name}: {str(e)}")
            continue
    
    if successful_plots == 0:
        print("❌ No models could be plotted successfully")
        plt.close()
        return
    
    # Enhanced plot styling
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=14, fontweight='bold')
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=14, fontweight='bold')
    plt.title(f'ROC Curves Comparison ({successful_plots} Models)', fontsize=16, fontweight='bold', pad=20)
    
    # Enhanced legend
    plt.legend(loc="lower right", fontsize=11, framealpha=0.9, shadow=True)
    
    # Enhanced grid
    plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Add performance summary
    if auc_scores:
        best_auc = max(auc_scores)
        plt.figtext(0.02, 0.02, f'Best AUC: {best_auc:.3f}', fontsize=10, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    
    plt.tight_layout()
    plt.show()
    
    print(f"✅ Successfully plotted ROC curves for {successful_plots} models")


def plot_pr_curves(models_dict, X_test_dict, y_test, model_data_mapping=None, figsize=(12, 8)):
    """
    Plot Precision-Recall curves for multiple models.
    
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
    figsize : tuple, default=(12, 8)
        Figure size for the plot
    """
    
    plt.figure(figsize=figsize)
    
    # Calculate baseline (random classifier) precision
    baseline_precision = np.sum(y_test) / len(y_test)
    plt.axhline(y=baseline_precision, color='k', linestyle='--', 
               label=f'Random Classifier (Precision = {baseline_precision:.3f})', alpha=0.8)
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(models_dict)))
    
    for i, (model_name, model) in enumerate(models_dict.items()):
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
            # Get prediction probabilities
            y_pred_proba = model.predict_proba(X_test_data)[:, 1]
            
            # Calculate PR curve
            precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_proba)
            auprc = auc(recall_vals, precision_vals)
            
            # Plot PR curve
            plt.plot(recall_vals, precision_vals, color=colors[i], lw=2, 
                    label=f'{model_name} (AUPRC = {auprc:.3f})')
            
        except Exception as e:
            print(f"⚠️ Error plotting {model_name}: {str(e)}")
            continue
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall (Sensitivity)', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_model_comparison(results_df, metrics=['ROC-AUC', 'AUPRC', 'Precision', 'Recall', 'F1-Score'], figsize=(15, 10)):
    """
    Plot comparison of multiple metrics across models.
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        Results dataframe from evaluate_models function
    metrics : list
        List of metrics to plot
    figsize : tuple, default=(15, 10)
        Figure size for the plot
    """
    
    # Filter available metrics
    available_metrics = [m for m in metrics if m in results_df.columns]
    
    if not available_metrics:
        print("⚠️ No valid metrics found in results dataframe")
        return
    
    # Create subplots
    n_metrics = len(available_metrics)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(results_df)))
    
    for i, metric in enumerate(available_metrics):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        
        # Create bar plot
        bars = ax.bar(results_df['Model'], results_df[metric], color=colors)
        
        # Add value labels on bars
        for bar, value in zip(bars, results_df[metric]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_title(f'{metric} Comparison', fontweight='bold')
        ax.set_ylabel(metric)
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Rotate x-axis labels if needed
        if len(results_df['Model'].iloc[0]) > 10:
            ax.tick_params(axis='x', rotation=45)
    
    # Hide empty subplots
    for i in range(n_metrics, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        if n_rows > 1:
            axes[row, col].set_visible(False)
        else:
            axes[col].set_visible(False)
    
    plt.tight_layout()
    plt.show()


def plot_confusion_matrices(models_dict, X_test_dict, y_test, model_data_mapping=None, figsize=(16, 12)):
    """
    Plot confusion matrices for multiple models.
    
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
    figsize : tuple, default=(16, 12)
        Figure size for the plot
    """
    
    n_models = len(models_dict)
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Ensure axes is always a flat array for easy indexing
    if n_models == 1:
        axes = np.array([axes])
    elif n_rows == 1 and n_cols > 1:
        axes = axes.reshape(1, -1).flatten()
    elif n_rows > 1 and n_cols == 1:
        axes = axes.reshape(-1, 1).flatten()
    elif n_rows > 1 and n_cols > 1:
        axes = axes.flatten()
    else:
        axes = np.array([axes])
    
    successful_plots = 0
    
    for i, (model_name, model) in enumerate(models_dict.items()):
        ax = axes[i]
        
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
            # Get predictions
            y_pred = model.predict(X_test_data)
            
            # Calculate confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            # Plot confusion matrix
            im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            ax.figure.colorbar(im, ax=ax)
            
            # Add text annotations
            thresh = cm.max() / 2.
            for row in range(cm.shape[0]):
                for col in range(cm.shape[1]):
                    ax.text(col, row, format(cm[row, col], 'd'),
                           ha="center", va="center",
                           color="white" if cm[row, col] > thresh else "black",
                           fontsize=14, fontweight='bold')
            
            # Set labels and title
            ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
            ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
            ax.set_title(f'{model_name}\nConfusion Matrix', fontsize=14, fontweight='bold')
            
            # Set tick labels
            class_names = ['No AKI', 'AKI']
            tick_marks = np.arange(len(class_names))
            ax.set_xticks(tick_marks)
            ax.set_yticks(tick_marks)
            ax.set_xticklabels(class_names)
            ax.set_yticklabels(class_names)
            
            # Calculate and display metrics
            tn, fp, fn, tp = cm.ravel()
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Add metrics text box
            metrics_text = f'Acc: {accuracy:.3f}\nPrec: {precision:.3f}\nRec: {recall:.3f}\nF1: {f1:.3f}'
            ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            successful_plots += 1
            
        except Exception as e:
            # Ensure ax is a matplotlib axes object before using text methods
            if hasattr(ax, 'text'):
                ax.text(0.5, 0.5, f'Error plotting\n{model_name}:\n{str(e)}', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
                ax.set_title(f'{model_name}\n(Error)', fontsize=14, color='red')
            print(f"⚠️ Error plotting confusion matrix for {model_name}: {str(e)}")
    
    # Hide empty subplots
    for i in range(n_models, len(axes)):
        if hasattr(axes[i], 'set_visible'):
            axes[i].set_visible(False)
    
    plt.suptitle(f'Confusion Matrices Comparison ({successful_plots} Models)', 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()
    
    print(f"✅ Successfully plotted confusion matrices for {successful_plots} models")


def save_plots(fig, filename, save_dir='plots'):
    """
    Save plots to specified directory.
    
    Parameters:
    -----------
    fig : matplotlib figure
        Figure to save
    filename : str
        Filename for the plot
    save_dir : str, default='plots'
        Directory to save plots
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(f"{save_dir}/{filename}", dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved to: {save_dir}/{filename}")
