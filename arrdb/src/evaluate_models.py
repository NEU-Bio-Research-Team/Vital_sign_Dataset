"""
Evaluation utilities for Arrhythmia Classification models.

Functions for calculating multi-class metrics, confusion matrices,
ROC curves, and visualization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_fscore_support, roc_curve, auc,
    average_precision_score, precision_recall_curve
)
from sklearn.preprocessing import label_binarize


def evaluate_multi_class(y_true, y_pred, y_pred_proba, class_names, task_name="Classification", label_encoder=None):
    """
    Evaluate multi-class classification performance.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    y_pred_proba : array-like
        Predicted probabilities (n_samples, n_classes)
    class_names : list
        List of class names
    task_name : str
        Name of task (for display)
    
    Returns:
    --------
    dict : Dictionary of metrics
    """
    # Convert to arrays for easier manipulation
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Filter out samples with labels not in class_names (unseen classes)
    # This handles cases where test set has classes not seen during training
    valid_mask = np.isin(y_true, class_names) & np.isin(y_pred, class_names)
    
    if np.sum(valid_mask) == 0:
        print(f"Warning: No valid samples after filtering. All samples have unseen classes.")
        # Return NaN metrics
        return {
            'task_name': task_name,
            'accuracy': np.nan,
            'precision_macro': np.nan,
            'recall_macro': np.nan,
            'f1_macro': np.nan,
            'f1_micro': np.nan,
            'f1_weighted': np.nan,
            'roc_auc_macro': np.nan,
            'auprc_macro': np.nan,
            'top3_accuracy': np.nan,
            'confusion_matrix': np.array([]),
            'per_class': {'precision': [], 'recall': [], 'f1': [], 'support': []},
            'class_names': class_names
        }
    
    # Filter to only valid samples
    y_true_filtered = y_true[valid_mask]
    y_pred_filtered = y_pred[valid_mask]
    y_pred_proba_filtered = y_pred_proba[valid_mask]
    
    if len(y_true_filtered) < len(y_true):
        n_filtered = len(y_true) - len(y_true_filtered)
        print(f"Warning: Filtered out {n_filtered} samples with unseen classes")
    
    # Use filtered data for all calculations
    y_true = y_true_filtered
    y_pred = y_pred_filtered
    y_pred_proba = y_pred_proba_filtered
    
    # Overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=class_names, zero_division=0
    )
    
    # Macro averages
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Micro averages (kept for completeness; not shown in the requested table)
    precision_micro = precision_score(y_true, y_pred, average='micro', zero_division=0)
    recall_micro = recall_score(y_true, y_pred, average='micro', zero_division=0)
    f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
    
    # Weighted averages
    precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # ROC-AUC (one-vs-rest)
    # Key insight: y_pred_proba columns are in the order of label_encoder.classes_ (if encoder exists)
    # Otherwise, sklearn models return probabilities in sorted order of unique classes seen during training
    try:
        # Get actual classes present in y_true and y_pred (both already filtered to class_names)
        actual_classes = sorted(list(set(y_true) | set(y_pred)))
        # Ensure all classes are in class_names
        actual_classes = [c for c in actual_classes if c in class_names]
        
        n_proba_classes = y_pred_proba.shape[1]
        
        # Determine the order of classes in y_pred_proba
        if label_encoder is not None:
            # Probabilities are in the order of label_encoder.classes_
            model_classes = list(label_encoder.classes_)
        else:
            # Fallback: assume sorted order (sklearn default)
            model_classes = sorted(list(set(y_pred)))
        
        # Filter model_classes to only include those in actual_classes
        model_classes = [c for c in model_classes if c in actual_classes]
        
        # If we have fewer model_classes than proba columns, there's a mismatch
        # Use only the classes that match
        if len(model_classes) != n_proba_classes:
            # Use actual_classes but ensure they match proba shape
            if n_proba_classes <= len(actual_classes):
                # Take first n_proba_classes from actual_classes
                actual_classes = sorted(actual_classes)[:n_proba_classes]
                model_classes = [c for c in model_classes if c in actual_classes]
            else:
                # This shouldn't happen, but handle it
                model_classes = model_classes[:n_proba_classes]
        
        # Now align probabilities: y_pred_proba columns correspond to model_classes
        # We need to map them to actual_classes order for label_binarize
        if len(actual_classes) == 0:
            roc_auc = np.nan
        elif len(actual_classes) == 1:
            # Only one class - cannot calculate ROC-AUC
            roc_auc = np.nan
        elif len(actual_classes) == 2:
            # Binary case
            # Find which column in y_pred_proba corresponds to the positive class
            positive_class = actual_classes[1]  # Second class as positive
            if positive_class in model_classes:
                pos_idx = model_classes.index(positive_class)
                if pos_idx < y_pred_proba.shape[1]:
                    y_true_binary = np.array([1 if y == positive_class else 0 for y in y_true])
                    roc_auc = roc_auc_score(y_true_binary, y_pred_proba[:, pos_idx])
                else:
                    roc_auc = np.nan
            else:
                # Positive class not in model - use first available
                if len(model_classes) >= 2:
                    pos_idx = 1
                    pos_class = model_classes[1]
                    y_true_binary = np.array([1 if y == pos_class else 0 for y in y_true])
                    roc_auc = roc_auc_score(y_true_binary, y_pred_proba[:, pos_idx])
                else:
                    roc_auc = np.nan
        else:
            # Multi-class case
            # Create binary labels for actual_classes
            y_true_binary = label_binarize(y_true, classes=actual_classes)
            
            # Map y_pred_proba columns (ordered by model_classes) to actual_classes order
            proba_aligned = np.zeros((len(y_true), len(actual_classes)))
            for i, cls in enumerate(actual_classes):
                if cls in model_classes:
                    idx_in_model = model_classes.index(cls)
                    if idx_in_model < y_pred_proba.shape[1]:
                        proba_aligned[:, i] = y_pred_proba[:, idx_in_model]
                    else:
                        # Class not in probabilities - fill with zeros (shouldn't happen)
                        proba_aligned[:, i] = 0.0
                else:
                    # Class not in model - fill with zeros
                    proba_aligned[:, i] = 0.0
            
            # Calculate ROC-AUC
            if y_true_binary.shape[1] == proba_aligned.shape[1] and y_true_binary.shape[1] > 1:
                # Check if all classes have at least one positive sample
                has_positive = np.sum(y_true_binary, axis=0) > 0
                if np.sum(has_positive) < 2:
                    # Need at least 2 classes with positive samples for macro ROC-AUC
                    roc_auc = np.nan
                else:
                    # Only calculate ROC-AUC for classes with positive samples
                    roc_auc_per_class = []
                    for i in range(len(actual_classes)):
                        if has_positive[i]:
                            try:
                                auc_score = roc_auc_score(y_true_binary[:, i], proba_aligned[:, i])
                                if not np.isnan(auc_score):
                                    roc_auc_per_class.append(auc_score)
                            except:
                                pass
                    if len(roc_auc_per_class) > 0:
                        roc_auc = np.mean(roc_auc_per_class)
                    else:
                        roc_auc = np.nan
            else:
                roc_auc = np.nan
    except Exception as e:
        print(f"Warning: Could not calculate ROC-AUC: {e}")
        import traceback
        traceback.print_exc()
        roc_auc = np.nan
    
    # AUPRC (Average Precision) for multi-class (one-vs-rest)
    # Use the same alignment logic as ROC-AUC
    try:
        # Get actual classes (reuse the logic from ROC-AUC)
        actual_classes = sorted(list(set(y_true) | set(y_pred)))
        actual_classes = [c for c in actual_classes if c in class_names]
        
        n_proba_classes = y_pred_proba.shape[1]
        
        # Determine the order of classes in y_pred_proba (same as ROC-AUC)
        if label_encoder is not None:
            model_classes = list(label_encoder.classes_)
        else:
            model_classes = sorted(list(set(y_pred)))
        
        # Filter model_classes to only include those in actual_classes
        model_classes = [c for c in model_classes if c in actual_classes]
        
        # Ensure alignment
        if len(model_classes) != n_proba_classes:
            if n_proba_classes <= len(actual_classes):
                actual_classes = sorted(actual_classes)[:n_proba_classes]
                model_classes = [c for c in model_classes if c in actual_classes]
            else:
                model_classes = model_classes[:n_proba_classes]
        
        if len(actual_classes) == 0 or len(actual_classes) == 1:
            auprc = np.nan
        elif len(actual_classes) == 2:
            # Binary case
            positive_class = actual_classes[1]
            if positive_class in model_classes:
                pos_idx = model_classes.index(positive_class)
                if pos_idx < y_pred_proba.shape[1]:
                    y_true_binary = np.array([1 if y == positive_class else 0 for y in y_true])
                    auprc = average_precision_score(y_true_binary, y_pred_proba[:, pos_idx])
                else:
                    auprc = np.nan
            else:
                if len(model_classes) >= 2:
                    pos_idx = 1
                    pos_class = model_classes[1]
                    y_true_binary = np.array([1 if y == pos_class else 0 for y in y_true])
                    auprc = average_precision_score(y_true_binary, y_pred_proba[:, pos_idx])
                else:
                    auprc = np.nan
        else:
            # Multi-class case
            y_true_binary = label_binarize(y_true, classes=actual_classes)
            
            # Map probabilities (same as ROC-AUC)
            proba_aligned = np.zeros((len(y_true), len(actual_classes)))
            for i, cls in enumerate(actual_classes):
                if cls in model_classes:
                    idx_in_model = model_classes.index(cls)
                    if idx_in_model < y_pred_proba.shape[1]:
                        proba_aligned[:, i] = y_pred_proba[:, idx_in_model]
                    else:
                        proba_aligned[:, i] = 0.0
                else:
                    proba_aligned[:, i] = 0.0
            
            # Calculate AUPRC per class
            if y_true_binary.shape[1] == proba_aligned.shape[1] and y_true_binary.shape[1] > 1:
                auprc_per_class = []
                for i in range(len(actual_classes)):
                    # Skip classes with no positive samples
                    if np.sum(y_true_binary[:, i]) > 0:
                        ap = average_precision_score(y_true_binary[:, i], proba_aligned[:, i])
                        if not np.isnan(ap):
                            auprc_per_class.append(ap)
                
                if len(auprc_per_class) > 0:
                    auprc = np.mean(auprc_per_class)  # Macro average
                else:
                    auprc = np.nan
            else:
                auprc = np.nan
    except Exception as e:
        print(f"Warning: Could not calculate AUPRC: {e}")
        import traceback
        traceback.print_exc()
        auprc = np.nan
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    
    # Top-k accuracy for rare classes
    top3_accuracy = calculate_top_k_accuracy(y_true, y_pred_proba, k=3, class_names=class_names)
    
    metrics = {
        'task_name': task_name,
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'roc_auc_macro': roc_auc,
        'auprc_macro': auprc,
        'top3_accuracy': top3_accuracy,
        'confusion_matrix': cm,
        'per_class': {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support
        },
        'class_names': class_names
    }
    
    return metrics


def calculate_top_k_accuracy(y_true, y_pred_proba, k=3, class_names=None):
    """
    Calculate top-k accuracy.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred_proba : array-like
        Predicted probabilities
    k : int, default=3
        k for top-k accuracy
    class_names : list, optional
        Class names
    
    Returns:
    --------
    float : Top-k accuracy
    """
    # Convert labels to indices
    if class_names is not None:
        label_to_idx = {label: idx for idx, label in enumerate(class_names)}
        y_true_idx = np.array([label_to_idx[label] for label in y_true])
    else:
        y_true_idx = y_true
    
    # Get top-k predictions
    top_k_preds = np.argsort(y_pred_proba, axis=1)[:, -k:]
    
    # Check if true label is in top-k
    correct = 0
    for i, true_label in enumerate(y_true_idx):
        if true_label in top_k_preds[i]:
            correct += 1
    
    top_k_acc = correct / len(y_true)
    return top_k_acc


def plot_confusion_matrix(cm, class_names, title, figsize=(12, 10), normalize=False):
    """
    Plot confusion matrix heatmap.
    
    Parameters:
    -----------
    cm : array-like
        Confusion matrix
    class_names : list
        Class names
    title : str
        Plot title
    figsize : tuple, default=(12, 10)
        Figure size
    normalize : bool, default=False
        Whether to normalize the matrix
    """
    plt.figure(figsize=figsize)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count' if not normalize else 'Proportion'})
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()


def plot_per_class_metrics(metrics_dict, figsize=(12, 6)):
    """
    Plot per-class metrics (precision, recall, F1).
    
    Parameters:
    -----------
    metrics_dict : dict
        Metrics dictionary from evaluate_multi_class
    figsize : tuple, default=(12, 6)
        Figure size
    """
    class_names = metrics_dict['class_names']
    per_class = metrics_dict['per_class']
    
    x = np.arange(len(class_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=figsize)
    
    precision = per_class['precision']
    recall = per_class['recall']
    f1 = per_class['f1']
    
    ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
    ax.bar(x, recall, width, label='Recall', alpha=0.8)
    ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'Per-Class Metrics - {metrics_dict["task_name"]}', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([0, 1])
    plt.tight_layout()


def plot_roc_curves_multiclass(y_true, y_pred_proba, class_names, title, figsize=(10, 8)):
    """
    Plot ROC curves for multi-class classification (one-vs-rest).
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred_proba : array-like
        Predicted probabilities
    class_names : list
        Class names
    title : str
        Plot title
    figsize : tuple, default=(10, 8)
        Figure size
    """
    # Binarize labels
    y_true_binary = label_binarize(y_true, classes=class_names)
    
    # Compute ROC for each class
    plt.figure(figsize=figsize)
    
    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true_binary[:, i], y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.2f})', linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.50)', linewidth=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()


def plot_learning_curves(train_losses, val_losses, train_accs, val_accs, title):
    """
    Plot learning curves (loss and accuracy).
    
    Parameters:
    -----------
    train_losses : list
        Training losses
    val_losses : list
        Validation losses
    train_accs : list
        Training accuracies
    val_accs : list
        Validation accuracies
    title : str
        Plot title
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curves
    ax1.plot(train_losses, label='Train Loss', linewidth=2)
    ax1.plot(val_losses, label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Learning Curves - Loss', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Accuracy curves
    ax2.plot(train_accs, label='Train Accuracy', linewidth=2)
    ax2.plot(val_accs, label='Val Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Learning Curves - Accuracy', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()


def create_metrics_comparison_table(models_results):
    """
    Create comparison table of model performance.
    
    Parameters:
    -----------
    models_results : dict
        Dictionary mapping model_name to metrics_dict
    
    Returns:
    --------
    pd.DataFrame : Comparison table
    """
    comparison_data = []
    
    for model_name, metrics in models_results.items():
        comparison_data.append({
            'Model': model_name,
            'Accuracy': f"{metrics['accuracy']:.4f}",
            'Precision (Macro)': f"{metrics['precision_macro']:.4f}",
            'Recall (Macro)': f"{metrics['recall_macro']:.4f}",
            'F1 (Macro)': f"{metrics['f1_macro']:.4f}",
            'Precision (Weighted)': f"{metrics.get('precision_weighted', float('nan')):.4f}",
            'Recall (Weighted)': f"{metrics.get('recall_weighted', float('nan')):.4f}",
            'F1 (Weighted)': f"{metrics['f1_weighted']:.4f}",
            'AUROC (Macro)': f"{metrics['roc_auc_macro']:.4f}",
            'AUPRC (Macro)': f"{metrics['auprc_macro']:.4f}"
        })
    
    df = pd.DataFrame(comparison_data)
    return df


def plot_feature_importance(model, feature_names, top_n=20, title="Feature Importance"):
    """
    Plot feature importance for tree-based models.
    
    Parameters:
    -----------
    model : sklearn model
        Fitted model with feature_importances_ attribute
    feature_names : list
        Feature names
    top_n : int, default=20
        Number of top features to show
    title : str
        Plot title
    """
    if not hasattr(model, 'feature_importances_'):
        print("Model does not have feature_importances_ attribute")
        return
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(top_n), importances[indices])
    plt.yticks(range(top_n), [feature_names[i] for i in indices])
    plt.xlabel('Importance', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()

