"""
Evaluation functions for AKI Prediction project (Enhanced Version).

This module includes:
- Comprehensive evaluation metrics
- Threshold optimization for imbalanced datasets
- Model performance comparison and saving utilities
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_curve, roc_auc_score, precision_recall_curve, auc,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)


def calculate_comprehensive_metrics(
    y_true, 
    y_pred, 
    y_pred_proba, 
    model_name="Model", 
    optimize_threshold=False, 
    metric_for_threshold="f1"
):
    """
    Calculate comprehensive evaluation metrics with optional threshold optimization.

    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_pred : array-like
        Predicted binary labels (based on default threshold 0.5)
    y_pred_proba : array-like
        Predicted probabilities for positive class
    model_name : str
        Name of the model
    optimize_threshold : bool, default=False
        Whether to find the optimal probability threshold
    metric_for_threshold : {'f1', 'recall'}, default='f1'
        Metric to optimize threshold for ('f1' or 'recall')

    Returns:
    --------
    dict : Dictionary containing all calculated metrics
    """

    # Default evaluation (threshold = 0.5)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    ppv = precision  # same as precision

    # AUC metrics
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
    auprc = auc(recalls, precisions)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0

    # Base metrics
    metrics = {
        "Model": model_name,
        "Threshold": 0.5,
        "Accuracy": accuracy,
        "Precision": precision,
        "PPV": ppv,
        "Recall": recall,
        "Specificity": specificity,
        "NPV": npv,
        "F1-Score": f1,
        "ROC-AUC": roc_auc,
        "AUPRC": auprc,
        "True Positives": tp,
        "False Positives": fp,
        "True Negatives": tn,
        "False Negatives": fn,
    }

    # ---------- Threshold Optimization ----------
    if optimize_threshold:
        best_thresh = 0.5
        best_score = f1  # base for comparison

        # Avoid zero division
        eps = 1e-10

        if metric_for_threshold == "f1":
            f1_scores = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + eps)
            idx = np.argmax(f1_scores)
            best_thresh, best_score = thresholds[idx], f1_scores[idx]

        elif metric_for_threshold == "recall":
            # Choose threshold giving recall >= target (e.g. 0.8) with max precision
            target_recall = 0.8
            valid_idx = np.where(recalls >= target_recall)[0]
            if len(valid_idx) > 0:
                idx = valid_idx[np.argmax(precisions[valid_idx])]
                best_thresh = thresholds[idx]
                best_score = recalls[idx]
            else:
                best_thresh = 0.5

        # Apply optimal threshold
        y_pred_opt = (y_pred_proba >= best_thresh).astype(int)

        # Recalculate metrics with optimal threshold
        accuracy = accuracy_score(y_true, y_pred_opt)
        precision = precision_score(y_true, y_pred_opt, zero_division=0)
        recall = recall_score(y_true, y_pred_opt, zero_division=0)
        f1 = f1_score(y_true, y_pred_opt, zero_division=0)
        cm = confusion_matrix(y_true, y_pred_opt)
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0

        # Update results
        metrics.update({
            "Threshold": best_thresh,
            "F1-Score (Optimized)": f1,
            "Recall (Optimized)": recall,
            "Precision (Optimized)": precision,
            "Accuracy (Optimized)": accuracy,
            "Specificity (Optimized)": specificity,
            "NPV (Optimized)": npv,
            "True Positives (Opt)": tp,
            "False Positives (Opt)": fp,
            "True Negatives (Opt)": tn,
            "False Negatives (Opt)": fn,
        })

    return metrics


def evaluate_models(models_dict, X_test_dict, y_test, model_data_mapping=None, optimize_threshold=False):
    """
    Evaluate multiple models with optional threshold optimization.
    """
    print("📊 Starting Comprehensive Model Evaluation...")
    print("=" * 70)
    results = []

    for model_name, model in models_dict.items():
        print(f"\n🔍 Evaluating {model_name}...")

        if model_data_mapping and model_name in model_data_mapping:
            data_type = model_data_mapping[model_name]
        else:
            data_type = "scaled" if "SVM" in model_name or "Logistic" in model_name else "imputed"

        X_test_data = X_test_dict[data_type]

        try:
            y_pred = model.predict(X_test_data)
            y_pred_proba = model.predict_proba(X_test_data)[:, 1]

            metrics = calculate_comprehensive_metrics(
                y_test, y_pred, y_pred_proba, model_name, optimize_threshold=optimize_threshold
            )
            results.append(metrics)

            if optimize_threshold:
                print(f"   ⚙️  Best Threshold: {metrics['Threshold']:.3f}")
                print(f"   ✅ F1 (Optimized): {metrics['F1-Score (Optimized)']:.4f}")
            print(f"   ✅ ROC-AUC: {metrics['ROC-AUC']:.4f}")

        except Exception as e:
            print(f"   ❌ Error evaluating {model_name}: {e}")
            continue

    results_df = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
    print(f"\n🎉 Evaluation completed for {len(results_df)} models!")
    print(f"🏆 Best model: {results_df.iloc[0]['Model']} (ROC-AUC: {results_df.iloc[0]['ROC-AUC']:.4f})")
    return results_df

def evaluate_single_model(model, model_name, X_test_data, y_test, optimize_threshold=False):
    """
    Evaluate a single model and optionally optimize threshold.
    """
    print(f"🔍 Evaluating {model_name}...")

    # Predict labels and probabilities
    y_pred = model.predict(X_test_data)
    y_pred_proba = model.predict_proba(X_test_data)[:, 1]

    # Compute metrics
    metrics = calculate_comprehensive_metrics(
        y_test, 
        y_pred, 
        y_pred_proba, 
        model_name, 
        optimize_threshold=optimize_threshold
    )

    # Display summary
    print(f"   ✅ ROC-AUC: {metrics['ROC-AUC']:.4f}")
    if optimize_threshold:
        print(f"   ⚙️  Optimal Threshold: {metrics['Threshold']:.3f}")
        print(f"   ✅ F1 (Optimized): {metrics['F1-Score (Optimized)']:.4f}")

    return y_pred, y_pred_proba, metrics


def print_evaluation_summary(results_df):
    """
    Print summary of evaluation results for multiple models.
    """
    threshold_key = "Threshold" if "Threshold" in results_df.columns else "Optimal Threshold"

    print("\n📈 Evaluation Results Summary:")
    print(f"{'Model':<20} {'ROC-AUC':<10} {'AUPRC':<10} {'F1-Score':<10} {'PPV':<10} {threshold_key:<10}")
    print("-" * 80)

    for _, row in results_df.iterrows():
        print(f"{row['Model']:<20} "
              f"{row['ROC-AUC']:<10.4f} "
              f"{row['AUPRC']:<10.4f} "
              f"{row['F1-Score']:<10.4f} "
              f"{row['PPV']:<10.4f} "
              f"{row[threshold_key]:<10.3f}")

    print("\n🏆 FINAL RANKINGS:")
    print(f"{'Rank':<5} {'Model':<20} {'ROC-AUC':<10} {'F1-Score':<10} {threshold_key:<10}")
    print("-" * 70)

    for i, (_, row) in enumerate(results_df.iterrows(), 1):
        print(f"{i:<5} "
              f"{row['Model']:<20} "
              f"{row['ROC-AUC']:<10.4f} "
              f"{row['F1-Score']:<10.4f} "
              f"{row[threshold_key]:<10.3f}")


def save_evaluation_results(results_df, filename='evaluation_results.csv'):
    results_df.to_csv(filename, index=False)
    print(f"✅ Evaluation results saved to: {filename}")


def load_evaluation_results(filename='evaluation_results.csv'):
    try:
        results_df = pd.read_csv(filename)
        print(f"✅ Evaluation results loaded from: {filename}")
        return results_df
    except FileNotFoundError:
        print(f"❌ Results file not found: {filename}")
        return None