"""
Main visualization module for AKI Prediction Project.

This script loads trained models, test data, and evaluation results,
then visualizes ROC curves, PR curves, confusion matrices, and metric comparisons.
"""

import os
import pandas as pd
import sys
import joblib
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils import load_vitaldb_data, preprocess_data, prepare_train_test_data
from evaluate import evaluate_models, load_evaluation_results
from visualization import (
    plot_roc_curves,
    plot_pr_curves,
    plot_model_comparison,
    plot_confusion_matrices
)


def main_visualize(model_dir="best_models", eval_csv="evaluation_outputs/evaluation_summary.csv"):
    print("🚀 Starting Visualization Pipeline...")
    print("=" * 70)

    # === Load data ===
    print("📦 Loading and preparing test data...")
    df = load_vitaldb_data()
    X, y, feature_names = preprocess_data(df)
    data_dict = prepare_train_test_data(X, y)
    X_test_dict = data_dict["X_test_dict"]
    y_test = data_dict["y_test"]

    # === Load trained models ===
    print(f"📂 Loading models from {model_dir} ...")
    models_dict = {}
    if not os.path.exists(model_dir):
        print(f"❌ Directory not found: {model_dir}")
    else:
        for file in os.listdir(model_dir):
            if file.endswith('.joblib'):
                model_name = file.replace('.joblib', '')
                path = os.path.join(model_dir, file)
                try:
                    model = joblib.load(path)
                    models_dict[model_name] = model
                    print(f"✅ Loaded model: {model_name}")
                except Exception as e:
                    print(f"❌ Failed to load {model_name}: {e}")

    if not models_dict:
        print("⚠️ No models were loaded. Please check your 'best_models' directory.")
        return

    print(f"✅ Successfully loaded {len(models_dict)} models: {list(models_dict.keys())}")

    # === Load or compute evaluation results ===
    print("\n📊 Loading or computing evaluation results...")
    if os.path.exists(eval_csv):
        print(f"📂 Found existing evaluation file: {eval_csv}")
        results_df = load_evaluation_results(eval_csv)
    else:
        print("⚙️ No evaluation file found — computing fresh evaluation...")
        results_df = evaluate_models(models_dict, X_test_dict, y_test)
        os.makedirs(os.path.dirname(eval_csv), exist_ok=True)
        results_df.to_csv(eval_csv, index=False)
        print(f"💾 Saved new evaluation summary to {eval_csv}")

    # === Visualization ===
    print("\n🎨 Generating visualizations...")

    # 1️⃣ ROC Curves
    print("📈 Plotting ROC Curves...")
    plot_roc_curves(models_dict, X_test_dict, y_test)

    # 2️⃣ Precision-Recall Curves
    print("📉 Plotting Precision-Recall Curves...")
    plot_pr_curves(models_dict, X_test_dict, y_test)

    # 3️⃣ Confusion Matrices
    print("🧩 Plotting Confusion Matrices...")
    plot_confusion_matrices(models_dict, X_test_dict, y_test)

    # 4️⃣ Model Metric Comparison
    if results_df is not None and not results_df.empty:
        print("📊 Plotting Metric Comparison...")
        plot_model_comparison(
            results_df,
            metrics=["ROC-AUC", "AUPRC", "Precision", "Recall", "F1-Score"]
        )
    else:
        print("⚠️ No results_df found or it's empty — skipping metric comparison plot.")

    print("\n🏁 Visualization pipeline completed successfully!")


if __name__ == "__main__":
    main_visualize()
