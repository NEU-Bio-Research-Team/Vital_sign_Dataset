"""
Main SHAP execution script for AKI Prediction project.

This script loads trained models from `best_models`,
loads and preprocesses test data, and generates SHAP explanations
for each trained model.
"""

import os
import sys
import joblib

# === Import from project modules ===
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from utils import load_vitaldb_data, preprocess_data, prepare_train_test_data
from train import load_model  # ✅ reuse your existing function
from shap_explainer import (
    explain_model_with_shap,
    analyze_logistic_regression_coefficients,
    plot_feature_importance_coefficients,
    save_shap_values
)


def load_all_models(best_models_dir="best_models"):
    """
    Load all trained models from the best_models directory
    using the `load_model()` function from train.py.
    """
    print(f"\n📦 Loading all models from '{best_models_dir}' ...")
    models_dict = {}

    if not os.path.exists(best_models_dir):
        print(f"❌ Directory not found: {best_models_dir}")
        return models_dict

    for file in os.listdir(best_models_dir):
        if file.endswith(".joblib"):
            model_name = file.replace(".joblib", "")
            model = load_model(model_name, best_models_dir)
            if model is not None:
                models_dict[model_name] = model

    print(f"📊 Total loaded models: {len(models_dict)}")
    return models_dict


def main_shap(model_dir="best_models"):
    """
    Run SHAP explanation for all models in model_dir.
    """
    print("\n🚀 Starting Explainable AI (SHAP) pipeline ...")
    print("=" * 70)

    # === 1️⃣ Load models ===
    models_dict = load_all_models(model_dir)
    if not models_dict:
        print("⚠️ No models found, exiting.")
        return

    # === 2️⃣ Load and preprocess data ===
    print("\n📦 Loading and preparing test data ...")
    df = load_vitaldb_data()
    X, y, feature_names = preprocess_data(df)
    data_dict = prepare_train_test_data(X, y)

    X_test_dict = data_dict["X_test_dict"]
    y_test = data_dict["y_test"]

    print(f"✅ Data prepared: {len(feature_names)} features, {len(y_test)} test samples")

    # === 3️⃣ Run SHAP explanations for each model ===
    for model_name, model in models_dict.items():
        print("\n" + "=" * 70)
        print(f"🏆 Running SHAP explanation for model: {model_name}")

        # Determine which test data version to use
        if "SVM" in model_name or "Logistic" in model_name:
            X_test_data = X_test_dict["scaled"]
        else:
            X_test_data = X_test_dict["imputed"]

        # Generate SHAP explanation
        explainer, shap_values = explain_model_with_shap(
            model=model,
            X_test_data=X_test_data,
            feature_names=feature_names,
            model_name=model_name,
            max_display=15
        )

        if shap_values is None:
            print(f"⚠️ Skipping {model_name} due to SHAP failure.")
            continue

        # === 4️⃣ If logistic regression → coefficient analysis ===
        if "Logistic" in model_name:
            analyze_logistic_regression_coefficients(model, feature_names, model_name)
            plot_feature_importance_coefficients(model, feature_names, model_name)

        # === 5️⃣ Save SHAP values ===
        save_shap_values(shap_values, model_name)

    print("\n🎯 All models explained successfully!")
    print("🏁 SHAP analysis pipeline completed.\n")


if __name__ == "__main__":
    main_shap()
