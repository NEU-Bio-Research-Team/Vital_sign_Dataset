"""
Main Evaluation Script for AKI Prediction Project

This script:
1. Loads trained models from 'best_models/' directory
2. Loads and preprocesses test data
3. Evaluates each model using multiple metrics (ROC-AUC, AUPRC, F1, etc.)
4. Displays summary and saves results to CSV
"""

import os
import joblib
import sys 
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils import load_vitaldb_data, preprocess_data, prepare_train_test_data
from evaluate import (
    evaluate_models,
    print_evaluation_summary,
    save_evaluation_results,
    load_evaluation_results
)


def load_all_models(best_models_dir='best_models'):
    """
    Load all trained models from the specified directory.
    
    Parameters:
    -----------
    best_models_dir : str, default='best_models'
        Directory containing saved .joblib models
    
    Returns:
    --------
    dict : {model_name: loaded_model}
    """
    models = {}
    if not os.path.exists(best_models_dir):
        print(f"❌ Directory not found: {best_models_dir}")
        return models

    for file in os.listdir(best_models_dir):
        if file.endswith('.joblib'):
            model_name = file.replace('.joblib', '')
            path = os.path.join(best_models_dir, file)
            try:
                model = joblib.load(path)
                models[model_name] = model
                print(f"✅ Loaded model: {model_name}")
            except Exception as e:
                print(f"❌ Failed to load {model_name}: {e}")
    return models


def main_evaluate():
    print("🚀 Starting Model Evaluation Pipeline...")
    print("=" * 80)

    # === 1. Load and preprocess data ===
    df = load_vitaldb_data()
    X, y, feature_names = preprocess_data(df)
    data_dict = prepare_train_test_data(X, y)

    X_test_dict = data_dict['X_test_dict']
    y_test = data_dict['y_test']

    # === 2. Load all trained models ===
    models_dict = load_all_models(best_models_dir='best_models')
    if not models_dict:
        print("⚠️ No models found. Please run train_main.py first.")
        return

    # === 3. Evaluate all models ===
    results_df = evaluate_models(models_dict, X_test_dict, y_test, optimize_threshold=True)

    # === 4. Display summary ===
    print_evaluation_summary(results_df)

    # === 5. Save results ===
    save_evaluation_results(results_df, filename='evaluation_results.csv')

    print("\n✅ Evaluation completed and saved.")
    print("=" * 80)

    return results_df


if __name__ == "__main__":
    results_df = main_evaluate()
