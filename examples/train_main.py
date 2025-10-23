"""
Main Training Script for AKI Prediction Project

This script runs the full training pipeline:
1. Load and preprocess the VitalDB dataset
2. Prepare train/test splits
3. Perform hyperparameter tuning on multiple models
4. Save the best trained models to disk
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils import load_vitaldb_data, preprocess_data, prepare_train_test_data
from train import (
    get_default_model_configs,
    hyperparameter_tuning,
    train_single_model,
    save_model
)


def main():
    print("🚀 Starting AKI Prediction Training Pipeline...")
    print("=" * 80)

    # === 1. Load & preprocess data ===
    df = load_vitaldb_data()
    X, y, feature_names = preprocess_data(df)
    data_dict = prepare_train_test_data(X, y)
    X_train_dict = data_dict['X_train_dict']
    y_train = data_dict['y_train']

    # === 2. Get default model configurations ===
    model_configs = get_default_model_configs()
    print(f"\n📋 Models available for training: {list(model_configs.keys())}")

    # === 3. Hyperparameter tuning ===
    tuned_models = hyperparameter_tuning(
        model_configs,
        X_train_dict,
        y_train,
        cv_folds=3,  # để giảm thời gian chạy ban đầu
        scoring='roc_auc'
    )

    # === 4. Save tuned models ===
    os.makedirs("best_models", exist_ok=True)
    for model_name, model in tuned_models.items():
        save_model(model, model_name, best_models_dir="best_models")

    print("\n✅ All models trained and saved successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
