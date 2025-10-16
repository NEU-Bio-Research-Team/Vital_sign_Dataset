#!/usr/bin/env python3
"""
Example script showing different ways to train specific models instead of all models.

This script demonstrates various approaches for training only selected models
instead of using the complete default configuration.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils import setup_plotting, load_vitaldb_data, preprocess_data, prepare_train_test_data
from train import get_default_model_configs, hyperparameter_tuning
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

def example_1_select_from_default():
    """Example 1: Select specific models from default configurations."""
    print("🎯 Example 1: Select specific models from default configurations")
    print("=" * 60)
    
    # Load and preprocess data
    df = load_vitaldb_data()
    X, y, feature_names = preprocess_data(df)
    data_dict = prepare_train_test_data(X, y)
    
    # Get all default configurations
    all_configs = get_default_model_configs()
    
    # Select only specific models
    selected_models = {
        'LogisticRegression': all_configs['LogisticRegression'],
        'XGBoost': all_configs['XGBoost']
    }
    
    print(f"Selected models: {list(selected_models.keys())}")
    
    # Train only selected models
    tuned_models = hyperparameter_tuning(
        selected_models, 
        data_dict['X_train_dict'], 
        data_dict['y_train']
    )
    
    return tuned_models, data_dict, feature_names

def example_2_custom_parameters():
    """Example 2: Create custom model configurations with different parameters."""
    print("\n🎯 Example 2: Custom model configurations with different parameters")
    print("=" * 60)
    
    # Load and preprocess data
    df = load_vitaldb_data()
    X, y, feature_names = preprocess_data(df)
    data_dict = prepare_train_test_data(X, y)
    
    # Create custom configurations with simplified parameter grids for faster training
    custom_models = {
        'LogisticRegression_Fast': {
            'model': LogisticRegression(random_state=0),
            'params': {
                'C': [0.1, 1, 10],  # Reduced parameter grid
                'solver': ['lbfgs'],  # Only one solver
                'class_weight': [None, 'balanced']
            },
            'data_type': 'scaled'
        },
        'XGBoost_Fast': {
            'model': XGBClassifier(random_state=0, eval_metric='logloss'),
            'params': {
                'n_estimators': [50, 100],  # Reduced parameter grid
                'max_depth': [3, 6],
                'learning_rate': [0.1, 0.2],
                'scale_pos_weight': [1, 18]
            },
            'data_type': 'imputed'
        }
    }
    
    print(f"Custom models: {list(custom_models.keys())}")
    
    # Train custom models
    tuned_models = hyperparameter_tuning(
        custom_models, 
        data_dict['X_train_dict'], 
        data_dict['y_train']
    )
    
    return tuned_models, data_dict, feature_names

def example_3_single_model():
    """Example 3: Train only a single model."""
    print("\n🎯 Example 3: Train only a single model")
    print("=" * 60)
    
    # Load and preprocess data
    df = load_vitaldb_data()
    X, y, feature_names = preprocess_data(df)
    data_dict = prepare_train_test_data(X, y)
    
    # Train only XGBoost
    single_model_config = {
        'XGBoost_Only': {
            'model': XGBClassifier(random_state=0, eval_metric='logloss'),
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.05, 0.1, 0.2],
                'scale_pos_weight': [1, 18, 20]
            },
            'data_type': 'imputed'
        }
    }
    
    print(f"Single model: {list(single_model_config.keys())}")
    
    # Train single model
    tuned_models = hyperparameter_tuning(
        single_model_config, 
        data_dict['X_train_dict'], 
        data_dict['y_train']
    )
    
    return tuned_models, data_dict, feature_names

def example_4_conditional_training():
    """Example 4: Conditional training based on available resources."""
    print("\n🎯 Example 4: Conditional training based on available resources")
    print("=" * 60)
    
    # Load and preprocess data
    df = load_vitaldb_data()
    X, y, feature_names = preprocess_data(df)
    data_dict = prepare_train_test_data(X, y)
    
    # Get all default configurations
    all_configs = get_default_model_configs()
    
    # Define different training scenarios
    scenarios = {
        'fast': ['LogisticRegression'],  # Quick training
        'balanced': ['LogisticRegression', 'XGBoost'],  # Balanced speed/performance
        'comprehensive': ['LogisticRegression', 'RandomForest', 'XGBoost', 'SVM']  # Full training
    }
    
    # Choose scenario (you can modify this)
    selected_scenario = 'balanced'  # Change this to 'fast', 'balanced', or 'comprehensive'
    
    print(f"Selected scenario: {selected_scenario}")
    
    # Build model configuration based on scenario
    selected_models = {}
    for model_name in scenarios[selected_scenario]:
        if model_name in all_configs:
            selected_models[model_name] = all_configs[model_name]
    
    print(f"Models to train: {list(selected_models.keys())}")
    
    # Train selected models
    tuned_models = hyperparameter_tuning(
        selected_models, 
        data_dict['X_train_dict'], 
        data_dict['y_train']
    )
    
    return tuned_models, data_dict, feature_names

def main():
    """Main function to demonstrate all examples."""
    setup_plotting()
    
    print("🚀 AKI Prediction - Training Specific Models Examples")
    print("=" * 70)
    
    # Run examples (uncomment the one you want to test)
    
    # Example 1: Select from default
    # tuned_models, data_dict, feature_names = example_1_select_from_default()
    
    # Example 2: Custom parameters
    # tuned_models, data_dict, feature_names = example_2_custom_parameters()
    
    # Example 3: Single model
    # tuned_models, data_dict, feature_names = example_3_single_model()
    
    # Example 4: Conditional training
    tuned_models, data_dict, feature_names = example_4_conditional_training()
    
    print(f"\n✅ Training completed!")
    print(f"Trained models: {list(tuned_models.keys())}")
    
    # Auto-generate model data mapping
    model_data_mapping = {}
    for model_name in tuned_models.keys():
        if 'Logistic' in model_name or 'SVM' in model_name:
            model_data_mapping[model_name] = 'scaled'
        else:
            model_data_mapping[model_name] = 'imputed'
    
    print(f"Model data mapping: {model_data_mapping}")
    
    return tuned_models, data_dict, feature_names, model_data_mapping

if __name__ == "__main__":
    tuned_models, data_dict, feature_names, model_data_mapping = main()
