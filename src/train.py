"""
Training functions for AKI Prediction project.

This module contains functions for model training, hyperparameter tuning,
and model configuration.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import joblib
import os


def get_default_model_configs():
    """
    Get default model configurations for hyperparameter tuning.
    
    Returns:
    --------
    dict : Model configurations
    """
    
    configs = {
        'LogisticRegression': {
            'model': LogisticRegression(random_state=0),
            'params': {
                'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                'solver': ['lbfgs', 'liblinear', 'saga'],
                'max_iter': [1000, 2000, 5000],
                'class_weight': [None, 'balanced']
            },
            'data_type': 'scaled'
        },
        
        'RandomForest': {
            'model': RandomForestClassifier(random_state=0, n_jobs=-1),
            'params': {
                'n_estimators': [100, 200, 300, 500],
                'max_depth': [5, 10, 20, 30, None],
                'min_samples_split': [2, 5, 10, 20],
                'min_samples_leaf': [1, 2, 4, 8],
                'class_weight': [None, 'balanced', 'balanced_subsample']
            },
            'data_type': 'imputed'
        },
        
        'XGBoost': {
            'model': XGBClassifier(random_state=0, eval_metric='logloss'),
            'params': {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [3, 4, 6, 8],
                'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
                'subsample': [0.6, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0],
                'scale_pos_weight': [1, 10, 18, 20]
            },
            'data_type': 'imputed'
        },
        
        'SVM': {
            'model': SVC(random_state=0, probability=True),
            'params': {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto', 0.01, 0.1],
                'kernel': ['rbf', 'linear', 'poly'],
                'class_weight': ['balanced']
            },
            'data_type': 'scaled'
        }
    }
    
    return configs


def hyperparameter_tuning(models_config, X_train_dict, y_train, cv_folds=5, random_state=0, scoring='roc_auc', n_jobs=-1):
    """
    Modular hyperparameter tuning function for multiple models.
    
    Parameters:
    -----------
    models_config : dict
        Dictionary with model names as keys and configuration as values
        Format: {'ModelName': {'model': ModelClass(), 'params': param_grid, 'data_type': 'scaled' or 'imputed'}}
    X_train_dict : dict
        Dictionary with training data: {'scaled': X_train_scaled, 'imputed': X_train_imputed}
    y_train : array-like
        Training labels
    cv_folds : int, default=5
        Number of cross-validation folds
    random_state : int, default=0
        Random state for reproducibility
    scoring : str, default='roc_auc'
        Scoring metric for hyperparameter tuning
    n_jobs : int, default=-1
        Number of parallel jobs
    
    Returns:
    --------
    dict : Dictionary with tuned models
    """
    
    print("🔧 Starting Modular Hyperparameter Tuning...")
    print("=" * 70)
    
    tuned_models = {}
    cv_folds_obj = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    for model_name, config in models_config.items():
        print(f"\n🎯 Tuning {model_name}...")
        
        # Get model and parameters
        model = config['model']
        param_grid = config['params']
        data_type = config.get('data_type', 'imputed')  # Default to imputed
        
        # Select appropriate training data
        if data_type == 'scaled':
            X_train_data = X_train_dict['scaled']
        else:
            X_train_data = X_train_dict['imputed']
        
        # Perform GridSearchCV
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=cv_folds_obj,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=0
        )
        
        # Fit the model
        grid_search.fit(X_train_data, y_train)
        
        # Store the best model
        tuned_models[model_name] = grid_search.best_estimator_
        
        print(f"   ✅ Best parameters: {grid_search.best_params_}")
        print(f"   📊 Best CV score: {grid_search.best_score_:.4f}")
    
    print(f"\n🎉 Hyperparameter tuning completed for {len(tuned_models)} models!")
    return tuned_models


def train_single_model(model, model_name, X_train_data, y_train, params=None):
    """
    Train a single model with optional hyperparameters.
    
    Parameters:
    -----------
    model : sklearn model
        Model to train
    model_name : str
        Name of the model
    X_train_data : array-like
        Training features
    y_train : array-like
        Training labels
    params : dict, optional
        Hyperparameters to set
    
    Returns:
    --------
    trained model
    """
    print(f"🎯 Training {model_name}...")
    
    if params:
        model.set_params(**params)
    
    model.fit(X_train_data, y_train)
    print(f"   ✅ {model_name} training completed")
    
    return model


def save_model(model, model_name, best_models_dir='best_models'):
    """
    Save a trained model to disk.
    
    Parameters:
    -----------
    model : trained model
        Model to save
    model_name : str
        Name of the model
    best_models_dir : str, default='best_models'
        Directory to save the model
    """
    # Create directory if it doesn't exist
    os.makedirs(best_models_dir, exist_ok=True)
    
    # Save model
    filename = f"{best_models_dir}/{model_name}.joblib"
    joblib.dump(model, filename)
    print(f"✅ Model saved to: {filename}")


def load_model(model_name, best_models_dir='best_models'):
    """
    Load a trained model from disk.
    
    Parameters:
    -----------
    model_name : str
        Name of the model
    best_models_dir : str, default='best_models'
        Directory containing the model
    
    Returns:
    --------
    trained model or None if not found
    """
    filename = f"{best_models_dir}/{model_name}.joblib"
    try:
        model = joblib.load(filename)
        print(f"✅ Model loaded from: {filename}")
        return model
    except FileNotFoundError:
        print(f"❌ Model file not found: {filename}")
        return None


def save_best_model(models_dict, X_test_dict, y_test, model_data_mapping=None, best_models_dir='best_models'):
    """
    Find and save the best performing model.
    
    Parameters:
    -----------
    models_dict : dict
        Dictionary with trained models
    X_test_dict : dict
        Dictionary with test data
    y_test : array-like
        True test labels
    model_data_mapping : dict, optional
        Dictionary mapping model names to data type
    best_models_dir : str, default='best_models'
        Directory to save the best model
    """
    
    # Import here to avoid circular import
    from .evaluate import evaluate_models
    
    # Evaluate all models to find the best one
    results_df = evaluate_models(models_dict, X_test_dict, y_test, model_data_mapping)
    best_model_name = results_df.iloc[0]['Model']
    best_model = models_dict[best_model_name]
    
    print(f"🏆 Best model: {best_model_name} (ROC-AUC: {results_df.iloc[0]['ROC-AUC']:.4f})")
    
    # Save the best model
    save_model(best_model, f"{best_model_name}_best", best_models_dir)
    
    # Save evaluation results
    results_df.to_csv(f"{best_models_dir}/best_model_results.csv", index=False)
    
    return best_model_name, best_model