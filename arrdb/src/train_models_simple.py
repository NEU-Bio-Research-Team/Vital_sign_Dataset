"""
Simplified training utilities without PyTorch dependencies for basic ML experiments.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_score

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_class_weight


def create_ml_models(task_type='rhythm', num_classes=11, class_weights=None, random_state=42):
    """
    Create traditional ML models for classification.
    """
    models = {}
    
    # Note: XGBoost will auto-encode string labels
    # But we need to not pre-specify num_class to allow this
    # Use more aggressive regularization to prevent overfitting
    if task_type == 'rhythm':
        models['XGBoost'] = XGBClassifier(
            objective='multi:softmax',
            n_estimators=200,  # Reduced from 300
            max_depth=3,  # Reduced from 4
            min_child_weight=7,  # Increased from 5
            learning_rate=0.03,  # Reduced from 0.05
            subsample=0.7,  # Reduced from 0.8
            colsample_bytree=0.7,  # Reduced from 0.8
            reg_alpha=0.1,  # L1 regularization
            reg_lambda=1.0,  # L2 regularization
            random_state=random_state,
            eval_metric='mlogloss',
            n_jobs=-1,
            enable_categorical=True
        )
    else:  # beat
        models['XGBoost'] = XGBClassifier(
            objective='multi:softmax',
            n_estimators=200,
            max_depth=3,
            min_child_weight=7,
            learning_rate=0.03,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=random_state,
            eval_metric='mlogloss',
            n_jobs=-1,
            enable_categorical=True
        )
    
    models['RandomForest'] = RandomForestClassifier(
        n_estimators=200,  # Reduced from 300
        max_depth=6,  # Reduced from 8
        min_samples_split=10,  # Increased from 5
        min_samples_leaf=5,  # Increased from 3
        max_features='sqrt',  # Limit features per split
        class_weight='balanced',
        random_state=random_state,
        n_jobs=-1
    )
    
    models['LogisticRegression'] = LogisticRegression(
        multi_class='ovr',
        solver='lbfgs',
        max_iter=1000,
        C=0.1,  # Stronger regularization (reduced from 0.5)
        class_weight='balanced',
        random_state=random_state,
        n_jobs=-1
    )
    
    return models


def train_traditional_ml(models_dict, X_train, y_train, X_val, y_val, task_name, calibrate=True):
    """
    Train traditional ML models.
    """
    results = {}
    
    print(f"\n{'='*60}")
    print(f"Training {task_name} Classification Models")
    print(f"{'='*60}\n")
    
    # Encode string labels to integers for XGBoost and uniform handling
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    
    y_train_encoded = y_train.copy()
    y_val_encoded = y_val.copy()
    
    if y_train.dtype == 'object':
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_val_encoded = label_encoder.transform(y_val)
        
        # Update XGBoost models with correct num_class
        num_classes = len(label_encoder.classes_)
        if 'XGBoost' in models_dict:
            models_dict['XGBoost'].set_params(num_class=num_classes)
        
        print(f"Encoded labels: {dict(zip(range(len(label_encoder.classes_)), label_encoder.classes_))}\n")
    
    # Compute sample weights (balanced) to reinforce minority classes
    classes_for_weights = np.unique(y_train_encoded)
    class_weights_arr = compute_class_weight('balanced', classes=classes_for_weights, y=y_train_encoded)
    class_to_weight = {c: w for c, w in zip(classes_for_weights, class_weights_arr)}
    sample_weight_train = np.array([class_to_weight[c] for c in y_train_encoded])
    sample_weight_val = np.array([class_to_weight[c] for c in y_val_encoded])

    for model_name, model in models_dict.items():
        print(f"Training {model_name}...")
        
        # Train model with encoded labels and sample weights if supported
        try:
            model.fit(X_train, y_train_encoded, sample_weight=sample_weight_train)
        except TypeError:
            model.fit(X_train, y_train_encoded)
        
        # Optional probability calibration using validation set
        calibrated_model = model
        if calibrate:
            try:
                calibrated = CalibratedClassifierCV(model, cv='prefit', method='isotonic')
                calibrated.fit(X_val, y_val_encoded, sample_weight=sample_weight_val)
                calibrated_model = calibrated
            except Exception:
                # Fallback to sigmoid if isotonic fails (e.g., not enough samples per class)
                try:
                    calibrated = CalibratedClassifierCV(model, cv='prefit', method='sigmoid')
                    calibrated.fit(X_val, y_val_encoded, sample_weight=sample_weight_val)
                    calibrated_model = calibrated
                except Exception:
                    calibrated_model = model
        
        # Evaluate on validation set using calibrated model
        train_acc = calibrated_model.score(X_train, y_train_encoded)
        val_acc = calibrated_model.score(X_val, y_val_encoded)
        
        # Get predictions
        y_train_pred = calibrated_model.predict(X_train)
        y_val_pred = calibrated_model.predict(X_val)
        
        # Decode predictions back to original labels
        if 'label_encoder' in locals():
            y_train_pred = label_encoder.inverse_transform(y_train_pred)
            y_val_pred = label_encoder.inverse_transform(y_val_pred)
        
        y_train_proba = calibrated_model.predict_proba(X_train)
        y_val_proba = calibrated_model.predict_proba(X_val)
        
        results[model_name] = {
            'model': calibrated_model,
            'label_encoder': label_encoder if 'label_encoder' in locals() else None,
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'y_train_pred': y_train_pred,
            'y_val_pred': y_val_pred,
            'y_train_proba': y_train_proba,
            'y_val_proba': y_val_proba
        }
        
        print(f"  Train Accuracy: {train_acc:.4f}")
        print(f"  Val Accuracy: {val_acc:.4f}\n")
    
    return results


def evaluate_models_on_test(trained_models, X_test, y_test):
    """
    Evaluate trained models on test set.
    """
    print("\nEvaluating on test set...")
    
    for model_name, result in trained_models.items():
        model = result['model']
        
        # Encode test labels if encoder exists
        y_test_encoded = y_test.copy()
        if result.get('label_encoder') is not None:
            label_encoder = result['label_encoder']
            
            # Handle unseen labels in test set
            y_test_list = y_test.tolist() if hasattr(y_test, 'tolist') else list(y_test)
            seen_classes = set(label_encoder.classes_)
            
            # Map unseen labels to the most common seen class (first class)
            default_class = label_encoder.classes_[0]
            y_test_mapped = []
            
            unseen_found = []
            for label in y_test_list:
                if label in seen_classes:
                    y_test_mapped.append(label)
                else:
                    y_test_mapped.append(default_class)
                    if label not in unseen_found:
                        unseen_found.append(label)
            
            if unseen_found:
                print(f"  Warning: Unseen labels in test set: {unseen_found}. Mapping to '{default_class}'")
            
            y_test_encoded = label_encoder.transform(y_test_mapped)
        
        test_acc = model.score(X_test, y_test_encoded)
        y_test_pred = model.predict(X_test)
        y_test_proba = model.predict_proba(X_test)
        
        # Decode predictions
        if result.get('label_encoder') is not None:
            label_encoder = result['label_encoder']
            y_test_pred = label_encoder.inverse_transform(y_test_pred)
        
        result['test_accuracy'] = test_acc
        result['y_test_pred'] = y_test_pred
        result['y_test_proba'] = y_test_proba
        
        print(f"{model_name} Test Accuracy: {test_acc:.4f}")
    
    return trained_models


def save_ml_models(trained_models, task_name, save_dir='experiments/saved_models'):
    """
    Save trained traditional ML models.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    for model_name, result in trained_models.items():
        model = result['model']
        filename = save_path / f"{task_name}_{model_name}.pkl"
        joblib.dump(model, filename)
        print(f"✅ Saved {filename}")

