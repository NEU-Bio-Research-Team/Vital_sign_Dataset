"""
Training utilities for Arrhythmia Classification models.

Functions for training traditional ML models (XGBoost, Random Forest, Logistic Regression)
and deep learning models (1D-CNN, LSTM) for both beat-level and rhythm-level classification.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Deep learning features will be disabled.")

try:
    from models import LightweightCNN1D, LightweightLSTM
except ImportError:
    LightweightCNN1D = None
    LightweightLSTM = None


def create_ml_models(task_type='rhythm', num_classes=11, class_weights=None, random_state=42):
    """
    Create traditional ML models for classification.
    
    Parameters:
    -----------
    task_type : str, default='rhythm'
        'beat' (4 classes) or 'rhythm' (11 classes)
    num_classes : int, default=11
        Number of classes
    class_weights : dict, optional
        Dictionary of class weights
    random_state : int, default=42
    
    Returns:
    --------
    dict : Dictionary of model configurations
    
    models = {}
    
    if task_type == 'rhythm':
        # Multi-class objective for XGBoost
        models['XGBoost'] = XGBClassifier(
            objective='multi:softmax',
            num_class=num_classes,
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            eval_metric='mlogloss',
            n_jobs=-1,
            enable_categorical=True  # Handle string labels
        )
    else:  # beat
        models['XGBoost'] = XGBClassifier(
            objective='multi:softmax',
            num_class=num_classes,
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            eval_metric='mlogloss',
            n_jobs=-1,
            enable_categorical=True  # Handle string labels
        )
    
    models['RandomForest'] = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=random_state,
        n_jobs=-1
    )
    
    models['LogisticRegression'] = LogisticRegression(
        multi_class='ovr',
        solver='lbfgs',
        max_iter=1000,
        C=1.0,
        class_weight='balanced',
        random_state=random_state,
        n_jobs=-1
    )
    
    return models


def train_traditional_ml(models_dict, X_train, y_train, X_val, y_val, task_name):
    """
    Train traditional ML models.
    
    Parameters:
    -----------
    models_dict : dict
        Dictionary of models to train
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training labels
    X_val : pd.DataFrame
        Validation features
    y_val : pd.Series
        Validation labels
    task_name : str
        Name of task (for saving models)
    
    Returns:
    --------
    dict : Dictionary of trained models with performance metrics
    """
    results = {}
    
    print(f"\n{'='*60}")
    print(f"Training {task_name} Classification Models")
    print(f"{'='*60}\n")
    
    for model_name, model in models_dict.items():
        print(f"Training {model_name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Evaluate on validation set
        train_acc = model.score(X_train, y_train)
        val_acc = model.score(X_val, y_val)
        
        # Get predictions
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        
        y_train_proba = model.predict_proba(X_train)
        y_val_proba = model.predict_proba(X_val)
        
        results[model_name] = {
            'model': model,
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
    
    Parameters:
    -----------
    trained_models : dict
        Dictionary of trained models (output from train_traditional_ml)
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test labels
    
    Returns:
    --------
    dict : Updated models dict with test predictions
    """
    print("\nEvaluating on test set...")
    
    for model_name, result in trained_models.items():
        model = result['model']
        
        test_acc = model.score(X_test, y_test)
        y_test_pred = model.predict(X_test)
        y_test_proba = model.predict_proba(X_test)
        
        result['test_accuracy'] = test_acc
        result['y_test_pred'] = y_test_pred
        result['y_test_proba'] = y_test_proba
        
        print(f"{model_name} Test Accuracy: {test_acc:.4f}")
    
    return trained_models


def save_ml_models(trained_models, task_name, save_dir='experiments/saved_models'):
    """
    Save trained traditional ML models.
    
    Parameters:
    -----------
    trained_models : dict
        Dictionary of trained models
    task_name : str
        Task name for file naming
    save_dir : str, default='experiments/saved_models'
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    for model_name, result in trained_models.items():
        model = result['model']
        filename = save_path / f"{task_name}_{model_name}.pkl"
        joblib.dump(model, filename)
        print(f"✅ Saved {filename}")


def create_dataloaders(X_train, y_train, X_val, y_val, batch_size=32):
    """
    Create PyTorch DataLoaders from sequences.
    
    Parameters:
    -----------
    X_train : np.ndarray
        Training sequences (n_samples, seq_length)
    y_train : np.ndarray
        Training labels
    X_val : np.ndarray
        Validation sequences
    y_val : np.ndarray
        Validation labels
    batch_size : int, default=32
    
    Returns:
    --------
    tuple : (train_loader, val_loader)
    """
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train).unsqueeze(1)  # Add channel dimension
    y_train_tensor = torch.LongTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val).unsqueeze(1)
    y_val_tensor = torch.LongTensor(y_val)
    
    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


def train_deep_learning(model, train_loader, val_loader, num_epochs=50, 
                        learning_rate=0.001, device='cuda', patience=10):
    """
    Train deep learning model with early stopping.
    
    Parameters:
    -----------
    model : nn.Module
        PyTorch model
    train_loader : DataLoader
        Training data loader
    val_loader : DataLoader
        Validation data loader
    num_epochs : int, default=50
        Maximum number of epochs
    learning_rate : float, default=0.001
        Learning rate
    device : str, default='cuda'
        Device to use ('cuda' or 'cpu')
    patience : int, default=10
        Early stopping patience
    
    Returns:
    --------
    dict : Training results with history and best model
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0
    
    print(f"\n{'='*60}")
    print(f"Training Deep Learning Model on {device}")
    print(f"{'='*60}\n")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()
        
        train_accuracy = 100 * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()
        
        val_accuracy = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%")
            print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
        
        # Early stopping
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    results = {
        'model': model,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'best_val_acc': best_val_acc
    }
    
    print(f"\n✅ Training complete. Best Val Accuracy: {best_val_acc:.2f}%\n")
    
    return results


def evaluate_dl_model(model, test_loader, device='cuda'):
    """
    Evaluate deep learning model on test set.
    
    Parameters:
    -----------
    model : nn.Module
        Trained PyTorch model
    test_loader : DataLoader
        Test data loader
    device : str, default='cuda'
    
    Returns:
    --------
    dict : Test predictions and probabilities
    """
    model.eval()
    all_preds = []
    all_probas = []
    all_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            
            _, predicted = torch.max(outputs, 1)
            probas = torch.softmax(outputs, dim=1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_probas.extend(probas.cpu().numpy())
            all_labels.extend(batch_y.numpy())
    
    results = {
        'y_test': np.array(all_labels),
        'y_test_pred': np.array(all_preds),
        'y_test_proba': np.array(all_probas)
    }
    
    return results


def save_dl_model(model, task_name, save_dir='experiments/saved_models'):
    """
    Save PyTorch model.
    
    Parameters:
    -----------
    model : nn.Module
        Trained model
    task_name : str
        Task name for file naming
    save_dir : str
        Save directory
    
    Returns:
    --------
    None
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    filename = save_path / f"{task_name}_LightweightCNN.pth"
    torch.save(model.state_dict(), filename)
    print(f"✅ Saved {filename}")


