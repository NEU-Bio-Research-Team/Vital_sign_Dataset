"""
Utility functions for AKI Prediction project.

This module contains utility functions for data loading, preprocessing,
and common operations used across the project.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


def setup_plotting():
    """Setup matplotlib and seaborn for consistent plotting."""
    import matplotlib
    matplotlib.rcParams['figure.figsize'] = [10, 6]
    matplotlib.rcParams['figure.dpi'] = 100

    try:
        plt.style.use('seaborn-v0_8')
    except:
        try:
            plt.style.use('seaborn')
        except:
            try:
                plt.style.use('ggplot')
            except:
                plt.style.use('default')

    try:
        sns.set_palette("husl")
    except:
        pass


def load_vitaldb_data():
    """
    Load and preprocess VitalDB dataset for AKI prediction.
    
    Returns:
    --------
    tuple: (X, y, feature_names, df)
        X: Feature matrix
        y: Target labels
        feature_names: List of feature names
        df: Original dataframe
    """
    print("🔄 Loading VitalDB dataset...")
    
    # Load datasets
    df = pd.read_csv('https://api.vitaldb.net/cases')
    df['sex'] = (df['sex'] == 'M')

    # Load labs
    df_labs = pd.read_csv('https://api.vitaldb.net/labs')
    df_labs = df_labs.loc[df_labs.name == 'cr']

    # Process postop creatinine level within 7 days after surgery
    df_labs = pd.merge(df, df_labs, on='caseid', how='left')
    df_labs = df_labs.loc[df_labs.dt > df_labs.opend]
    df_labs = df_labs.loc[df_labs.dt < df_labs.opend + 7 * 3600 * 24]
    df_labs = df_labs.groupby('caseid')['result'].max().reset_index()
    df_labs.rename(columns={'result':'postop_cr'}, inplace=True)

    df = pd.merge(df, df_labs, on='caseid', how='left')
    df.dropna(subset=['preop_cr', 'postop_cr'], inplace=True)
    
    print(f"✅ Dataset loaded: {len(df)} records")
    print(f"📊 Features available: {len(df.columns)}")
    
    return df


def preprocess_data(df):
    """
    Preprocess the dataset for machine learning.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Raw dataset
    
    Returns:
    --------
    tuple: (X, y, feature_names)
        X: Feature matrix
        y: Target labels
        feature_names: List of feature names
    """
    print("🔧 Preprocessing data...")
    
    # Remove categorical variables
    df = df.drop(columns=['department','optype', 'dx', 'opname', 'approach', 'position', 
                         'ane_type', 'cormack', 'airway', 'tubesize', 'dltubesize', 
                         'lmasize', 'preop_ecg', 'preop_pft', 'iv1', 'iv2', 
                         'aline1', 'aline2', 'cline1', 'cline2'])
    df = df.astype(float)

    # KDIGO stage I - AKI definition
    df['aki'] = df['postop_cr'] > df['preop_cr'] * 1.5

    # Remove outcome variables and prepare features
    y = df['aki'].values.flatten()
    df['andur'] = df['aneend'] - df['anestart']
    df = df.drop(columns=['aki', 'postop_cr', 'death_inhosp','caseid',
                         'subjectid','icu_days','casestart','caseend',
                         'anestart','aneend','opstart','opend','adm','dis'])

    # Store feature names
    feature_names = df.columns.tolist()
    
    # Input variables
    X = df.values
    
    print(f"✅ Data preprocessing completed")
    print(f"📊 Final dataset shape: {X.shape}")
    print(f"🎯 Target distribution: {sum(y)}/{len(y)} positive cases ({np.mean(y)*100:.2f}%)")
    
    return X, y, feature_names


def prepare_train_test_data(X, y, test_size=0.2, random_state=0):
    """
    Split data into train/test sets and prepare for different model types.
    
    Parameters:
    -----------
    X : array-like
        Feature matrix
    y : array-like
        Target labels
    test_size : float, default=0.2
        Proportion of data for testing
    random_state : int, default=0
        Random state for reproducibility
    
    Returns:
    --------
    dict: Dictionary containing train/test data for different preprocessing types
    """
    print("🔧 Preparing train/test data...")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    # Scale features for models that need it
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)
    
    print(f"📊 Training set: {X_train_imp.shape}")
    print(f"📊 Test set: {X_test_imp.shape}")
    
    # Prepare data dictionaries
    X_train_dict = {
        'scaled': X_train_scaled,
        'imputed': X_train_imp
    }
    
    X_test_dict = {
        'scaled': X_test_scaled, 
        'imputed': X_test_imp
    }
    
    data_dict = {
        'X_train_dict': X_train_dict,
        'X_test_dict': X_test_dict,
        'y_train': y_train,
        'y_test': y_test,
        'imputer': imputer,
        'scaler': scaler
    }
    
    return data_dict


def save_predictions(y_true, y_pred, y_pred_proba, model_name, results_dir='results'):
    """
    Save prediction results to CSV files.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    y_pred_proba : array-like
        Predicted probabilities
    model_name : str
        Name of the model
    results_dir : str, default='results'
        Directory to save results
    """
    import os
    
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    })
    
    # Save to CSV
    filename = f"{results_dir}/{model_name}_predictions.csv"
    results_df.to_csv(filename, index=False)
    print(f"✅ Predictions saved to: {filename}")


def load_predictions(model_name, results_dir='results'):
    """
    Load prediction results from CSV files.
    
    Parameters:
    -----------
    model_name : str
        Name of the model
    results_dir : str, default='results'
        Directory containing results
    
    Returns:
    --------
    pandas.DataFrame: Prediction results
    """
    filename = f"{results_dir}/{model_name}_predictions.csv"
    try:
        results_df = pd.read_csv(filename)
        print(f"✅ Predictions loaded from: {filename}")
        return results_df
    except FileNotFoundError:
        print(f"❌ Prediction file not found: {filename}")
        return None
