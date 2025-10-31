"""
Data preprocessing utilities for VitalDB Arrhythmia Database.

Functions for loading all cases, extracting features, creating labels,
sequence windowing, train/test splits, and class weight calculation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler

from data_loader import load_metadata, load_annotation_file, get_rhythm_statistics
from feature_extractor import (
    calculate_rr_intervals,
    extract_hrv_features,
    extract_time_features,
    extract_rhythm_features
)

# Force reload module to ensure we have latest version
import importlib
import sys
if 'preprocess' in sys.modules:
    importlib.reload(sys.modules['preprocess'])
    
# Classes to exclude during preprocessing
EXCLUDED_CLASSES = {"Unclassifiable", "VT"}

def _is_excluded(label: object) -> bool:
    """
    Return True if the label should be excluded (in EXCLUDED_CLASSES or NaN).
    Handles various NaN representations: None, np.nan, pd.NA, string 'nan', 'NaN', etc.
    """
    # Check for various NaN representations
    if pd.isna(label) or label is None:
        return True
    
    # Convert to string and check for NaN strings
    try:
        label_str = str(label).strip().lower()
        if label_str in ['nan', 'none', '', 'null']:
            return True
        # Check if it's in excluded classes
        return str(label) in EXCLUDED_CLASSES
    except Exception:
        return False

def _filter_excluded_from_dataframe(df: pd.DataFrame, label_column: str) -> pd.DataFrame:
    """
    Remove rows whose label is Unclassifiable, VT, or NaN.
    """
    if label_column not in df.columns:
        return df
    mask = df[label_column].apply(lambda x: not _is_excluded(x))
    filtered = df[mask].reset_index(drop=True)
    removed = len(df) - len(filtered)
    if removed > 0:
        print(f"Filtered {removed} samples from '{label_column}' (excluded classes: {sorted(EXCLUDED_CLASSES)} + NaN)")
    return filtered

def load_all_cases(data_path=None):
    """
    Load annotation files for all cases in the dataset.
    Filters out rows with excluded labels (Unclassifiable, VT, NaN) from annotation dataframes.
    
    Parameters:
    -----------
    data_path : str or Path, optional
        Path to annotation files directory
    
    Returns:
    --------
    dict : Dictionary mapping case_id to annotation dataframe
    """
    metadata = load_metadata(data_path)
    case_ids = metadata['case_id'].tolist()
    
    annotations = {}
    failed_cases = []
    total_removed = 0
    
    for case_id in case_ids:
        try:
            df = load_annotation_file(case_id, data_path)
            
            # Filter out rows with excluded rhythm labels or NaN
            if 'rhythm_label' in df.columns:
                original_len = len(df)
                df = _filter_excluded_from_dataframe(df, 'rhythm_label')
                removed = original_len - len(df)
                total_removed += removed
            
            annotations[case_id] = df
        except Exception as e:
            print(f"Warning: Could not load case {case_id}: {e}")
            failed_cases.append(case_id)
            continue
    
    print(f"✅ Successfully loaded {len(annotations)}/{len(case_ids)} cases")
    if total_removed > 0:
        print(f"⚠️  Filtered {total_removed} total annotation rows with excluded labels (Unclassifiable, VT, NaN)")
    if failed_cases:
        print(f"⚠️  Failed to load {len(failed_cases)} cases: {failed_cases[:10]}...")
    
    return annotations, metadata


def extract_patient_level_features(annotation_df, case_id):
    """
    Extract comprehensive features for a single patient.
    
    Parameters:
    -----------
    annotation_df : pd.DataFrame
        Annotation dataframe with time_second, beat_type, rhythm_label
    case_id : int
        Case ID
    
    Returns:
    --------
    dict : Dictionary of features
    """
    # Calculate RR intervals
    rr_intervals = calculate_rr_intervals(annotation_df['time_second'].values)
    
    # Extract HRV features
    hrv_features = extract_hrv_features(rr_intervals)
    
    # Extract time features
    time_features = extract_time_features(annotation_df)
    
    # Extract rhythm features
    rhythm_features = extract_rhythm_features(annotation_df)
    
    # Combine all features
    combined_features = {
        'case_id': case_id,
        **hrv_features,
        **time_features,
        **rhythm_features
    }
    
    return combined_features


def create_tabular_dataset(annotations_dict, metadata_df):
    """
    Create tabular dataset from all patient annotations.
    
    Parameters:
    -----------
    annotations_dict : dict
        Dictionary of annotation dataframes
    metadata_df : pd.DataFrame
        Metadata dataframe
    
    Returns:
    --------
    pd.DataFrame : Feature dataframe with one row per patient
    """
    all_features = []
    
    print(f"Extracting features from {len(annotations_dict)} patients...")
    
    for case_id, df in annotations_dict.items():
        try:
            features = extract_patient_level_features(df, case_id)
            all_features.append(features)
        except Exception as e:
            print(f"Warning: Feature extraction failed for case {case_id}: {e}")
            continue
    
    features_df = pd.DataFrame(all_features)
    print(f"✅ Extracted {len(features_df.columns)} features for {len(features_df)} patients")
    
    return features_df


def create_beat_labels(annotations_dict):
    """
    Create beat-level labels for each patient.
    
    Parameters:
    -----------
    annotations_dict : dict
        Dictionary of annotation dataframes
    
    Returns:
    --------
    pd.DataFrame : DataFrame with case_id and beat labels
    """
    beat_labels = []
    
    for case_id, df in annotations_dict.items():
        # For beat-level classification, aggregate to patient level
        # Use majority beat type as label
        beat_counts = df['beat_type'].value_counts()
        dominant_beat = beat_counts.index[0]
        beat_labels.append({
            'case_id': case_id,
            'beat_label': dominant_beat
        })
    
    labels_df = pd.DataFrame(beat_labels)
    # Beats exclusion not requested explicitly; keep as-is.
    return labels_df


def create_rhythm_labels(annotations_dict, metadata_df):
    """
    Create rhythm-level labels for each patient.
    Filters out excluded classes (Unclassifiable, VT, NaN).
    
    Parameters:
    -----------
    annotations_dict : dict
        Dictionary of annotation dataframes
    metadata_df : pd.DataFrame
        Metadata dataframe with rhythm_classes
    
    Returns:
    --------
    pd.DataFrame : DataFrame with case_id and rhythm labels (excluding Unclassifiable, VT, NaN)
    """
    rhythm_labels = []
    
    for case_id, df in annotations_dict.items():
        # Get rhythm classes from metadata
        if case_id in metadata_df['case_id'].values:
            rhythm_str = metadata_df[metadata_df['case_id'] == case_id]['rhythm_classes'].iloc[0]
            
            # Handle NaN in metadata
            if pd.isna(rhythm_str):
                continue
            
            rhythms = [r.strip() for r in str(rhythm_str).split(',')]
            
            # Use first (primary) rhythm as label, but skip if excluded
            primary_rhythm = rhythms[0]
            if _is_excluded(primary_rhythm):
                continue
            
            rhythm_labels.append({
                'case_id': case_id,
                'rhythm_label': primary_rhythm
            })
    
    labels_df = pd.DataFrame(rhythm_labels)
    # Additional filter as safety check
    labels_df = _filter_excluded_from_dataframe(labels_df, 'rhythm_label')
    return labels_df


def create_sequence_windows(annotation_df, window_size=60, stride=30, task='rhythm'):
    """
    Create fixed-length sequence windows for deep learning.
    Only creates windows where the label is not excluded (Unclassifiable, VT, NaN).
    
    Parameters:
    -----------
    annotation_df : pd.DataFrame
        Annotation dataframe (should already be filtered for excluded labels)
    window_size : int, default=60
        Number of beats per window
    stride : int, default=30
        Number of beats to stride between windows
    task : str, default='rhythm'
        'rhythm' or 'beat' - determines label type
    
    Returns:
    --------
    tuple : (sequences, labels)
        sequences: (n_windows, window_size) array of RR intervals
        labels: (n_windows,) array of labels (all valid, no excluded classes)
    """
    # Calculate RR intervals
    rr_intervals = calculate_rr_intervals(annotation_df['time_second'].values)
    
    sequences = []
    labels = []
    
    # Create overlapping windows
    for i in range(0, len(rr_intervals) - window_size + 1, stride):
        window_rr = rr_intervals[i:i + window_size]
        
        if len(window_rr) == window_size:
            # Get label for this window (majority label)
            if task == 'rhythm':
                rhythm_window = annotation_df['rhythm_label'].iloc[i+1:i+1+window_size]
                label = rhythm_window.mode().iloc[0] if len(rhythm_window.mode()) > 0 else rhythm_window.iloc[0]
            else:  # beat
                beat_window = annotation_df['beat_type'].iloc[i+1:i+1+window_size]
                label = beat_window.mode().iloc[0] if len(beat_window.mode()) > 0 else beat_window.iloc[0]
            
            # Skip windows with excluded labels
            if _is_excluded(label):
                continue
            
            sequences.append(window_rr)
            labels.append(label)
    
    return np.array(sequences), np.array(labels)


def create_sequence_dataset(annotations_dict, metadata_df, window_size=60, stride=30, task='rhythm'):
    """
    Create sequence dataset for deep learning from all patients.
    
    Parameters:
    -----------
    annotations_dict : dict
        Dictionary of annotation dataframes
    metadata_df : pd.DataFrame
        Metadata dataframe
    window_size : int, default=60
        Number of beats per window
    stride : int, default=30
        Stride between windows
    task : str, default='rhythm'
        'rhythm' or 'beat'
    
    Returns:
    --------
    tuple : (sequences, labels, patient_ids)
    """
    all_sequences = []
    all_labels = []
    patient_ids = []
    
    print(f"Creating sequence windows (size={window_size}, stride={stride})...")
    
    for case_id, df in annotations_dict.items():
        try:
            sequences, labels = create_sequence_windows(df, window_size, stride, task)
            all_sequences.append(sequences)
            all_labels.append(labels)
            patient_ids.extend([case_id] * len(labels))
        except Exception as e:
            print(f"Warning: Window creation failed for case {case_id}: {e}")
            continue
    
    # Concatenate all sequences
    sequences = np.concatenate(all_sequences, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    patient_ids = np.array(patient_ids)
    
    # Filter out excluded labels (applies to rhythm task; for beat we honor same rule if present)
    valid_mask = np.array([not _is_excluded(lbl) for lbl in labels])
    if valid_mask.sum() != len(labels):
        removed = int((~valid_mask).sum())
        print(f"Filtered {removed} windows with excluded labels ({sorted(EXCLUDED_CLASSES)} + NaN)")
        sequences = sequences[valid_mask]
        labels = labels[valid_mask]
        patient_ids = patient_ids[valid_mask]

    print(f"✅ Created {len(sequences)} windows from {len(annotations_dict)} patients")
    
    return sequences, labels, patient_ids


def split_data_patients(X_df, y_df, test_size=0.2, val_size=0.2, random_state=42, stratify=True):
    """
    Split data with patient-level stratification.
    
    Parameters:
    -----------
    X_df : pd.DataFrame
        Feature dataframe with case_id column
    y_df : pd.DataFrame
        Label dataframe with case_id column
    test_size : float, default=0.2
        Size of test set (20%)
    val_size : float, default=0.2
        Size of validation set (20%)
    random_state : int, default=42
        Random seed
    stratify : bool, default=True
        Whether to use stratified splitting
    
    Returns:
    --------
    dict : Dictionary with train/val/test splits
    """
    # Merge features and labels
    data = pd.merge(X_df, y_df, on='case_id')
    
    # Extract feature columns (exclude case_id and label columns)
    # Only include numeric columns to avoid string column errors
    feature_cols = [col for col in X_df.columns if col != 'case_id']
    label_col = y_df.columns[1]  # Second column is the label
    
    X = data[feature_cols]
    y = data[label_col]
    case_ids = data['case_id']
    
    # Remove non-numeric columns (like 'dominant_rhythm' string column)
    X_numeric = X.select_dtypes(include=[np.number])
    
    # Check if stratification is possible
    # Stratification requires at least 2 samples per class
    if stratify:
        class_counts = y.value_counts()
        min_class_count = class_counts.min()
        if min_class_count < 2:
            print(f"Warning: Some classes have <2 samples (min={min_class_count}). Disabling stratification.")
            stratify = False
    
    X = X_numeric
    
    # First split: train+val vs test
    if stratify:
        X_temp, X_test, y_temp, y_test, ids_temp, ids_test = train_test_split(
            X, y, case_ids, test_size=test_size, random_state=random_state, stratify=y
        )
    else:
        X_temp, X_test, y_temp, y_test, ids_temp, ids_test = train_test_split(
            X, y, case_ids, test_size=test_size, random_state=random_state
        )
    
    # Second split: train vs val
    val_size_adjusted = val_size / (1 - test_size)  # Adjust for already split data
    if stratify:
        X_train, X_val, y_train, y_val, ids_train, ids_val = train_test_split(
            X_temp, y_temp, ids_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
        )
    else:
        X_train, X_val, y_train, y_val, ids_train, ids_val = train_test_split(
            X_temp, y_temp, ids_temp, test_size=val_size_adjusted, random_state=random_state
        )
    
    splits = {
        'X_train': X_train, 'y_train': y_train, 'ids_train': ids_train,
        'X_val': X_val, 'y_val': y_val, 'ids_val': ids_val,
        'X_test': X_test, 'y_test': y_test, 'ids_test': ids_test
    }
    
    print(f"✅ Train/Val/Test split: {len(X_train)}/{len(X_val)}/{len(X_test)} patients")
    
    return splits


def calculate_class_weights(y, classes):
    """
    Calculate balanced class weights.
    
    Parameters:
    -----------
    y : array-like
        Class labels
    classes : array-like
        Unique class labels
    
    Returns:
    --------
    dict : Dictionary mapping class to weight
    """
    # Convert classes to numpy array
    classes_array = np.array(classes)
    
    weights = compute_class_weight('balanced', classes=classes_array, y=y)
    weight_dict = dict(zip(classes, weights))
    
    print("Class weights:", weight_dict)
    
    return weight_dict


def preprocess_for_ml(features_df, labels_df, test_size=0.2, val_size=0.2, scale=True, random_state=42):
    """
    Complete preprocessing pipeline for traditional ML models.
    
    Parameters:
    -----------
    features_df : pd.DataFrame
        Feature dataframe
    labels_df : pd.DataFrame
        Label dataframe
    test_size : float, default=0.2
        Test set size
    val_size : float, default=0.2
        Validation set size
    scale : bool, default=True
        Whether to standardize features
    random_state : int, default=42
    
    Returns:
    --------
    dict : Preprocessing results with splits, scaler, class weights
    """
    # Split data
    splits = split_data_patients(
        features_df, labels_df, test_size=test_size, val_size=val_size, random_state=random_state
    )
    
    # Get unique classes and calculate weights
    label_col = labels_df.columns[1]
    all_labels = pd.concat([splits['y_train'], splits['y_val'], splits['y_test']])
    unique_classes = sorted(all_labels.unique())
    class_weights = calculate_class_weights(all_labels, unique_classes)
    
    # Scaling (only numeric columns)
    if scale:
        scaler = StandardScaler()
        
        # Select only numeric columns
        numeric_cols = splits['X_train'].select_dtypes(include=[np.number]).columns
        
        # Fit on training data
        X_train_scaled = scaler.fit_transform(splits['X_train'][numeric_cols])
        X_train_full = splits['X_train'].copy()
        X_train_full[numeric_cols] = X_train_scaled
        splits['X_train'] = X_train_full
        
        # Transform validation and test data
        X_val_scaled = scaler.transform(splits['X_val'][numeric_cols])
        X_val_full = splits['X_val'].copy()
        X_val_full[numeric_cols] = X_val_scaled
        splits['X_val'] = X_val_full
        
        X_test_scaled = scaler.transform(splits['X_test'][numeric_cols])
        X_test_full = splits['X_test'].copy()
        X_test_full[numeric_cols] = X_test_scaled
        splits['X_test'] = X_test_full
    else:
        scaler = None
    
    result = {
        'splits': splits,
        'scaler': scaler,
        'class_weights': class_weights,
        'unique_classes': unique_classes,
        'label_col': label_col
    }
    
    return result


# Beat and Rhythm label constants
BEAT_LABELS = ['N', 'S', 'V', 'U']
RHYTHM_LABELS = ['N', 'AFIB/AFL', 'SR-mPVC-BT', 'SR-mPAC-BT', 
                 'SVTA', 'VT', 'SND', 'MAT', 'Noise', 'AVB', 'Unclassifiable']

