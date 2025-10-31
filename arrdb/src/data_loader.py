"""
Data loading utilities for VitalDB Arrhythmia Database.

Functions to load metadata and annotation files.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_metadata(data_path=None):
    """
    Load the metadata CSV file.
    
    Parameters:
    -----------
    data_path : str or Path, optional
        Path to the LabelFile directory. Default is './LabelFile'
    
    Returns:
    --------
    pd.DataFrame : Metadata dataframe
    """
    
    if data_path is None:
        # Default path assuming we're in arrdb/
        data_path = Path(__file__).parent.parent / 'LabelFile'
    else:
        data_path = Path(data_path)
    
    metadata_file = data_path / 'metadata.csv'
    
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found at: {metadata_file}")
    
    df = pd.read_csv(metadata_file)
    
    return df


def load_annotation_file(case_id, data_path=None):
    """
    Load annotation file for a specific case.
    
    Parameters:
    -----------
    case_id : int or str
        Case ID to load
    data_path : str or Path, optional
        Path to the annotation files directory. Default is './LabelFile/Annotation_Files_250907'
    
    Returns:
    --------
    pd.DataFrame : Annotation dataframe with columns:
        - time_second: R-peak timestamp
        - beat_type: Individual beat classification (N, S, V, U)
        - rhythm_label: Overall rhythm label
        - bad_signal_quality: Boolean flag
        - bad_signal_quality_label: Quality issue description
    """
    
    if data_path is None:
        # Default path
        data_path = Path(__file__).parent.parent / 'LabelFile' / 'Annotation_Files_250907'
    else:
        data_path = Path(data_path)
    
    annotation_file = data_path / f'Annotation_file_{case_id}.csv'
    
    if not annotation_file.exists():
        raise FileNotFoundError(f"Annotation file not found for case {case_id}: {annotation_file}")
    
    df = pd.read_csv(annotation_file)
    
    return df


def load_multiple_cases(case_ids, data_path=None):
    """
    Load annotation files for multiple cases.
    
    Parameters:
    -----------
    case_ids : list
        List of case IDs to load
    data_path : str or Path, optional
        Path to annotation files directory
    
    Returns:
    --------
    dict : Dictionary mapping case_id to annotation dataframe
    """
    
    annotations = {}
    
    for case_id in case_ids:
        try:
            annotations[case_id] = load_annotation_file(case_id, data_path)
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            continue
    
    return annotations


def get_rhythm_statistics(metadata_df, annotation_data=None):
    """
    Get statistics about rhythm classes in the dataset.
    
    Parameters:
    -----------
    metadata_df : pd.DataFrame
        Metadata dataframe from load_metadata()
    annotation_data : dict, optional
        Dictionary of annotation dataframes from load_multiple_cases()
        If None, will only use metadata
    
    Returns:
    --------
    dict : Statistics about rhythm classes
    """
    
    # Get unique rhythm classes from metadata
    all_rhythms = set()
    for rhythm_str in metadata_df['rhythm_classes']:
        rhythms = [r.strip() for r in str(rhythm_str).split(',')]
        all_rhythms.update(rhythms)
    
    # Count occurrences
    rhythm_counts = {rhythm: 0 for rhythm in all_rhythms}
    rhythm_patients = {rhythm: set() for rhythm in all_rhythms}
    
    for idx, row in metadata_df.iterrows():
        case_id = row['case_id']
        rhythms = [r.strip() for r in str(row['rhythm_classes']).split(',')]
        
        for rhythm in rhythms:
            rhythm_counts[rhythm] += 1
            rhythm_patients[rhythm].add(case_id)
    
    # Count unique patients per rhythm
    rhythm_unique_patients = {rhythm: len(patients) for rhythm, patients in rhythm_patients.items()}
    
    return {
        'all_rhythms': sorted(list(all_rhythms)),
        'rhythm_counts': rhythm_counts,
        'unique_patients': rhythm_unique_patients,
        'total_unique_rhythms': len(all_rhythms)
    }


def filter_by_rhythm(metadata_df, rhythm_label):
    """
    Filter metadata to find cases with a specific rhythm label.
    
    Parameters:
    -----------
    metadata_df : pd.DataFrame
        Metadata dataframe
    rhythm_label : str
        Rhythm label to filter by
    
    Returns:
    --------
    pd.DataFrame : Filtered metadata
    """
    
    mask = metadata_df['rhythm_classes'].str.contains(rhythm_label, na=False, regex=False)
    
    return metadata_df[mask].copy()


def get_dataset_summary(metadata_df, annotation_dict=None):
    """
    Get overall dataset summary statistics.
    
    Parameters:
    -----------
    metadata_df : pd.DataFrame
        Metadata dataframe
    annotation_dict : dict, optional
        Dictionary of annotation dataframes
    
    Returns:
    --------
    dict : Summary statistics
    """
    
    summary = {
        'total_patients': len(metadata_df),
        'total_duration_sec': metadata_df['analyzed_duration_sec'].sum(),
        'total_duration_hours': metadata_df['analyzed_duration_sec'].sum() / 3600,
        'total_beats': metadata_df['total_beats'].sum(),
        'avg_duration_per_patient': metadata_df['analyzed_duration_sec'].mean(),
        'avg_beats_per_patient': metadata_df['total_beats'].mean(),
        'median_duration': metadata_df['analyzed_duration_sec'].median(),
        'median_beats': metadata_df['total_beats'].median()
    }
    
    return summary

