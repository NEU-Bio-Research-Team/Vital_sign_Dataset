"""
Feature extraction utilities for ECG arrhythmia analysis.

Functions to calculate RR intervals, HRV features, and temporal patterns.
"""

import pandas as pd
import numpy as np


def calculate_rr_intervals(timestamps):
    """
    Calculate RR intervals from R-peak timestamps.
    
    Parameters:
    -----------
    timestamps : array-like
        R-peak timestamps in seconds
    
    Returns:
    --------
    np.array : RR intervals in seconds
    """
    
    timestamps = np.array(timestamps)
    rr_intervals = np.diff(timestamps)  # Difference between consecutive beats
    
    return rr_intervals


def extract_hrv_features(rr_intervals):
    """
    Extract heart rate variability (HRV) features from RR intervals.
    
    Parameters:
    -----------
    rr_intervals : array-like
        RR intervals in seconds
    
    Returns:
    --------
    dict : Dictionary of HRV features
    """
    
    if len(rr_intervals) < 2:
        return {
            'mean_rr': np.nan,
            'std_rr': np.nan,
            'rmssd': np.nan,
            'pnn50': np.nan,
            'pnn20': np.nan,
            'min_rr': np.nan,
            'max_rr': np.nan,
            'range_rr': np.nan,
            'mean_hr': np.nan,
            'std_hr': np.nan
        }
    
    rr_intervals = np.array(rr_intervals)
    
    # Filter out extreme values (likely artifacts)
    rr_filtered = rr_intervals[(rr_intervals > 0.3) & (rr_intervals < 3.0)]
    
    if len(rr_filtered) == 0:
        rr_filtered = rr_intervals
    
    # Time domain features
    mean_rr = np.mean(rr_filtered)
    std_rr = np.std(rr_filtered)
    
    # RMSSD: Root Mean Square of Successive Differences
    if len(rr_filtered) > 1:
        diff_rr = np.diff(rr_filtered)
        rmssd = np.sqrt(np.mean(diff_rr ** 2))
    else:
        rmssd = np.nan
    
    # pNN50: Percentage of RR intervals differing by >50ms
    if len(rr_filtered) > 1:
        diff_rr = np.diff(rr_filtered)
        pnn50 = np.sum(np.abs(diff_rr) > 0.05) / len(diff_rr) * 100
        pnn20 = np.sum(np.abs(diff_rr) > 0.02) / len(diff_rr) * 100
    else:
        pnn50 = np.nan
        pnn20 = np.nan
    
    # Range features
    min_rr = np.min(rr_filtered)
    max_rr = np.max(rr_filtered)
    range_rr = max_rr - min_rr
    
    # Heart rate (beats per minute)
    mean_hr = 60.0 / mean_rr if mean_rr > 0 else np.nan
    hr_values = 60.0 / rr_filtered[rr_filtered > 0]
    std_hr = np.std(hr_values) if len(hr_values) > 0 else np.nan
    
    return {
        'mean_rr': mean_rr,
        'std_rr': std_rr,
        'rmssd': rmssd,
        'pnn50': pnn50,
        'pnn20': pnn20,
        'min_rr': min_rr,
        'max_rr': max_rr,
        'range_rr': range_rr,
        'mean_hr': mean_hr,
        'std_hr': std_hr
    }


def extract_time_features(beats_df):
    """
    Extract temporal features from beats dataframe.
    
    Parameters:
    -----------
    beats_df : pd.DataFrame
        Annotation dataframe with time_second, beat_type, rhythm_label
    
    Returns:
    --------
    dict : Dictionary of temporal features
    """
    
    features = {}
    
    # Total duration
    if len(beats_df) > 0:
        features['total_duration_sec'] = beats_df['time_second'].iloc[-1] - beats_df['time_second'].iloc[0]
        features['total_beats'] = len(beats_df)
        features['beat_rate_bpm'] = features['total_beats'] / features['total_duration_sec'] * 60
    else:
        features['total_duration_sec'] = 0
        features['total_beats'] = 0
        features['beat_rate_bpm'] = 0
    
    # Beat type distribution
    beat_counts = beats_df['beat_type'].value_counts()
    total_beats = len(beats_df)
    
    for beat_type in ['N', 'S', 'V', 'U']:
        count = beat_counts.get(beat_type, 0)
        features[f'beat_type_{beat_type}_count'] = count
        features[f'beat_type_{beat_type}_percentage'] = count / total_beats * 100 if total_beats > 0 else 0
    
    # Rhythm label distribution
    rhythm_counts = beats_df['rhythm_label'].value_counts()
    rhythm_percentages = rhythm_counts / total_beats * 100 if total_beats > 0 else pd.Series()
    
    features['unique_rhythms'] = len(rhythm_counts)
    features['dominant_rhythm'] = rhythm_counts.index[0] if len(rhythm_counts) > 0 else 'Unknown'
    features['dominant_rhythm_percentage'] = rhythm_percentages.iloc[0] if len(rhythm_percentages) > 0 else 0
    
    # Signal quality
    if 'bad_signal_quality' in beats_df.columns:
        quality_counts = beats_df['bad_signal_quality'].value_counts()
        bad_quality_pct = quality_counts.get(True, 0) / total_beats * 100 if total_beats > 0 else 0
        features['bad_signal_quality_percentage'] = bad_quality_pct
    else:
        features['bad_signal_quality_percentage'] = 0
    
    # RR interval features
    rr_intervals = calculate_rr_intervals(beats_df['time_second'].values)
    hrv_features = extract_hrv_features(rr_intervals)
    features.update(hrv_features)
    
    return features


def extract_rhythm_features(annotation_data):
    """
    Extract features related to rhythm patterns and transitions.
    
    Parameters:
    -----------
    annotation_data : pd.DataFrame
        Annotation dataframe
    
    Returns:
    --------
    dict : Dictionary of rhythm features
    """
    
    features = {}
    
    if len(annotation_data) == 0:
        return features
    
    # Rhythm transitions
    rhythm_changes = (annotation_data['rhythm_label'] != annotation_data['rhythm_label'].shift()).sum()
    total_segments = len(annotation_data)
    features['rhythm_transition_rate'] = rhythm_changes / total_segments if total_segments > 0 else 0
    
    # Rhythm episode lengths
    rhythm_labels = annotation_data['rhythm_label'].values
    episode_lengths = []
    current_rhythm = rhythm_labels[0]
    current_length = 1
    
    for i in range(1, len(rhythm_labels)):
        if rhythm_labels[i] == current_rhythm:
            current_length += 1
        else:
            episode_lengths.append(current_length)
            current_rhythm = rhythm_labels[i]
            current_length = 1
    episode_lengths.append(current_length)  # Last episode
    
    if len(episode_lengths) > 0:
        features['mean_episode_length'] = np.mean(episode_lengths)
        features['median_episode_length'] = np.median(episode_lengths)
        features['max_episode_length'] = np.max(episode_lengths)
        features['min_episode_length'] = np.min(episode_lengths)
        features['std_episode_length'] = np.std(episode_lengths)
    else:
        features['mean_episode_length'] = 0
        features['median_episode_length'] = 0
        features['max_episode_length'] = 0
        features['min_episode_length'] = 0
        features['std_episode_length'] = 0
    
    return features

