"""
Visualization utilities for arrhythmia analysis.

Functions to create plots for rhythm distributions, beat types, and temporal patterns.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def plot_rhythm_distribution(rhythm_stats, save_path=None):
    """
    Plot distribution of rhythm classes.
    
    Parameters:
    -----------
    rhythm_stats : dict
        Statistics from get_rhythm_statistics()
    save_path : str, optional
        Path to save the figure
    """
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Bar chart
    rhythms = list(rhythm_stats['rhythm_counts'].keys())
    counts = [rhythm_stats['rhythm_counts'][r] for r in rhythms]
    
    ax1 = axes[0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(rhythms)))
    ax1.barh(rhythms, counts, color=colors)
    ax1.set_xlabel('Number of Cases', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Rhythm Class', fontsize=12, fontweight='bold')
    ax1.set_title('Rhythm Class Distribution (Cases)', fontsize=14, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # Pie chart
    ax2 = axes[1]
    wedges, texts, autotexts = ax2.pie(counts, labels=rhythms, autopct='%1.1f%%', 
                                        startangle=90, colors=colors)
    ax2.set_title('Rhythm Class Proportions', fontsize=14, fontweight='bold')
    
    # Make percentage text bold and readable
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontweight('bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_beat_type_distribution(beat_counts, save_path=None):
    """
    Plot distribution of beat types.
    
    Parameters:
    -----------
    beat_counts : dict or pd.Series
        Beat type counts
    save_path : str, optional
        Path to save the figure
    """
    
    if isinstance(beat_counts, dict):
        beat_counts = pd.Series(beat_counts)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors_dict = {'N': '#2ecc71', 'S': '#3498db', 'V': '#e74c3c', 'U': '#f39c12'}
    colors = [colors_dict.get(bt, '#95a5a6') for bt in beat_counts.index]
    
    bars = ax.bar(beat_counts.index, beat_counts.values, color=colors, alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_xlabel('Beat Type', fontsize=12, fontweight='bold')
    ax.set_title('Beat Type Distribution', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_patient_statistics(metadata_df, save_path=None):
    """
    Plot patient-level statistics.
    
    Parameters:
    -----------
    metadata_df : pd.DataFrame
        Metadata dataframe
    save_path : str, optional
        Path to save the figure
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Duration distribution
    ax1 = axes[0, 0]
    ax1.hist(metadata_df['analyzed_duration_sec'] / 60, bins=30, color='#3498db', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Recording Duration (minutes)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax1.set_title('Distribution of Recording Duration', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Beat count distribution
    ax2 = axes[0, 1]
    ax2.hist(metadata_df['total_beats'], bins=30, color='#e74c3c', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Total Beats', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax2.set_title('Distribution of Total Beats per Patient', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Duration vs Beats scatter
    ax3 = axes[1, 0]
    ax3.scatter(metadata_df['analyzed_duration_sec'] / 60, metadata_df['total_beats'], 
               alpha=0.6, s=50, color='#27ae60')
    ax3.set_xlabel('Duration (minutes)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Total Beats', fontsize=12, fontweight='bold')
    ax3.set_title('Duration vs Total Beats', fontsize=13, fontweight='bold')
    ax3.grid(alpha=0.3)
    
    # 4. Beats per minute
    beats_per_min = (metadata_df['total_beats'] / (metadata_df['analyzed_duration_sec'] / 60)).fillna(0)
    ax4 = axes[1, 1]
    ax4.hist(beats_per_min, bins=30, color='#f39c12', alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Beats per Minute', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax4.set_title('Heart Rate Distribution', fontsize=13, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_temporal_patterns(annotation_df, case_id=None, save_path=None):
    """
    Plot temporal patterns of RR intervals and heart rate.
    
    Parameters:
    -----------
    annotation_df : pd.DataFrame
        Annotation dataframe for a single case
    case_id : int or str, optional
        Case ID for title
    save_path : str, optional
        Path to save the figure
    """
    
    # Calculate RR intervals
    timestamps = annotation_df['time_second'].values
    rr_intervals = np.diff(timestamps)
    
    # Calculate heart rate
    hr_values = 60.0 / rr_intervals
    hr_times = timestamps[1:]  # Align with RR intervals
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    
    # 1. RR intervals over time
    ax1 = axes[0]
    ax1.plot(hr_times, rr_intervals, 'b-', alpha=0.6, linewidth=1)
    ax1.set_ylabel('RR Interval (sec)', fontsize=12, fontweight='bold')
    ax1.set_title(f'RR Intervals Over Time (Case {case_id})' if case_id else 'RR Intervals Over Time', 
                 fontsize=13, fontweight='bold')
    ax1.grid(alpha=0.3)
    
    # 2. Heart rate over time
    ax2 = axes[1]
    ax2.plot(hr_times, hr_values, 'r-', alpha=0.6, linewidth=1)
    ax2.set_ylabel('Heart Rate (BPM)', fontsize=12, fontweight='bold')
    ax2.set_title('Heart Rate Over Time', fontsize=13, fontweight='bold')
    ax2.grid(alpha=0.3)
    
    # 3. Rhythm labels over time
    ax3 = axes[2]
    
    # Create colormap for rhythm types
    unique_rhythms = annotation_df['rhythm_label'].unique()
    colors_map = plt.cm.Set3(np.linspace(0, 1, len(unique_rhythms)))
    rhythm_colors = {rhythm: colors_map[i] for i, rhythm in enumerate(unique_rhythms)}
    
    for rhythm in unique_rhythms:
        mask = annotation_df['rhythm_label'] == rhythm
        times = annotation_df[mask]['time_second']
        ax3.scatter(times, [rhythm] * len(times), c=[rhythm_colors[rhythm]], 
                   label=rhythm, alpha=0.6, s=20)
    
    ax3.set_ylabel('Rhythm Label', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax3.set_title('Rhythm Transitions Over Time', fontsize=13, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_quality_analysis(quality_stats, save_path=None):
    """
    Plot signal quality analysis.
    
    Parameters:
    -----------
    quality_stats : dict
        Quality statistics
    save_path : str, optional
        Path to save the figure
    """
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    categories = list(quality_stats.keys())
    percentages = list(quality_stats.values())
    
    colors = ['#e74c3c' if p > 10 else '#27ae60' for p in percentages]
    
    bars = ax.bar(categories, percentages, color=colors, alpha=0.8, edgecolor='black')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Quality Category', fontsize=12, fontweight='bold')
    ax.set_title('Signal Quality Analysis', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


