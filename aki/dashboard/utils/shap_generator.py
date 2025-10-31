"""
SHAP plot generator for chatbot display.

Creates SHAP waterfall/bar plots with medical color scheme.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import base64
import io


def generate_shap_plot(top_factors, prediction_result):
    """
    Generate a SHAP-style waterfall plot visualization.
    
    Parameters:
    -----------
    top_factors : list
        List of risk factors with contributions
    prediction_result : dict
        Full prediction results
    
    Returns:
    --------
    str : Base64 encoded PNG image
    """
    
    # Extract factors and contributions
    factor_names = [f['factor'] for f in top_factors[:8]]  # Top 8
    contributions = [f['contribution'] for f in top_factors[:8]]
    
    # Add prediction probability at end
    factor_names.append('Model Prediction')
    contributions.append(prediction_result['probability'])
    
    # Create figure with medical theme
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Flowchart color scheme
    colors = ['#2E86AB', '#06A77D', '#3498DB', '#F18F01', '#6A4C93',
              '#C73E1D', '#27AE60', '#8E44AD']
    
    # Bar positions
    y_pos = np.arange(len(factor_names))
    
    # Create horizontal bars
    bars = ax.barh(y_pos, contributions, color=colors[:len(factor_names)], 
                   alpha=0.8, edgecolor='#2C3E50', linewidth=1.5)
    
    # Customize axes
    ax.set_yticks(y_pos)
    ax.set_yticklabels(factor_names, fontsize=10)
    ax.set_xlabel('Risk Contribution', fontsize=12, fontweight='bold', color='#2C3E50')
    ax.set_title('SHAP Explanation - Key Risk Factors', 
                 fontsize=14, fontweight='bold', color='#2C3E50', pad=15)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, contributions)):
        if i == len(bars) - 1:  # Highlight final prediction
            ax.text(bar.get_width()/2, bar.get_y() + bar.get_height()/2,
                   f'{val:.2f}', ha='center', va='center',
                   fontweight='bold', fontsize=11, color='#FFFFFF')
        else:
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{val:.2f}', va='center', fontsize=9, color='#2C3E50')
    
    # Style
    ax.set_facecolor('#FFFFFF')
    ax.grid(axis='x', alpha=0.3, color='#BDC3C7', linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#2C3E50')
    ax.spines['bottom'].set_color('#2C3E50')
    
    # Add separator line before final prediction
    if len(factor_names) > 1:
        ax.axhline(y=len(factor_names)-1.5, color='#2C3E50', linewidth=2, alpha=0.3)
    
    plt.tight_layout()
    
    # Convert to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return img_base64


def generate_risk_gauge(probability):
    """
    Generate a risk probability gauge visualization.
    
    Parameters:
    -----------
    probability : float
        Risk probability (0-1)
    
    Returns:
    --------
    str : Base64 encoded PNG image
    """
    
    from matplotlib.patches import Circle, Wedge, Rectangle
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim([-1.3, 1.3])
    ax.set_ylim([-1.3, 1.3])
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Determine risk zone
    if probability < 0.25:
        color = '#27AE60'  # Green (low risk)
        zone = 'Low Risk'
    elif probability < 0.50:
        color = '#F18F01'  # Orange (medium risk)
        zone = 'Medium Risk'
    else:
        color = '#C73E1D'  # Red (high risk)
        zone = 'High Risk'
    
    # Draw gauge arc
    wedge = Wedge(center=(0, 0), r=1.0, theta1=0, theta2=180, 
                  width=0.3, color=color, alpha=0.3)
    ax.add_patch(wedge)
    
    # Draw needle
    angle = probability * 180  # 0-180 degrees
    angle_rad = np.deg2rad(angle)
    needle_length = 0.8
    needle_x = needle_length * np.sin(angle_rad)
    needle_y = needle_length * np.cos(angle_rad)
    
    ax.plot([0, needle_x], [0, needle_y], 'k-', linewidth=3)
    ax.plot(0, 0, 'ko', markersize=8)
    
    # Add percentage text
    ax.text(0, 0.3, f'{probability*100:.1f}%', 
           ha='center', fontsize=24, fontweight='bold', color=color)
    ax.text(0, 0.1, zone, ha='center', fontsize=12, color='#2C3E50')
    
    # Add scale labels
    ax.text(0, -1.15, 'Low', ha='center', fontsize=10, color='#27AE60')
    ax.text(-1.0, -0.15, 'High', ha='center', fontsize=10, color='#C73E1D')
    ax.text(1.0, -0.15, 'Medium', ha='center', fontsize=10, color='#F18F01')
    
    plt.tight_layout()
    
    # Convert to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return img_base64

