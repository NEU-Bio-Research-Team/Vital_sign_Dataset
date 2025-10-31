"""
Vital signs panel component for AXKI dashboard.

Displays time-series vital signs visualization using Plotly.
"""

from dash import dcc
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_vitals_panel(vital_signs_data):
    """
    Create the vital signs visualization panel.
    
    Parameters:
    -----------
    vital_signs_data : pd.DataFrame
        Time-series vital signs data
    
    Returns:
    --------
    dcc.Graph : Dash graph component
    """
    
    if vital_signs_data is None or len(vital_signs_data) == 0:
        return dcc.Graph(figure={
            'data': [],
            'layout': go.Layout(
                title='No vital signs data available',
                paper_bgcolor='#2d2d2d',
                plot_bgcolor='#1a1a1a',
                font={'color': '#e8e8e8'}
            )
        })
    
    # Vital signs to display with their colors (from flowchart)
    vital_configs = {
        'ART_SBP': {'color': '#C73E1D', 'unit': 'mmHg', 'name': 'Systolic BP'},
        'ART_DBP': {'color': '#F18F01', 'unit': 'mmHg', 'name': 'Diastolic BP'},
        'PLETH_HR': {'color': '#2E86AB', 'unit': 'bpm', 'name': 'Heart Rate'},
        'PLETH_SPO2': {'color': '#06A77D', 'unit': '%', 'name': 'SpO2'},
        'ECO2_ETCO2': {'color': '#6A4C93', 'unit': 'mmHg', 'name': 'End-tidal CO2'},
        'ART_MBP': {'color': '#3498DB', 'unit': 'mmHg', 'name': 'Mean BP'},
        'RESP_RR': {'color': '#8E44AD', 'unit': '/min', 'name': 'Resp Rate'},
        'TEMP_TEMP': {'color': '#C73E1D', 'unit': '°C', 'name': 'Temperature'}
    }
    
    # Filter to available vital signs
    available_signs = [vs for vs in vital_configs.keys() if vs in vital_signs_data.columns]
    
    if len(available_signs) == 0:
        return dcc.Graph(figure={
            'data': [],
            'layout': go.Layout(
                title='No vital signs data available',
                paper_bgcolor='#2d2d2d',
                plot_bgcolor='#1a1a1a',
                font={'color': '#e8e8e8'}
            )
        })
    
    # Select first 6 for visualization
    selected_signs = available_signs[:6]
    
    # Create subplots
    rows = len(selected_signs)
    fig = make_subplots(
        rows=rows, cols=1,
        subplot_titles=[vital_configs[vs]['name'] for vs in selected_signs],
        vertical_spacing=0.08,
        shared_xaxes=True
    )
    
    # Add traces for each vital sign
    for idx, vital_sign in enumerate(selected_signs, 1):
        config = vital_configs[vital_sign]
        
        # Get data
        time_data = vital_signs_data.get('time', vital_signs_data.index)
        vital_data = vital_signs_data[vital_sign]
        
        # Create trace
        trace = go.Scatter(
            x=time_data,
            y=vital_data,
            mode='lines',
            name=config['name'],
            line=dict(color=config['color'], width=2),
            hovertemplate=f'<b>{config["name"]}</b><br>' +
                         f'Time: %{{x:.1f}} min<br>' +
                         f'Value: %{{y:.1f}} {config["unit"]}<extra></extra>',
            showlegend=True
        )
        
        fig.add_trace(trace, row=idx, col=1)
        
        # Add horizontal reference line (mean) - DARK MODE
        mean_val = vital_data.mean()
        fig.add_hline(
            y=mean_val, 
            line_dash="dash", 
            line_color='#666666',  # Lighter gray for dark mode
            opacity=0.6,
            row=idx, col=1
        )
    
    # Update layout - DARK MODE
    fig.update_layout(
        height=200 * rows,
        paper_bgcolor='#2d2d2d',  # Dark card background
        plot_bgcolor='#1a1a1a',   # Dark plot background
        title={
            'text': '📊 Giám Sát Dấu Hiệu Sinh Tồn',
            'font': {'size': 18, 'color': '#e8e8e8'}  # Light text
        },
        hovermode='closest',
        showlegend=False,
        font={'color': '#e8e8e8'}  # Light text for all fonts
    )
    
    # Update axes - DARK MODE
    fig.update_xaxes(
        title_text='', 
        showgrid=True, 
        gridcolor='#444444',  # Darker grid in dark mode
        gridwidth=1, 
        zeroline=False,
        color='#e8e8e8'  # Light axis text
    )
    fig.update_yaxes(
        title_text='', 
        showgrid=True, 
        gridcolor='#444444',  # Darker grid in dark mode
        gridwidth=1, 
        zeroline=False,
        color='#e8e8e8'  # Light axis text
    )
    
    return dcc.Graph(
        figure=fig,
        id='vitals-graph',
        config={
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['lasso2d', 'select2d']
        }
    )

