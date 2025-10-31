"""
AXKI Medical Dashboard - Main Application

Interactive medical dashboard for AKI prediction with real-time vital signs
visualization, ML model selection, and AI-powered clinical explanations.

Author: AXKI Research Team
Date: 2024
"""

import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
from datetime import datetime

# Import custom components
from components.vitals_panel import create_vitals_panel
from components.prediction_panel import create_prediction_panel, create_prediction_results
from components.chatbot_panel import create_chatbot_panel, generate_chat_responses, create_ai_message

# Import utilities
from utils.data_generator import (
    generate_synthetic_vitals, 
    generate_patient_info, 
    get_all_patient_scenarios
)
from utils.predictor import predict_aki_risk
from utils.shap_generator import generate_shap_plot

# Initialize Dash app with Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Custom CSS
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>AXKI Medical Dashboard</title>
        {%favicon%}
        {%css%}
        <style>
            body { font-family: 'Roboto', sans-serif; }
            .navbar { box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .card { border-radius: 8px; }
            .btn { border-radius: 5px; }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Application Title Bar
navbar = dbc.Navbar(
    dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H3("🩺 AXKI Medical Dashboard", className="text-white mb-0"),
                html.P("Hệ thống AI dự đoán nguy cơ suy thận cấp sau phẫu thuật", 
                      className="text-white-50 mb-0")
            ])
        ], align="center", className="g-0"),
        dbc.Row([
            dbc.Col([
                html.Small(f"Phiên: {datetime.now().strftime('%d/%m/%Y %H:%M')}", 
                          className="text-white-50")
            ])
        ], className="g-0")
    ], fluid=True),
    color="dark",
    dark=True,
    className="mb-4"
)

# Patient Scenario Selector
scenario_selector = dbc.Card([
    dbc.CardBody([
        html.H5("Chọn Tình Huống Bệnh Nhân", className="mb-3 text-light"),
        dcc.Dropdown(
            id='scenario-selector',
            options=[
                {'label': 'Bệnh nhân nguy cơ thấp', 'value': 'low_risk'},
                {'label': 'Bệnh nhân thông thường', 'value': 'normal'},
                {'label': 'Bệnh nhân nguy cơ trung bình', 'value': 'medium_risk'},
                {'label': 'Bệnh nhân nguy cơ cao', 'value': 'high_risk'}
            ],
            value='medium_risk',
            clearable=False,
            className="mb-2"
        ),
        dbc.Button("🔄 Tải Bệnh Nhân Mới", id="load-patient-btn", 
                  color="primary", size="sm", className="w-100")
    ])
], className="mb-3")

# Main layout
app.layout = dbc.Container([
    # Navbar
    navbar,
    
    # Patient selector
    scenario_selector,
    
    # Three-panel layout
    dbc.Row([
        # Left Panel: Vital Signs
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("📊 Dấu Hiệu Sinh Tồn", 
                             className="bg-primary text-white"),
                dbc.CardBody([
                    html.Div(id="vitals-display")
                ])
            ])
        ], width=5, className="pe-2"),
        
        # Center Panel: Prediction Controls
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("🔮 Dự Đoán Nguy Cơ AKI", 
                             className="bg-success text-white"),
                dbc.CardBody([
                    html.Div(id="prediction-panel-display")
                ])
            ])
        ], width=4, className="px-2"),
        
        # Right Panel: Chatbot
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("🤖 Trợ Lý AI Y Khoa", 
                             className="bg-info text-white"),
                dbc.CardBody([
                    html.Div(id="chatbot-display")
                ])
            ])
        ], width=3, className="ps-2")
    ], className="g-0"),
    
    # Hidden data store
    dcc.Store(id='vital-signs-store'),
    dcc.Store(id='patient-info-store'),
    dcc.Store(id='prediction-result-store')
    
], fluid=True, className="px-3")


# Callback: Load patient data
@app.callback(
    [Output('vital-signs-store', 'data'),
     Output('patient-info-store', 'data'),
     Output('vitals-display', 'children'),
     Output('prediction-panel-display', 'children')],
    [Input('scenario-selector', 'value'),
     Input('load-patient-btn', 'n_clicks')]
)
def load_patient_data(scenario, n_clicks):
    """Load new patient vital signs and metadata."""
    
    # Generate synthetic data
    vital_signs = generate_synthetic_vitals(n_points=2000, scenario=scenario)
    patient_info = generate_patient_info(scenario)
    
    # Create displays
    vitals_display = create_vitals_panel(vital_signs)
    prediction_display = create_prediction_panel()
    
    # Convert to JSON-serializable
    vital_signs_json = vital_signs.to_dict('records')
    vital_signs_json.append({'__index__': list(vital_signs.index)})
    
    return (
        {'data': vital_signs.to_dict('records'), 
         'index': list(vital_signs.index),
         'columns': list(vital_signs.columns)},
        patient_info,
        vitals_display,
        prediction_display
    )


# Callback: Update patient info display
@app.callback(
    [Output('patient-id', 'children'),
     Output('patient-name', 'children'),
     Output('patient-age', 'children'),
     Output('patient-sex', 'children'),
     Output('patient-surgery', 'children')],
    [Input('patient-info-store', 'modified_timestamp')],
    [State('patient-info-store', 'data')]
)
def update_patient_info(ts, patient_info):
    if patient_info is None:
        return "N/A", "N/A", "N/A", "N/A", "N/A"
    
    return (
        patient_info.get('patient_id', 'N/A'),
        patient_info.get('name', 'N/A'),
        f"{patient_info.get('age', 'N/A')} years",
        patient_info.get('sex', 'N/A'),
        patient_info.get('surgery_type', 'N/A')
    )


# Callback: Predict button
@app.callback(
    [Output('prediction-results', 'children'),
     Output('prediction-result-store', 'data'),
     Output('chatbot-display', 'children'),
     Output('loading-spinner', 'children')],
    [Input('predict-btn', 'n_clicks')],
    [State('model-selector', 'value'),
     State('vital-signs-store', 'data'),
     State('patient-info-store', 'data')]
)
def run_prediction(n_clicks, model_name, vital_signs_data, patient_info):
    """Run AKI risk prediction and update displays."""
    
    import time
    
    if n_clicks == 0 or vital_signs_data is None:
        return html.Div(), None, create_chatbot_panel(), html.Div()
    
    # Show loading spinner
    spinner = dbc.Spinner(html.Div(id="spinner-output"), color="primary")
    
    # Simulate prediction time
    time.sleep(1)
    
    # Reconstruct DataFrame
    vital_signs_df = pd.DataFrame(
        vital_signs_data['data'],
        index=vital_signs_data['index'],
        columns=vital_signs_data['columns']
    )
    
    # Run prediction
    prediction_result = predict_aki_risk(patient_info, vital_signs_df, model_name)
    
    # Generate SHAP image
    shap_image = generate_shap_plot(
        prediction_result['top_factors'], 
        prediction_result
    )
    
    # Create results display
    results_display = create_prediction_results(prediction_result)
    
    # Generate chat responses
    chat_messages = generate_chat_responses(prediction_result)
    
    # Add SHAP image to second message
    if len(chat_messages) > 1 and shap_image:
        chat_messages[1] = create_ai_message(
            "Key risk factors identified:\n\n" + 
            "\n".join([f"• {f['factor']}" for f in prediction_result['top_factors'][:5]]),
            shap_image_base64=shap_image
        )
    
    chatbot_display = html.Div(chat_messages)
    
    return (
        results_display,
        prediction_result,
        chatbot_display,
        html.Div()
    )


if __name__ == '__main__':
    print("="*70)
    print("🩺 AXKI Medical Dashboard")
    print("="*70)
    print("\n📊 Starting dashboard...")
    print("🌐 Access at: http://localhost:8050")
    print("📖 Loading patient scenarios...")
    print("✅ Dashboard ready!")
    print("\n" + "="*70)
    
    app.run(debug=True, host='0.0.0.0', port=8050)

