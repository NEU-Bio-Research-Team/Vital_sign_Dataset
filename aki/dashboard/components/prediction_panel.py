"""
Prediction panel component for AXKI dashboard.

Contains model selection, prediction controls, and results display.
"""

from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go


def create_prediction_panel():
    """Create the prediction control and results panel."""
    
    return dbc.Container([
        # Patient Info Card
        dbc.Card([
            dbc.CardHeader("Thông Tin Bệnh Nhân", className="bg-primary text-white"),
            dbc.CardBody([
                html.H6("ID Bệnh Nhân:", className="text-muted mb-1"),
                html.P(id="patient-id", className="h5"),
                html.Hr(),
                html.H6("Tên:", className="text-muted mb-1"),
                html.P(id="patient-name", className="mb-0"),
                html.H6("Tuổi:", className="text-muted mb-1 mt-2"),
                html.P(id="patient-age", className="mb-0"),
                html.H6("Giới Tính:", className="text-muted mb-1 mt-2"),
                html.P(id="patient-sex", className="mb-0"),
                html.H6("Loại Phẫu Thuật:", className="text-muted mb-1 mt-2"),
                html.P(id="patient-surgery", className="mb-0 small")
            ])
        ], className="mb-3"),
        
        # Model Selection
        dbc.Card([
            dbc.CardHeader("Chọn Mô Hình Dự Đoán", className="bg-success text-white"),
            dbc.CardBody([
                dcc.Dropdown(
                    id='model-selector',
                    options=[
                        {'label': 'Điểm AKI Truyền Thống (KDIGO)', 'value': 'Traditional_AKI_Score'},
                        {'label': 'AXKI - Hồi Quy Logistic', 'value': 'LogisticRegression'},
                        {'label': 'AXKI - Random Forest', 'value': 'RandomForest'},
                        {'label': 'AXKI - XGBoost', 'value': 'XGBoost'},
                        {'label': 'AXKI - SVM', 'value': 'SVM'}
                    ],
                    value='XGBoost',
                    clearable=False,
                    className="mb-3"
                ),
                html.Div(id="model-info", className="text-muted small mb-3"),
                dbc.Button(
                    "🔮 Dự Đoán Nguy Cơ AKI",
                    id="predict-btn",
                    color="danger",
                    size="lg",
                    className="w-100",
                    disabled=False
                )
            ])
        ], className="mb-3"),
        
        # Loading Spinner
        html.Div(id="loading-spinner", className="text-center mb-3"),
        
        # Results Display
        html.Div(id="prediction-results"),
        
        # Model Comparison
        html.Div(id="model-comparison")
        
    ], fluid=True)


def create_prediction_results(prediction_result):
    """Create results display from prediction."""
    
    if prediction_result is None:
        return html.Div()
    
    probability = prediction_result['probability_percent']
    risk_class = prediction_result['risk_class']
    
    # Risk badge color
    badge_colors = {
        'Low': 'success',
        'Medium': 'warning',
        'High': 'danger'
    }
    
    badge_color = badge_colors.get(risk_class, 'secondary')
    
    # Confidence interval
    ci = prediction_result['confidence_interval']
    
    return dbc.Card([
        dbc.CardHeader(
            html.H5("🩺 Kết Quả Dự Đoán", className="mb-0"),
            className="bg-info text-white"
        ),
        dbc.CardBody([
            # Large probability display
            html.Div([
                html.H2(f"{probability}%", 
                        className=f"text-{badge_color} text-center mb-2"),
                html.P(f"Xác Suất Nguy Cơ AKI", 
                      className="text-center text-muted mb-3")
            ], className="text-center"),
            
            # Risk classification
            dbc.Badge(
                f"Mức Nguy Cơ: {risk_class}",
                color=badge_color,
                className="mb-3 p-2",
                style={'fontSize': '14px'}
            ),
            
            html.Hr(),
            
            # Confidence interval
            html.Div([
                html.H6("Khoảng Tin Cậy", className="text-muted mb-1"),
                html.P(f"{ci[0]*100:.1f}% - {ci[1]*100:.1f}%", 
                      className="mb-0 small")
            ]),
            
            # Top risk factors
            html.Hr(),
            html.H6("Các Yếu Tố Nguy Cơ", className="text-muted mb-2"),
            html.Ul([
                html.Li(factor['factor'], className="small")
                for factor in prediction_result['top_factors'][:3]
            ]),
            
            # Model metrics
            html.Hr(),
            html.H6("Hiệu Suất Mô Hình", className="text-muted mb-2"),
            html.Div([
                html.P(f"Độ Chính Xác: {prediction_result['model_metrics']['accuracy']:.1%}", 
                      className="small mb-1"),
                html.P(f"AUC-ROC: {prediction_result['model_metrics']['auc']:.2f}", 
                      className="small mb-0")
            ])
        ])
    ], className="mt-3")

