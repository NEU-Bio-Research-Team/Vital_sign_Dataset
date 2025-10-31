"""
Chatbot panel component for AXKI dashboard.

AI explanation interface with pre-scripted responses.
"""

from dash import html, dcc
import dash_bootstrap_components as dbc


def create_chatbot_panel():
    """Create the chatbot interface panel."""
    
    return html.Div([
    # Chat header
    dbc.Card([
        dbc.CardHeader([
            html.Div([
                html.H5("🤖 Trợ Lý AI AXKI", className="mb-0"),
                dbc.Badge("Tích Hợp LLM Sắp Tới", color="info", className="ms-2")
            ], className="d-flex align-items-center")
        ], className="bg-secondary text-white")
    ], className="mb-3"),
        
        # Chat messages container - DARK MODE
        html.Div(
            id="chat-messages",
            style={
                'height': '500px',
                'overflowY': 'scroll',
                'padding': '15px',
                'backgroundColor': '#1a1a1a',  # Dark background
                'borderRadius': '5px'
            }
        ),
        
        # Typing indicator
        html.Div(id="typing-indicator", className="text-center text-muted")
        
    ])


def create_ai_message(message_text, shap_image_base64=None):
    """
    Create an AI assistant chat message bubble.
    
    Parameters:
    -----------
    message_text : str
        Message content
    shap_image_base64 : str
        Optional base64 encoded SHAP image
    
    Returns:
    --------
    html.Div : Message bubble component
    """
    
    message_content = [
        html.Div([
            html.I(className="fas fa-robot me-2"),
            html.Span("AXKI AI", className="fw-bold")
        ], className="d-flex align-items-center mb-1"),
        html.P(message_text, className="mb-0 text-light", 
               style={'whiteSpace': 'pre-line'})
    ]
    
    # Add SHAP image if provided
    if shap_image_base64:
        message_content.append(
            html.Div([
                html.Img(
                    src=f"data:image/png;base64,{shap_image_base64}",
                    style={'width': '100%', 'marginTop': '10px'}
                )
            ])
        )
    
    return dbc.Card([
        dbc.CardBody(message_content)
    ], className="mb-3", style={'maxWidth': '85%', 'marginLeft': 'auto', 
                                'backgroundColor': '#2a3a4a',  # Dark blue
                                'borderColor': '#2E86AB'})


def create_user_message(message_text):
    """
    Create a user (doctor) message bubble.
    
    Parameters:
    -----------
    message_text : str
    
    Returns:
    --------
    html.Div : Message bubble component
    """
    
    return dbc.Card([
        dbc.CardBody([
            html.Div([
                html.I(className="fas fa-user-md me-2"),
                html.Span("Bác Sĩ", className="fw-bold")
            ], className="d-flex align-items-center mb-1"),
            html.P(message_text, className="mb-0 text-light", 
                   style={'whiteSpace': 'pre-line'})
        ])
    ], className="mb-3", style={'maxWidth': '85%', 'marginRight': 'auto',
                                 'backgroundColor': '#3a2a4a',  # Dark purple
                                 'borderColor': '#6A4C93'})


def generate_chat_responses(prediction_result):
    """
    Generate pre-scripted AI chat responses based on prediction.
    
    Parameters:
    -----------
    prediction_result : dict
        Prediction results including probability, risk class, factors
    
    Returns:
    --------
    list : List of message components
    """
    
    if prediction_result is None:
        return []
    
    messages = []
    
    # Message 1: Risk assessment
    prob = prediction_result['probability_percent']
    risk_class = prediction_result['risk_class']
    
    message1 = (
        f"Dựa trên dấu hiệu sinh tồn và dữ liệu lâm sàng của bệnh nhân, "
        f"mô hình AXKI dự đoán **{prob}% nguy cơ** suy thận cấp tính sau phẫu thuật "
        f"(AKI). Phân Loại Nguy Cơ: **{risk_class}**."
    )
    messages.append(create_ai_message(message1))
    
    # Message 2: Top risk factors
    top_factors = prediction_result['top_factors']
    factor_text = "Các yếu tố nguy cơ chính:\n\n"
    for factor in top_factors[:5]:
        factor_text += f"• {factor['factor']} (+{factor['contribution']:.2f} nguy cơ)\n"
    
    messages.append(create_ai_message(factor_text))
    
    # Message 3: Clinical recommendations
    recommendations = prediction_result['recommendation']
    rec_text = "Khuyến Nghị Lâm Sàng:\n\n"
    for i, rec in enumerate(recommendations, 1):
        rec_text += f"{i}. {rec}\n"
    
    messages.append(create_ai_message(rec_text))
    
    # Message 4: Future LLM integration note
    future_msg = (
        "🔮 Cải Tiến Tương Lai: Hệ thống này sẽ tích hợp với "
        "Mô hình Ngôn ngữ Lớn (LLM) để cung cấp hướng dẫn lâm sàng "
        "cá nhân hóa và phiên hỏi đáp tương tác."
    )
    messages.append(create_ai_message(future_msg))
    
    return messages

