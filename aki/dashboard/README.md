# 🩺 AXKI Medical Dashboard

AI-Powered Acute Kidney Injury (AKI) Risk Prediction System

## Overview

The AXKI Medical Dashboard is a professional medical decision-support system that simulates clinical decision-making for postoperative AKI risk assessment. The dashboard combines real-time vital signs visualization, multiple ML model predictions, and AI-powered clinical explanations.

### Key Features

- **📊 Real-Time Vital Signs Monitoring**: Time-series visualization of patient vital signs (BP, HR, SpO2, CO2, etc.)
- **🔮 ML-Based Predictions**: Compare 5 different models:
  - Traditional AKI Score (KDIGO)
  - Logistic Regression
  - Random Forest
  - XGBoost
  - SVM
- **🤖 AI Clinical Assistant**: Intelligent chatbot explaining predictions with SHAP interpretability
- **🎨 Medical-Grade UI**: Professional hospital system aesthetic
- **📈 Model Interpretability**: SHAP waterfall plots showing top risk factors

## Architecture

```
dashboard/
├── app.py                    # Main Dash application
├── components/               # Dashboard components
│   ├── vitals_panel.py      # Vital signs visualization
│   ├── prediction_panel.py  # Prediction controls & results
│   └── chatbot_panel.py     # AI explanation interface
├── utils/                    # Utilities
│   ├── data_generator.py    # Synthetic patient data
│   ├── predictor.py         # Mock prediction engine
│   └── shap_generator.py    # SHAP plot generation
├── assets/
│   └── styles.css          # Custom medical styling
└── requirements_dashboard.txt
```

## Installation

### 1. Install Dependencies

```bash
cd dashboard
pip install -r requirements_dashboard.txt
```

Or if you're in the main project:

```bash
pip install -r requirements.txt
pip install -r dashboard/requirements_dashboard.txt
```

### 2. Run the Dashboard

```bash
cd dashboard
python app.py
```

Access at: **http://localhost:8050**

## Usage

### 1. Select Patient Scenario

Choose from 4 pre-configured patient scenarios:
- **Low Risk**: Healthy patient, stable vitals
- **Normal**: Standard surgical patient
- **Medium Risk**: Older patient, some instability
- **High Risk**: Elderly patient, unstable vitals

### 2. View Vital Signs

The left panel displays real-time vital signs:
- Systolic/Diastolic BP
- Heart Rate
- SpO2 (Oxygen saturation)
- End-tidal CO2
- Mean BP
- Respiratory Rate

### 3. Predict AKI Risk

1. Select a model from the dropdown
2. Click "🔮 Predict AKI Risk"
3. View the risk probability and classification

### 4. Review AI Explanation

After prediction, the chatbot automatically:
- Summarizes the risk assessment
- Lists top contributing risk factors
- Provides clinical recommendations
- Displays SHAP interpretability plot

## Patient Scenarios

All scenarios use synthetic vital signs data for demonstration purposes. Each scenario has different:
- Age profile
- Vital sign patterns
- Expected AKI risk probabilities

## Future Enhancements

- 🔮 **LLM Integration**: Replace pre-scripted responses with actual LLM (GPT-4, Claude, etc.)
- 📊 **Real Patient Data**: Connect to actual VitalDB API
- 💾 **Export Reports**: PDF export with charts and recommendations
- 🔄 **Model Comparison**: Side-by-side comparison of all models
- 📱 **Mobile Support**: Responsive design for tablet use

## Technical Details

### Technology Stack

- **Framework**: Dash (Plotly)
- **Visualization**: Plotly
- **Styling**: Bootstrap Components
- **Backend**: Python 3.x
- **ML**: scikit-learn, XGBoost (simulated)

### Color Scheme

Uses AXKI flowchart color palette:
- `#2E86AB` - Ocean Blue (primary signals)
- `#06A77D` - Medical Teal (processing)
- `#C73E1D` - Clinical Red (alerts)
- `#F18F01` - Warm Orange (medium risk)
- `#2C3E50` - Dark Blue-Gray (text)

### Model Performance

Mock metrics based on real ML model performance:

| Model | Accuracy | AUC | Sensitivity | Specificity |
|-------|----------|-----|-------------|-------------|
| Traditional AKI Score | 0.72 | 0.68 | 0.65 | 0.74 |
| Logistic Regression | 0.84 | 0.89 | 0.82 | 0.85 |
| Random Forest | 0.87 | 0.92 | 0.85 | 0.88 |
| XGBoost | 0.89 | 0.94 | 0.87 | 0.90 |
| SVM | 0.85 | 0.90 | 0.83 | 0.86 |

## Screenshots

*(Screenshots to be added after first run)*

## License

Part of the AXKI research project.

## Contact

For questions or issues, please contact the AXKI research team.

