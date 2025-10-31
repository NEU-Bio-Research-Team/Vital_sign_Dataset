# 🚀 Quick Start Guide - AXKI Medical Dashboard

## Start the Dashboard

```bash
cd dashboard
python app.py
```

Then open your browser to: **http://localhost:8050**

## What You'll See

### Three-Panel Layout

1. **Left Panel (40%)**: Patient vital signs time-series
   - Real-time monitoring of: BP, HR, SpO2, CO2, Resp Rate
   - Interactive Plotly charts
   - Medical color scheme

2. **Center Panel (30%)**: AKI Risk Prediction
   - Patient information
   - Model selector (5 ML models)
   - Predict button → Get risk percentage
   - Results with risk classification

3. **Right Panel (30%)**: AI Clinical Assistant
   - Chat interface
   - Auto-populated AI responses
   - SHAP interpretability plots
   - Clinical recommendations

## Try These Scenarios

1. **Select Different Patients**: Click scenario dropdown to try:
   - Low Risk (healthy, stable)
   - Normal (standard patient)
   - Medium Risk (elderly, some instability)
   - High Risk (critical, unstable)

2. **Run Predictions**: 
   - Select any model from dropdown
   - Click "🔮 Predict AKI Risk"
   - Watch the AI chatbot explain the results

3. **Load New Patient**: 
   - Click "🔄 Load New Patient"
   - See different vital sign patterns

## Features

✅ 5 ML models (LR, RF, XGBoost, SVM) + Traditional AKI Score  
✅ Real-time vital signs visualization  
✅ AI chatbot with pre-scripted explanations  
✅ SHAP interpretability (waterfall plots)  
✅ Professional medical UI  
✅ Mock predictions with realistic probabilities  
✅ Clinical recommendations per risk level  

## Future: LLM Integration

The chatbot is currently pre-scripted but designed for easy LLM integration. The UI shows "LLM Integration Coming" badge.

---

**Enjoy exploring the AXKI Dashboard! 🩺**

