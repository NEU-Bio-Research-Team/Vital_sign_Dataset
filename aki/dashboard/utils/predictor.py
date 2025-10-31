"""
Mock prediction engine for AXKI dashboard.

Simulates ML model predictions based on patient vital signs and metadata.
Returns realistic probabilities and classifications.
"""

import numpy as np


def predict_aki_risk(patient_info, vital_signs_data, model_name='LogisticRegression'):
    """
    Mock prediction engine that generates realistic AKI risk predictions.
    
    Parameters:
    -----------
    patient_info : dict
        Patient metadata (age, sex, surgery type, etc.)
    vital_signs_data : pd.DataFrame
        Time-series vital signs data
    model_name : str
        Model to use: 'Traditional_AKI_Score', 'LogisticRegression', 
                     'RandomForest', 'XGBoost', 'SVM'
    
    Returns:
    --------
    dict : Prediction results
        - probability: float (0-1)
        - risk_class: str ('Low', 'Medium', 'High')
        - confidence_interval: tuple (lower, upper)
        - top_factors: list of contributing risk factors
        - model_name: str
    """
    
    # Base risk calculation from patient info
    base_risk = calculate_base_risk(patient_info, vital_signs_data)
    
    # Adjust for different models (add slight variations)
    model_adjustments = {
        'Traditional_AKI_Score': 0.0,
        'LogisticRegression': np.random.uniform(-0.05, 0.05),
        'RandomForest': np.random.uniform(-0.03, 0.08),
        'XGBoost': np.random.uniform(-0.02, 0.05),
        'SVM': np.random.uniform(-0.04, 0.04)
    }
    
    probability = np.clip(base_risk + model_adjustments.get(model_name, 0), 0, 1)
    
    # Classify risk level
    if probability < 0.25:
        risk_class = 'Low'
        confidence = 0.95
    elif probability < 0.50:
        risk_class = 'Medium'
        confidence = 0.90
    else:
        risk_class = 'High'
        confidence = 0.85
    
    # Generate top risk factors
    top_factors = identify_top_risk_factors(patient_info, vital_signs_data, probability)
    
    # Calculate confidence interval
    margin = (1 - confidence) / 2
    confidence_interval = (
        max(0, probability - margin),
        min(1, probability + margin)
    )
    
    # Model performance metrics (pretend values)
    model_metrics = get_model_metrics(model_name)
    
    return {
        'probability': round(probability, 3),
        'probability_percent': round(probability * 100, 1),
        'risk_class': risk_class,
        'confidence_interval': (
            round(confidence_interval[0], 3),
            round(confidence_interval[1], 3)
        ),
        'top_factors': top_factors,
        'model_name': model_name,
        'model_metrics': model_metrics,
        'recommendation': get_clinical_recommendation(risk_class)
    }


def calculate_base_risk(patient_info, vital_signs_data):
    """
    Calculate base risk probability based on patient factors.
    
    Uses realistic rules-based logic:
    - Age >65 increases risk
    - Hypotension increases risk
    - Elevated creatinine increases risk
    - Low SpO2 increases risk
    - Unstable vitals increase risk
    """
    
    age = patient_info.get('age', 50)
    sex = patient_info.get('sex', 'Male')
    
    # Calculate vital sign statistics
    vital_stats = {
        'mean_mbp': vital_signs_data['ART_MBP'].mean() if 'ART_MBP' in vital_signs_data.columns else 75,
        'min_mbp': vital_signs_data['ART_MBP'].min() if 'ART_MBP' in vital_signs_data.columns else 70,
        'mean_hr': vital_signs_data['PLETH_HR'].mean() if 'PLETH_HR' in vital_signs_data.columns else 75,
        'mean_spo2': vital_signs_data['PLETH_SPO2'].mean() if 'PLETH_SPO2' in vital_signs_data.columns else 98,
        'mean_etco2': vital_signs_data['ECO2_ETCO2'].mean() if 'ECO2_ETCO2' in vital_signs_data.columns else 35,
    }
    
    # Simulate lab values based on vital signs
    simulated_creatinine = simulate_creatinine(vital_stats, age)
    
    base_risk = 0.15  # Baseline 15% risk
    
    # Age factor
    if age >= 75:
        base_risk += 0.25
    elif age >= 65:
        base_risk += 0.15
    elif age >= 55:
        base_risk += 0.08
    
    # BP factor (hypotension)
    if vital_stats['min_mbp'] < 70:
        base_risk += 0.20
    elif vital_stats['min_mbp'] < 75:
        base_risk += 0.10
    
    # HR factor (tachycardia)
    if vital_stats['mean_hr'] > 100:
        base_risk += 0.08
    elif vital_stats['mean_hr'] > 90:
        base_risk += 0.04
    
    # SpO2 factor
    if vital_stats['mean_spo2'] < 95:
        base_risk += 0.15
    elif vital_stats['mean_spo2'] < 97:
        base_risk += 0.05
    
    # Creatinine factor
    if simulated_creatinine > 1.5:
        base_risk += 0.30
    elif simulated_creatinine > 1.2:
        base_risk += 0.15
    elif simulated_creatinine > 1.0:
        base_risk += 0.08
    
    # Surgery type factor
    surgery = patient_info.get('surgery_type', '').lower()
    if 'emergency' in surgery or 'major' in surgery:
        base_risk += 0.15
    elif 'transplant' in surgery:
        base_risk += 0.20
    elif 'oncology' in surgery:
        base_risk += 0.12
    
    # Comorbidities factor
    comorbidities = patient_info.get('comorbidities', [])
    if 'Chronic Kidney Disease' in comorbidities:
        base_risk += 0.25
    elif 'Diabetes' in comorbidities or 'Hypertension' in comorbidities:
        base_risk += 0.10
    
    # Clip to reasonable range
    base_risk = np.clip(base_risk, 0, 1)
    
    return base_risk


def simulate_creatinine(vital_stats, age):
    """Simulate creatinine level based on vitals and age."""
    base_cr = 0.8 + (age - 50) * 0.01
    
    # Adjust for hypotension
    if vital_stats['min_mbp'] < 70:
        base_cr += 0.2
    elif vital_stats['min_mbp'] < 75:
        base_cr += 0.1
    
    # Add some randomness
    base_cr += np.random.uniform(-0.1, 0.1)
    
    return np.clip(base_cr, 0.6, 2.5)


def identify_top_risk_factors(patient_info, vital_signs_data, probability):
    """Identify top contributing risk factors for explanation."""
    
    import pandas as pd
    
    factors = []
    
    age = patient_info.get('age', 50)
    if age >= 65:
        factors.append({
            'factor': f'Age >65 years ({age})',
            'contribution': min(0.30, probability * 0.3)
        })
    
    if 'ART_MBP' in vital_signs_data.columns:
        min_mbp = vital_signs_data['ART_MBP'].min()
        if min_mbp < 75:
            factors.append({
                'factor': f'Low mean BP ({min_mbp:.0f} mmHg)',
                'contribution': min(0.25, probability * 0.25)
            })
    
    if 'PLETH_SPO2' in vital_signs_data.columns:
        mean_spo2 = vital_signs_data['PLETH_SPO2'].mean()
        if mean_spo2 < 96:
            factors.append({
                'factor': f'Low SpO2 ({mean_spo2:.0f}%)',
                'contribution': min(0.20, probability * 0.2)
            })
    
    comorbidities = patient_info.get('comorbidities', [])
    if 'Chronic Kidney Disease' in comorbidities:
        factors.append({
            'factor': 'Pre-existing CKD',
            'contribution': min(0.30, probability * 0.3)
        })
    
    # Add creatinine factor
    sim_cr = simulate_creatinine(
        {
            'min_mbp': vital_signs_data.get('ART_MBP', pd.Series([75])).min(),
            'mean_spo2': vital_signs_data.get('PLETH_SPO2', pd.Series([98])).mean()
        },
        age
    )
    if sim_cr > 1.2:
        factors.append({
            'factor': f'Elevated creatinine ({sim_cr:.2f} mg/dL)',
            'contribution': min(0.25, probability * 0.25)
        })
    
    # Sort by contribution
    factors = sorted(factors, key=lambda x: x['contribution'], reverse=True)[:5]
    
    return factors


def get_model_metrics(model_name):
    """Return pretend model performance metrics."""
    
    metrics = {
        'Traditional_AKI_Score': {
            'accuracy': 0.72,
            'auc': 0.68,
            'sensitivity': 0.65,
            'specificity': 0.74
        },
        'LogisticRegression': {
            'accuracy': 0.84,
            'auc': 0.89,
            'sensitivity': 0.82,
            'specificity': 0.85
        },
        'RandomForest': {
            'accuracy': 0.87,
            'auc': 0.92,
            'sensitivity': 0.85,
            'specificity': 0.88
        },
        'XGBoost': {
            'accuracy': 0.89,
            'auc': 0.94,
            'sensitivity': 0.87,
            'specificity': 0.90
        },
        'SVM': {
            'accuracy': 0.85,
            'auc': 0.90,
            'sensitivity': 0.83,
            'specificity': 0.86
        }
    }
    
    return metrics.get(model_name, metrics['LogisticRegression'])


def get_clinical_recommendation(risk_class):
    """Get clinical recommendations based on risk class."""
    
    recommendations = {
        'Low': [
            'Continue routine postoperative monitoring',
            'Standard creatinine checks (24h, 48h)',
            'Maintain adequate hydration'
        ],
        'Medium': [
            'Monitor renal function closely (q6h creatinine)',
            'Optimize fluid balance',
            'Avoid nephrotoxic medications',
            'Consider daily nephrology consult'
        ],
        'High': [
            'Intensive renal monitoring (q4h creatinine)',
            'Immediate nephrology consultation',
            'Aggressive fluid resuscitation',
            'Avoid all nephrotoxic agents',
            'Consider ICU transfer for close monitoring'
        ]
    }
    
    return recommendations.get(risk_class, recommendations['Low'])


def compare_models(patient_info, vital_signs_data):
    """Run all models and return comparison."""
    
    models = [
        'Traditional_AKI_Score',
        'LogisticRegression',
        'RandomForest',
        'XGBoost',
        'SVM'
    ]
    
    results = {}
    for model in models:
        results[model] = predict_aki_risk(patient_info, vital_signs_data, model)
    
    return results

