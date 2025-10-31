"""
Synthetic vital signs data generator for AXKI dashboard.

Adapted from data_vis.ipynb to generate realistic patient scenarios.
"""

import pandas as pd
import numpy as np


def generate_synthetic_vitals(n_points=2000, mean_values=None, scenario='normal'):
    """
    Generate synthetic vital signs time-series data for visualization.
    
    Parameters:
    -----------
    n_points : int
        Number of time points to generate (default: 2000)
    mean_values : dict
        Dictionary of mean values for each vital sign
    scenario : str
        Patient scenario: 'low_risk', 'normal', 'medium_risk', 'high_risk'
    
    Returns:
    --------
    pd.DataFrame : Synthetic time-series vital signs data
    """
    
    # Scenario-specific vital sign configs
    scenarios = {
        'low_risk': {
            'ART_SBP': {'mean': 115, 'std': 10, 'min': 95, 'max': 140},
            'ART_DBP': {'mean': 65, 'std': 8, 'min': 50, 'max': 85},
            'PLETH_HR': {'mean': 68, 'std': 10, 'min': 50, 'max': 90},
            'PLETH_SPO2': {'mean': 99, 'std': 1, 'min': 97, 'max': 100},
            'ECO2_ETCO2': {'mean': 32, 'std': 3, 'min': 25, 'max': 40},
            'ART_MBP': {'mean': 80, 'std': 8, 'min': 65, 'max': 100},
            'RESP_RR': {'mean': 12, 'std': 2, 'min': 10, 'max': 16},
            'TEMP_TEMP': {'mean': 36.5, 'std': 0.3, 'min': 36, 'max': 37}
        },
        'normal': {
            'ART_SBP': {'mean': 120, 'std': 15, 'min': 80, 'max': 180},
            'ART_DBP': {'mean': 70, 'std': 10, 'min': 40, 'max': 120},
            'PLETH_HR': {'mean': 75, 'std': 15, 'min': 40, 'max': 150},
            'PLETH_SPO2': {'mean': 98, 'std': 2, 'min': 85, 'max': 100},
            'ECO2_ETCO2': {'mean': 35, 'std': 5, 'min': 20, 'max': 60},
            'ART_MBP': {'mean': 85, 'std': 12, 'min': 50, 'max': 130},
            'RESP_RR': {'mean': 12, 'std': 3, 'min': 8, 'max': 30},
            'TEMP_TEMP': {'mean': 36.5, 'std': 0.5, 'min': 35, 'max': 38.5}
        },
        'medium_risk': {
            'ART_SBP': {'mean': 110, 'std': 18, 'min': 70, 'max': 180},
            'ART_DBP': {'mean': 62, 'std': 12, 'min': 40, 'max': 100},
            'PLETH_HR': {'mean': 85, 'std': 18, 'min': 55, 'max': 160},
            'PLETH_SPO2': {'mean': 96, 'std': 3, 'min': 88, 'max': 99},
            'ECO2_ETCO2': {'mean': 38, 'std': 6, 'min': 22, 'max': 55},
            'ART_MBP': {'mean': 75, 'std': 15, 'min': 50, 'max': 125},
            'RESP_RR': {'mean': 16, 'std': 4, 'min': 10, 'max': 32},
            'TEMP_TEMP': {'mean': 36.8, 'std': 0.6, 'min': 35.5, 'max': 38}
        },
        'high_risk': {
            'ART_SBP': {'mean': 100, 'std': 20, 'min': 60, 'max': 160},
            'ART_DBP': {'mean': 55, 'std': 15, 'min': 35, 'max': 95},
            'PLETH_HR': {'mean': 95, 'std': 20, 'min': 60, 'max': 170},
            'PLETH_SPO2': {'mean': 94, 'std': 4, 'min': 85, 'max': 98},
            'ECO2_ETCO2': {'mean': 42, 'std': 8, 'min': 25, 'max': 60},
            'ART_MBP': {'mean': 65, 'std': 18, 'min': 45, 'max': 115},
            'RESP_RR': {'mean': 20, 'std': 5, 'min': 12, 'max': 35},
            'TEMP_TEMP': {'mean': 37.2, 'std': 0.7, 'min': 36, 'max': 38.5}
        }
    }
    
    # Use scenario config or custom mean_values
    if mean_values is None:
        vital_signs_config = scenarios.get(scenario, scenarios['normal'])
    else:
        vital_signs_config = mean_values
    
    df_vitals = pd.DataFrame()
    
    # Generate time axis (in minutes)
    df_vitals['time'] = np.linspace(0, n_points // 10, n_points)  # Convert to minutes
    
    # Generate each vital sign with realistic patterns
    for vital_sign, config in vital_signs_config.items():
        # Create base signal with trend and noise
        trend = np.linspace(0, np.random.uniform(-2, 2), n_points)
        noise = np.random.normal(0, config['std'] * 0.3, n_points)
        
        # Add periodic variations (simulate physiological patterns)
        period = np.random.uniform(50, 200)
        periodic = config['std'] * 0.5 * np.sin(2 * np.pi * df_vitals.index / period)
        
        # Generate signal
        signal = config['mean'] + trend + periodic + noise
        
        # Add occasional spikes/dips (simulate events)
        spike_prob = 0.02
        spikes = np.random.choice([-1, 0, 1], size=n_points, 
                                  p=[spike_prob, 1-2*spike_prob, spike_prob])
        signal += spikes * config['std'] * 2
        
        # Clip to realistic ranges
        signal = np.clip(signal, config['min'], config['max'])
        
        df_vitals[vital_sign] = signal
    
    return df_vitals


def generate_patient_info(scenario='normal'):
    """
    Generate realistic patient metadata for demo purposes.
    
    Parameters:
    -----------
    scenario : str
        Patient scenario
    
    Returns:
    --------
    dict : Patient information
    """
    
    import random
    
    # Different patient profiles for different scenarios (Vietnamese names)
    profiles = {
        'low_risk': {
            'names': ['Nguyễn Văn Hải', 'Trần Thị Mai', 'Lê Văn Nam', 'Phạm Thị Hương', 'Hoàng Văn Đức'],
            'ages': range(25, 50),
            'sexes': ['Nam', 'Nữ'],
            'surgeries': ['Phẫu thuật nội soi túi mật', 'Phẫu thuật nội soi khớp gối', 'Sửa thoát vị bẹn']
        },
        'normal': {
            'names': ['Đỗ Văn Long', 'Bùi Thị Lan', 'Vũ Văn Hùng', 'Ngô Thị Dung'],
            'ages': range(35, 65),
            'sexes': ['Nam', 'Nữ'],
            'surgeries': ['Phẫu thuật nội soi cắt ruột thừa', 'Cắt đại tràng', 'Cắt túi mật']
        },
        'medium_risk': {
            'names': ['Đặng Văn Minh', 'Lý Thị Hoa', 'Phan Văn Tuấn', 'Chu Thị Mai'],
            'ages': range(55, 75),
            'sexes': ['Nam', 'Nữ'],
            'surgeries': ['Phẫu thuật ổ bụng lớn', 'Phẫu thuật tim', 'Bypass mạch máu']
        },
        'high_risk': {
            'names': ['Trịnh Văn Sơn', 'Lưu Thị Bình', 'Đinh Văn Tài', 'Võ Thị Anh'],
            'ages': range(65, 85),
            'sexes': ['Nam', 'Nữ'],
            'surgeries': ['Mở bụng cấp cứu', 'Phẫu thuật ghép tạng', 'Cắt khối ung thư']
        }
    }
    
    profile = profiles.get(scenario, profiles['normal'])
    
    return {
        'patient_id': f'P{random.randint(10000, 99999)}',
        'name': random.choice(profile['names']),
        'age': random.choice(profile['ages']),
        'sex': random.choice(profile['sexes']),
        'surgery_type': random.choice(profile['surgeries']),
        'surgery_duration_min': random.randint(45, 180),
        'bmi': round(random.uniform(20, 35), 1),
        'comorbidities': random.choice([
            ['None'],
            ['Hypertension'],
            ['Diabetes', 'Hypertension'],
            ['Coronary Artery Disease'],
            ['Chronic Kidney Disease', 'Diabetes', 'Hypertension']
        ])
    }


def get_all_patient_scenarios():
    """
    Get list of all available patient scenarios.
    
    Returns:
    --------
    list : Available scenarios with descriptions
    """
    return [
        {'id': 'low_risk', 'name': 'Low Risk Patient', 'description': 'Healthy patient, stable vitals'},
        {'id': 'normal', 'name': 'Normal Patient', 'description': 'Standard surgical patient'},
        {'id': 'medium_risk', 'name': 'Medium Risk Patient', 'description': 'Older patient, some instability'},
        {'id': 'high_risk', 'name': 'High Risk Patient', 'description': 'Elderly patient, unstable vitals'}
    ]

