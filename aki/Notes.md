# Research Notes: Postoperative Acute Kidney Injury Prediction from Vital Signs

## Suggested Paper Title

**"Predicting Postoperative Acute Kidney Injury from Vital Signs: A Comparative Study of Patient-Level and Temporal Feature-Based Machine Learning Approaches"**

*Alternative titles:*
- "Machine Learning for AKI Prediction: Integrating Tabular Clinical Data with Temporal Vital Sign Patterns"
- "Explainable AI for Acute Kidney Injury Prediction: Combining Patient-Level Features with Intraoperative Temporal Dynamics"
- "AXKI: An eXplainable AI Framework for Postoperative Acute Kidney Injury Risk Assessment"

---

## 1. Research Overview

### 1.1 Objective
This research investigates machine learning approaches for predicting postoperative Acute Kidney Injury (AKI) using vital signs and clinical data from surgical patients. The study compares traditional patient-level approaches with novel temporal feature extraction methods to improve AKI prediction accuracy.

### 1.2 Clinical Background
- **AKI Definition**: KDIGO Stage I criteria (postoperative creatinine > 1.5 × preoperative creatinine)
- **Time Window**: Postoperative AKI detection within 7 days after surgery
- **Clinical Significance**: Early AKI prediction can improve patient outcomes through proactive intervention
- **Challenge**: Highly imbalanced dataset (5.26% positive class) requiring specialized handling

### 1.3 Dataset
- **Source**: VitalDB surgical patient database
- **Total Cases**: 3,989 patients
- **Features**: 43 numerical features (after preprocessing)
- **AKI Prevalence**: 210 cases (5.26%) - highly imbalanced
- **Data Types**: 
  - **Patient-level**: Demographics, preoperative labs, surgical characteristics
  - **Temporal**: Intraoperative vital signs (BP, HR, SpO2, etc.) extracted from time-series data

---

## 2. Methodology

### 2.1 Data Preprocessing Pipeline

#### 2.1.1 Data Loading
```
VitalDB API
    ↓
Load Cases Table (clinical data)
    ↓
Load Labs Table (creatinine levels)
    ↓
Merge and Filter (preop + postop cr available)
    ↓
Create AKI Labels (KDIGO Stage I)
    ↓
3,989 cases with 210 AKI (5.26%)
```

#### 2.1.2 Patient-Level Features (Baseline)
**Tabular Features (43 features)**:
- Demographics: age, sex, height, weight, BMI
- Preoperative labs: Hb, platelets, electrolytes, liver function tests
- Surgical characteristics: ASA score, emergency status, duration
- Excluded: Categorical variables (department, operation type, etc.)

**Preprocessing Steps**:
1. Remove categorical variables
2. Convert to numeric types
3. Calculate derived features (e.g., anesthesia duration)
4. Handle missing values: SimpleImputer (mean strategy)
5. Feature scaling: StandardScaler (for models requiring normalization)

#### 2.1.3 Temporal Feature Extraction (Novel Approach)

**Window-Level Feature Extraction**:
Following the `mbp_aki.ipynb` pattern from VitalDB examples, we extract temporal features from intraoperative vital signs:

**Signals Extracted**:
1. **ART_MBP** (Mean Arterial Pressure): 70.6% coverage (2,817/3,989 cases)
   - Threshold features: Percentage time below 40-80 mmHg (41 features)
   - Statistical features: mean, std, min, max, percentiles, IQR, CV
   - Clinical thresholds: time below 65/60/55 mmHg (hypotension)

2. **ART_SBP** (Systolic BP): 70.4% coverage
   - Threshold features: Percentage time below 90-140 mmHg
   - Statistical features: Full statistical summary

3. **ART_DBP** (Diastolic BP): 70.4% coverage
   - Threshold features: Percentage time below 50-90 mmHg
   - Statistical features: Full statistical summary

4. **PLETH_HR** (Heart Rate): 99.9% coverage (3,987/3,989 cases)
   - Statistical features: Full statistical summary
   - Clinical thresholds: bradycardia (<60 bpm), tachycardia (>100 bpm)

5. **PLETH_SPO2** (Oxygen Saturation): 100.0% coverage (3,988/3,989 cases)
   - Threshold features: Percentage time below 90-95% (hypoxemia)
   - Statistical features: Full statistical summary

**Total Temporal Features**: 130 features per patient

**Aggregation Strategy**: Window-level features aggregated to patient-level (mean, percentiles, thresholds) for direct comparison with baseline models.

### 2.2 Model Training and Evaluation

#### 2.2.1 Models Evaluated
1. **XGBoost** (Gradient Boosting)
   - Handles imbalanced data via `scale_pos_weight`
   - Non-linear feature interactions
   - Feature importance available

2. **Random Forest** (Ensemble Method)
   - Robust to outliers
   - Built-in feature importance
   - Class weights for imbalance

3. **Logistic Regression** (Baseline)
   - Interpretable coefficients
   - Fast training and prediction
   - Class weights for imbalance

#### 2.2.2 Three Experimental Configurations

**Configuration 1: Tabular Features Only (Baseline)**
- Input: 43 patient-level clinical features
- Purpose: Establish baseline performance
- Models: XGBoost, Random Forest, Logistic Regression

**Configuration 2: Temporal Features Only**
- Input: 130 temporal vital sign features
- Purpose: Assess predictive power of temporal patterns alone
- Models: XGBoost, Random Forest, Logistic Regression

**Configuration 3: Combined Features**
- Input: 173 features (43 tabular + 130 temporal)
- Purpose: Determine if temporal features improve baseline
- Models: XGBoost, Random Forest, Logistic Regression

#### 2.2.3 Data Splitting and Evaluation
- **Split Ratio**: 80% train / 20% test
- **Random State**: 42 (reproducibility)
- **Stratification**: Yes (maintains class imbalance ratio)
- **Metrics**:
  - ROC-AUC (primary metric for imbalanced data)
  - PR-AUC (Precision-Recall AUC - important for rare positive class)
  - Accuracy, Precision, Recall, F1-Score
  - Confusion Matrix

---

## 3. Experimental Results

### 3.1 Model Performance Comparison

#### 3.1.1 Tabular Features Only (Baseline)

| Model | ROC-AUC | PR-AUC | Accuracy | F1-Score |
|-------|---------|--------|----------|----------|
| XGBoost | 0.7577 | 0.2425 | 0.9474 | 0.1923 |
| Random Forest | 0.7308 | 0.2437 | 0.9486 | 0.1277 |
| Logistic Regression | 0.6575 | 0.2370 | 0.9511 | 0.2041 |

**Key Observations**:
- XGBoost achieves best ROC-AUC (0.7577)
- Random Forest has highest PR-AUC (0.2437)
- High accuracy (>94%) but low F1-scores due to class imbalance
- All models struggle with precision/recall trade-off

#### 3.1.2 Temporal Features Only

| Model | ROC-AUC | PR-AUC | Accuracy | F1-Score |
|-------|---------|--------|----------|----------|
| XGBoost | 0.7178 | 0.1257 | 0.9373 | 0.0385 |
| Logistic Regression | 0.7166 | 0.1030 | 0.9373 | 0.0000 |
| Random Forest | 0.6244 | 0.0867 | 0.9399 | 0.0000 |

**Key Observations**:
- Temporal features alone underperform baseline (ROC-AUC: 0.6862 vs 0.7153)
- Poor PR-AUC indicates difficulty with rare positive class
- Very low precision/recall (often 0.0) for Random Forest and Logistic Regression
- Suggests temporal features need tabular context for effective prediction

#### 3.1.3 Combined Features (Tabular + Temporal)

| Model | ROC-AUC | PR-AUC | Accuracy | F1-Score |
|-------|---------|--------|----------|----------|
| XGBoost | **0.7873** | 0.2282 | 0.9411 | 0.0784 |
| Random Forest | **0.7778** | 0.1820 | 0.9424 | 0.0000 |
| Logistic Regression | **0.7562** | 0.2026 | 0.9424 | 0.1786 |

**Key Observations**:
- **Combined features improve ROC-AUC across all models**:
  - XGBoost: +3.9% improvement (0.7577 → 0.7873)
  - Random Forest: +6.4% improvement (0.7308 → 0.7778)
  - Logistic Regression: +15.0% improvement (0.6575 → 0.7562)
- Lower variance in performance (std: 0.0159 vs 0.0518 for tabular)
- Consistent improvement suggests temporal features provide complementary information
- PR-AUC slightly decreases (-5.88%) indicating need for better imbalance handling

### 3.2 Statistical Summary by Feature Set

| Feature Set | Mean ROC-AUC | Mean PR-AUC | Mean F1-Score |
|-------------|--------------|-------------|---------------|
| Tabular Only | 0.7153 ± 0.0518 | 0.2410 ± 0.0036 | 0.1747 ± 0.0411 |
| Temporal Only | 0.6862 ± 0.0536 | 0.1051 ± 0.0196 | 0.0128 ± 0.0222 |
| Combined | **0.7738 ± 0.0159** | 0.2043 ± 0.0232 | 0.0857 ± 0.0895 |

**Key Findings**:
- **Combined features achieve highest mean ROC-AUC** (0.7738)
- **Combined features show lowest variance** (0.0159) - more consistent performance
- Temporal-only approach is insufficient alone
- PR-AUC needs improvement through better class imbalance handling

### 3.3 Best Models by Metric

- **Best ROC-AUC**: XGBoost on Combined Features (0.7873)
- **Best PR-AUC**: Random Forest on Tabular Features (0.2437)
- **Best F1-Score**: Logistic Regression on Tabular Features (0.2041)

---

## 4. Key Findings and Insights

### 4.1 Temporal Features Are Beneficial When Combined

**Evidence**:
- Combined features improve ROC-AUC by 3.9-15% depending on model
- Lowest variance suggests more stable predictions
- Consistent improvement across all three model types

**Clinical Interpretation**:
- Intraoperative vital sign patterns (especially hypotension events) provide complementary information to static patient characteristics
- Temporal features capture dynamic physiological responses during surgery
- Pattern-based features (e.g., percentage time below thresholds) align with clinical reasoning

### 4.2 Temporal Features Alone Are Insufficient

**Evidence**:
- Temporal-only ROC-AUC (0.6862) < Baseline (0.7153)
- Very poor PR-AUC (0.1051) and F1-scores
- Missing patient context (demographics, comorbidities) limits predictive power

**Implication**:
- Temporal patterns need baseline patient characteristics for context
- Clinical features (age, ASA score, preoperative labs) provide essential risk stratification
- Hybrid approach (combined) is optimal

### 4.3 Class Imbalance Remains a Challenge

**Evidence**:
- High accuracy (>94%) but low F1-scores (<0.2)
- PR-AUC decreases with combined features (-5.88%)
- Many models achieve 0.0 precision/recall

**Recommendations**:
1. Enhanced class imbalance handling:
   - SMOTE oversampling (computational cost permitting)
   - Threshold optimization for precision/recall trade-off
   - Ensemble methods combining multiple strategies
2. Cost-sensitive learning:
   - Adjust class weights dynamically
   - Use ROC-AUC for model selection, PR-AUC for threshold tuning
3. Alternative evaluation:
   - Focus on high-risk patient identification
   - Use calibration curves for probability assessment

### 4.4 Model-Specific Insights

**XGBoost**:
- Best overall performer across all configurations
- Handles feature interactions well (benefits from combined features)
- Consistent improvement with temporal features

**Random Forest**:
- Largest improvement with combined features (+6.4%)
- Robust to feature redundancy (handles 173 features well)
- Best PR-AUC on tabular-only features

**Logistic Regression**:
- Largest relative improvement (+15.0%) suggests temporal features provide linearizable patterns
- Good interpretability for clinical decision-making
- Best F1-score on tabular-only features

---

## 5. Methodology: AXKI Framework

**AXKI** (eXplainable AI and Machine learning for Acute Kidney Injury) is our proposed framework:

### 5.1 Framework Components

1. **Data Input**: 
   - Patient-level clinical data (demographics, labs, surgical characteristics)
   - Temporal intraoperative vital signs (BP, HR, SpO2)

2. **Preprocessing**:
   - Tabular: Missing value imputation, feature scaling
   - Temporal: Window-level feature extraction, aggregation to patient-level

3. **Feature Engineering**:
   - Static features: 43 clinical features
   - Temporal features: 130 vital sign-derived features (thresholds, statistics, clinical markers)

4. **ML-Based Prediction**:
   - Multiple models: XGBoost, Random Forest, Logistic Regression
   - Hyperparameter tuning with class imbalance handling
   - Ensemble potential for improved robustness

5. **XAI-Based Model Selection**:
   - SHAP (SHapley Additive exPlanations) for interpretability
   - Feature importance analysis
   - Clinical reasoning alignment

6. **Clinical Decision Support**:
   - Risk scores with confidence intervals
   - Interpretable explanations for clinicians
   - Dashboard integration for real-time monitoring

### 5.2 Workflow Diagram

```
Clinical Data (VitalDB)
    ↓
┌─────────────────────────────────────┐
│ 1. Data Loading & Preprocessing     │
│    - Load cases and labs            │
│    - Create AKI labels (KDIGO)      │
│    - Extract tabular features (43)  │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 2. Temporal Feature Extraction      │
│    - Load intraoperative signals    │
│    - Extract window-level features  │
│    - Aggregate to patient-level     │
│    - 130 temporal features          │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 3. Feature Combination              │
│    - Merge tabular + temporal       │
│    - Total: 173 features            │
│    - Train/test split (80/20)       │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 4. Model Training                   │
│    - XGBoost, RF, LR                │
│    - Class imbalance handling       │
│    - Hyperparameter tuning          │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 5. Evaluation & Interpretation      │
│    - ROC-AUC, PR-AUC, F1            │
│    - SHAP explanations              │
│    - Feature importance             │
└─────────────────────────────────────┘
    ↓
Clinical Decision Support
```

---

## 6. Notebooks and Experiments

### 6.1 Patient-Level Experiments (Pat_)

**Pat_dl_examination.ipynb**:
- Examines feasibility of deep learning for tabular data
- Compares MLP, DeepMLP, ResNet architectures
- Baseline comparison: XGBoost, Random Forest, Logistic Regression
- **Finding**: Traditional ML outperforms DL on this tabular dataset

**Pat_aki_prediction.ipynb**:
- Initial ML model implementation
- Basic training and evaluation
- Model comparison framework

**Pat_aki_pred_hyper.ipynb**:
- Comprehensive hyperparameter tuning
- GridSearchCV with cross-validation
- Best model selection and persistence

**Pat_dataset_analysis.ipynb**:
- Class imbalance investigation
- Feature distribution analysis
- Missing value analysis
- **Finding**: Severe class imbalance (5.26%) explains high accuracy but low AUC/PRC

**Pat_data_vis.ipynb**:
- Comprehensive dataset visualization
- Feature correlations
- Target distribution analysis
- Time-series signal visualization (synthetic data for demo)

**Pat_example_train.ipynb**:
- Training example using modular package
- Demonstrates hyperparameter tuning workflow
- Model evaluation and saving

**Pat_example_eval.ipynb**:
- Evaluation example
- SHAP explanation generation
- Model interpretability demonstration

### 6.2 Window-Level Experiments (Win_)

**Win_windowaki_examine.ipynb**:
- Explores availability of raw temporal vital signs in VitalDB
- Tests temporal data loading via vitaldb library
- Assesses feasibility of window-level approach
- **Finding**: 
  - Temporal signals accessible (ART_MBP: 70.6%, PLETH_HR: 99.9%, PLETH_SPO2: 100%)
  - Window-level feature extraction feasible
  - Recommended: Hybrid approach (window-level extraction, patient-level prediction)

### 6.3 Combined Experiments (Com_)

**Com_temporal_features_aki.ipynb**:
- **Main Experiment**: Extracts temporal features and compares configurations
- Loads temporal signals for all 3,989 cases
- Extracts 130 temporal features (thresholds, statistics, clinical markers)
- Trains models on three configurations:
  1. Tabular only (baseline)
  2. Temporal only
  3. Combined (tabular + temporal)
- **Key Result**: Combined features improve ROC-AUC by 3.9-15%
- **Conclusion**: Temporal features are useful when combined with tabular features

---

## 7. Visualizations and Figures

### 7.1 Performance Comparison Plots
- Bar charts comparing ROC-AUC, PR-AUC, F1-Score across feature sets
- ROC curves for each model (Tabular vs Temporal vs Combined)
- Precision-Recall curves for imbalanced data assessment

### 7.2 Dataset Characteristics
- Class distribution (highly imbalanced: 5.26% AKI)
- Feature correlation heatmaps
- Missing value patterns
- Distribution plots for key features (age, creatinine, BMI)

### 7.3 SHAP Interpretability
- Feature importance plots (SHAP values)
- Waterfall plots for individual predictions
- Summary plots for global feature importance
- Clinical decision support integration

---

## 8. Clinical Implications

### 8.1 Practical Applications
1. **Preoperative Risk Assessment**:
   - Use tabular features for baseline risk stratification
   - Identify high-risk patients before surgery

2. **Intraoperative Monitoring**:
   - Integrate temporal features for real-time risk updates
   - Alert clinicians to hypotension patterns associated with AKI

3. **Postoperative Surveillance**:
   - Combine predictions with clinical monitoring
   - Early intervention for high-risk patients

### 8.2 Model Deployment Considerations
1. **Feature Availability**:
   - Tabular features: Readily available from EHR
   - Temporal features: Require intraoperative monitoring data
   - Deployment strategy depends on available infrastructure

2. **Computational Requirements**:
   - Tabular-only models: Fast, low-resource
   - Combined models: Slightly higher computational cost
   - Temporal feature extraction: Moderate cost (can be precomputed)

3. **Interpretability**:
   - SHAP explanations critical for clinical acceptance
   - Feature importance aligns with clinical reasoning
   - Temporal thresholds (e.g., time below 65 mmHg) are clinically meaningful

---

## 9. Future Directions

### 9.1 Immediate Improvements
1. **Class Imbalance Handling**:
   - Implement SMOTE or ADASYN oversampling
   - Optimize decision thresholds for PR-AUC
   - Explore ensemble methods

2. **Feature Engineering**:
   - Additional temporal patterns (trends, volatility, events)
   - Interaction features between temporal and tabular
   - Domain-specific clinical features

3. **Model Development**:
   - Ensemble of best models
   - Deep learning for temporal sequences (LSTM/GRU)
   - Attention mechanisms for temporal feature importance

### 9.2 Advanced Research
1. **Temporal Sequence Modeling**:
   - Direct window-level prediction (not aggregated)
   - Sequence-to-sequence models
   - Early warning during surgery

2. **Multi-Task Learning**:
   - Predict AKI severity (stages I, II, III)
   - Simultaneous prediction of related outcomes
   - Transfer learning from other surgical cohorts

3. **Causal Inference**:
   - Understand causal pathways to AKI
   - Intervention recommendations
   - Counterfactual analysis

### 9.3 Clinical Integration
1. **Real-Time Systems**:
   - Integration with hospital monitoring systems
   - Alert generation for high-risk patterns
   - Dashboard for ICU/hospital units

2. **Validation Studies**:
   - External validation on different hospitals
   - Prospective validation study
   - Cost-effectiveness analysis

---

## 10. Conclusion

This research demonstrates that:

1. **Temporal features provide value** when combined with tabular clinical features (3.9-15% ROC-AUC improvement)

2. **Temporal features alone are insufficient** - they require patient-level context for effective prediction

3. **Class imbalance remains challenging** - requires specialized handling strategies (current PR-AUC: 0.20-0.24)

4. **XGBoost on combined features** achieves best ROC-AUC (0.7873), suitable for clinical decision support

5. **AXKI framework** provides interpretable, clinically-relevant predictions with SHAP explanations

**Recommendation**: Deploy combined feature models with enhanced class imbalance handling for improved precision-recall performance.

---

## 11. References and Resources

### Datasets
- VitalDB: https://vitaldb.net/
- VitalDB API Documentation: https://vitaldb.net/dataset/?query=api

### Key Papers
- KDIGO Clinical Practice Guideline for Acute Kidney Injury
- VitalDB: A Multi-Parameter Database for Research in Anesthesia
- SHAP: A Unified Approach to Interpreting Model Predictions

### Code and Tools
- vitaldb Python library: `pip install vitaldb`
- SHAP: `pip install shap`
- XGBoost, scikit-learn, pandas, numpy

---

**Last Updated**: 2024-11-01  
**Author**: Research Assistant  
**Status**: Active Research Project
