# OUTLINE FOR "PROPOSED METHOD" SECTION
## AXKI: Explainable AI Scoring System for Acute Kidney Injury Risk Assessment

---

## SECTION STRUCTURE

### **Figure X: Overview of the AXKI method**
_[Insert flowchart here]_

---

### **3.1. Method Overview**

#### **3.1.1. Objectives and Scope**
- Overall description of the AXKI system
- Objective: Predict postoperative acute kidney injury risk
- Scope: Using clinical vital signs from VitalDB
- Features: Integration of Machine Learning and Explainable AI (XAI)

#### **3.1.2. Overall Architecture**
- Description of processing flow from input to output
- Explanation of each step's role in the pipeline
- Relationship between components

---

### **3.2. Input Data**

#### **3.2.1. Data Source**
- Description of VitalDB dataset
- Number of cases and collection period
- Characteristics of clinical vital signs

#### **3.2.2. Clinical Vital Signs Used**
- **ECG**: Heart rate, rhythm abnormalities
- **Plethysmography (PPG)**:
  - Pulse rate (PLETH_HR)
  - Oxygen saturation (PLETH_SPO2)
- **Arterial Pressure:**
  - Systolic blood pressure (ART_SBP)
  - Diastolic blood pressure (ART_DBP)
- **Capnography (ECO2):**
  - End-tidal CO2 (ECO2_ETCO2)
- **Surgical Data:**
  - Patient demographics (age, sex, BMI)
  - Surgery information (type, duration)
  - Blood tests (pre-op and post-op creatinine)

#### **3.2.3. AKI Definition**
- KDIGO Stage I criteria
- Formula: `AKI = (postop_cr > preop_cr × 1.5)`
- Rationale for this definition

---

### **3.3. Data Preprocessing**

#### **3.3.1. Data Merging and Cleaning**
- Merge data from multiple sources (cases, labs, vitals)
- Handle categorical variables
- Remove irrelevant features (caseid, subjectid)

#### **3.3.2. Class Balancing**
- **Problem:** Severe class imbalance (~18:1)
- **Solution:**
  - Use StratifiedKFold to ensure uniform distribution
  - Apply SMOTE/undersampling if needed
- Importance of class balancing

#### **3.3.3. Data Splitting**
- Train/test ratio: 80/20
- Stratified splitting to maintain AKI ratio
- Cross-validation with StratifiedKFold (5-fold)

#### **3.3.4. Missing Value Imputation**
- **Method:** Mean imputation for numeric features
- **Rationale:** Preserve dataset size
- Applied to Random Forest and XGBoost

#### **3.3.5. Data Standardization**
- Use StandardScaler
- Applied to Logistic Regression and SVM
- Formula and rationale

---

### **3.4. Machine Learning Models**

#### **3.4.1. Model Selection**
Explanation of using 4 ML models:

**a) Logistic Regression**
- Simple, interpretable model
- Uses standardized data
- Baseline model

**b) Random Forest**
- Ensemble learning with multiple decision trees
- Handles non-linear relationships well
- Robust to outliers
- Uses imputed data

**c) XGBoost**
- Powerful gradient boosting
- High performance on structured data
- Uses imputed data

**d) Support Vector Machine (SVM)**
- Good classification with RBF kernel
- Effective with high-dimensional data
- Uses standardized data

#### **3.4.2. Model Training**
- **Hyperparameter Tuning:** GridSearchCV
- **Validation:** StratifiedKFold (5-fold)
- **Selection Metric:** ROC-AUC
- Hyperparameters tuned for each model

#### **3.4.3. Model Evaluation**
- **Key Metrics:**
  - ROC-AUC (Area Under the ROC Curve)
  - AUPRC (Area Under the Precision-Recall Curve)
  - Accuracy, Precision, Recall, F1-Score
  - PPV, NPV, Specificity
- Performance comparison between models
- Best model selection based on ROC-AUC

---

### **3.5. Model Explainability with XAI**

#### **3.5.1. Importance of XAI in Medicine**
- Need for explanation in clinical decision support
- Increases trust and acceptance by clinicians
- Ensures transparency and fairness

#### **3.5.2. SHAP (SHapley Additive exPlanations)**
- Introduction to SHAP framework
- Explanation of SHAP values:
  - Positive: Increases AKI risk
  - Negative: Decreases AKI risk
- Why SHAP is suitable for medical AI

#### **3.5.3. SHAP Explainers**
- **TreeExplainer:** For Random Forest and XGBoost
- **LinearExplainer:** For Logistic Regression
- **KernelExplainer:** For SVM
- Rationale for each explainer choice

#### **3.5.4. Feature Importance Analysis**
- SHAP summary plot (beeswarm plot)
- Identify most important features
- Explain how features affect predictions
- Concrete interpretability examples

#### **3.5.5. Individual Prediction Explanations**
- Waterfall plots for individual cases
- Force plots to visualize contributions
- Help clinicians understand prediction rationale

---

### **3.6. Clinical Decision Support Process**

#### **3.6.1. Integration with Clinical Workflow**
- Combine model output with clinical judgment
- Support decision-making without replacing clinicians
- Telemedicine and remote monitoring applications

#### **3.6.2. System Output**
- **Binary Prediction:** AKI or no AKI
- **Risk Score:** AKI probability (0-1)
- **Feature Contributions:** SHAP values
- **Alerts:** Warning when risk is high

#### **3.6.3. Operational Process**
- Real-time monitoring with time-series vitals
- Continuous risk assessment
- Integration with EHR systems
- Feedback loop for model improvement

---

### **3.7. Advantages of Proposed Method**

#### **3.7.1. Comprehensiveness**
- Uses multiple vital sign sources
- Ensemble of multiple ML models
- XAI for prediction explanations

#### **3.7.2. Practicality**
- Easily collectible data (routine monitoring)
- Real-time predictions
- Easy integration into clinical workflow

#### **3.7.3. Explainability**
- Intuitive SHAP visualizations
- Individual explanations
- Feature importance rankings

#### **3.7.4. Scalability**
- Can add new features
- Can update model periodically
- Transfer learning to other datasets

---

### **3.8. Summary**

- Summarize main steps of AXKI method
- Emphasize novelty and contributions
- Connect to initial objectives
- Transition to Experiments/Results section

---

## NOTES FOR AUTHORS

### **Technical Details to Include:**
1. Pseudocode for main algorithm
2. Tables with tuned hyperparameters
3. Specific dataset statistics (N, features, distribution)
4. SHAP plot examples and interpretations
5. Comparison with baseline methods

### **Format Should Include:**
- Flowchart at section beginning (Figure 1)
- Sub-figures for important components
- Tables to summarize information
- Pseudocode in Appendix

### **Related Citations:**
- VitalDB dataset
- KDIGO criteria
- SHAP paper
- ML model documentation
- Related AKI prediction studies

### **Suggested Length:**
- Total: 8-12 pages (depending on journal)
- Each subsection: 1-2 pages
- Figures and tables: 2-3 pages

