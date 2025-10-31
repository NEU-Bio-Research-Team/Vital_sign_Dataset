# ĐỀ CƯƠNG PHẦN "THỰC NGHIỆM VÀ KẾT QUẢ"
## AXKI: Explainable AI System for Acute Kidney Injury Risk Assessment

---

## CẤU TRÚC PHẦN THỰC NGHIỆM VÀ KẾT QUẢ

### **4.1. Môi trường thực nghiệm (Experimental Setup)**

#### **4.1.1. Phần cứng và phần mềm**
- Máy tính: CPU specifications, RAM, GPU (nếu có)
- Hệ điều hành
- Python version và các thư viện chính:
  - scikit-learn (version)
  - XGBoost (version)
  - SHAP (version)
  - pandas, numpy, matplotlib, seaborn

#### **4.1.2. Cấu hình tham số**
- Random seed: 0 hoặc 42 (để reproducibility)
- Train/test split: 80/20
- Cross-validation: StratifiedKFold với 5 folds
- Hyperparameter tuning: GridSearchCV
- Evaluation metrics: ROC-AUC, AUPRC, Accuracy, Precision, Recall, F1-Score, PPV, NPV, Specificity

#### **4.1.3. Dataset**
- Số lượng samples: 3,989 cases sau khi loại bỏ missing values
- Số lượng features: 43 features (sau preprocessing)
- Class distribution: 
  - AKI positive: ~167 cases (5.2%)
  - AKI negative: ~3,022 cases (94.8%)
  - Class imbalance ratio: ~18:1
- Train set: 3,191 samples
- Test set: 798 samples

---

### **4.2. Kết quả phân tích dữ liệu (Data Analysis Results)**

#### **4.2.1. Đặc điểm dataset**
- Bảng 1: Thống kê mô tả các features quan trọng
  - Mean, median, std cho continuous variables
  - Frequency, percentage cho categorical variables
  - Missing value percentages

#### **4.2.2. Class imbalance analysis**
- Hình 2: Class distribution (bar chart hoặc pie chart)
- Ảnh hưởng của class imbalance lên model performance
- Analysis: High accuracy (>94%) nhưng low ROC-AUC và AUPRC

#### **4.2.3. Feature importance analysis**
- Hình 3: Correlation heatmap giữa features và AKI
- Top 10 features highly correlated với AKI
- Feature engineering insights

---

### **4.3. Kết quả so sánh các mô hình ML (ML Model Comparison Results)**

#### **4.3.1. Performance của các mô hình cơ bản**
- Bảng 2: Performance comparison table
  - Models: Logistic Regression, Random Forest, XGBoost, SVM
  - Metrics: ROC-AUC, AUPRC, Accuracy, Precision, Recall, F1-Score, PPV, NPV, Specificity
  - Baseline results without class balancing

**Expected Results (from notebooks):**
- XGBoost: ROC-AUC: 0.82-0.84, Accuracy: 0.94
- Random Forest: ROC-AUC: 0.78-0.83, Accuracy: 0.94
- Logistic Regression: ROC-AUC: 0.71-0.79, Accuracy: 0.74
- SVM: Similar to Logistic Regression

#### **4.3.2. ROC Curves**
- Hình 4: ROC curves comparison cho tất cả models
- AUC values cho từng model
- Interpretation của ROC curves

#### **4.3.3. Precision-Recall Curves**
- Hình 5: PR curves comparison
- AUPRC values (Area Under Precision-Recall Curve)
- Đặc biệt quan trọng cho imbalanced data

#### **4.3.4. Confusion Matrices**
- Hình 6: Confusion matrices cho từng model
- True Positives, True Negatives, False Positives, False Negatives
- Clinical interpretation của confusion matrices

---

### **4.4. Kết quả tối ưu hóa siêu tham số (Hyperparameter Tuning Results)**

#### **4.4.1. Hyperparameter search space**
- Bảng 3: Hyperparameter ranges cho từng model
  - Logistic Regression: C, penalty, solver
  - Random Forest: n_estimators, max_depth, min_samples_split
  - XGBoost: learning_rate, max_depth, n_estimators, subsample
  - SVM: C, gamma, kernel

#### **4.4.2. Best hyperparameters**
- Bảng 4: Optimal hyperparameters sau GridSearchCV
- Cross-validation scores
- Improvement so với default parameters

#### **4.4.3. Impact của hyperparameter tuning**
- Bảng 5: Performance comparison before/after tuning
- ROC-AUC improvement
- Other metrics improvements

---

### **4.5. Kết quả sau cân bằng lớp (Class Balancing Results)**

#### **4.5.1. StratifiedKFold approach**
- Method: StratifiedKFold để maintain class ratio trong CV
- Results: Improved generalization

#### **4.5.2. SMOTE oversampling (optional)**
- Bảng 6: SMOTE results comparison
- Before/After class distribution
- Performance impact:
  - Balanced Logistic Regression: ROC-AUC: 0.61-0.71
  - Balanced Random Forest: ROC-AUC: 0.85 (improved!)
  - Balanced XGBoost: ROC-AUC: 0.78

#### **4.5.3. Final model selection**
- Best model: XGBoost hoặc Random Forest (based on ROC-AUC)
- Rationale: Highest discrimination ability
- Trade-offs giữa các metrics

---

### **4.6. Kết quả giải thích mô hình với SHAP (SHAP Interpretability Results)**

#### **4.6.1. Global feature importance**
- Hình 7: SHAP summary plot (beeswarm plot)
- Top 10 most important features
- Feature contributions to predictions
- Colors indicate feature values (high/low)

#### **4.6.2. Feature importance rankings**
- Bảng 7: Feature importance scores từ SHAP
- Mean absolute SHAP values
- Clinical interpretation của top features

#### **4.6.3. Individual prediction explanations**
- Hình 8: Waterfall plot cho một case example
- Hình 9: Force plot visualization
- Explaining prediction cho một patient cụ thể

#### **4.6.4. Feature interactions**
- Hình 10: SHAP interaction values (nếu applicable)
- Pairwise feature interactions
- Clinical insights

---

### **4.7. So sánh với các phương pháp baseline (Baseline Comparison)**

#### **4.7.1. Baseline methods**
- Bảng 8: Comparison với:
  - Dummy classifier (majority class)
  - Simple rule-based method
  - Previous AKI prediction studies

#### **4.7.2. Statistical significance**
- Statistical tests (t-test, Mann-Whitney U test)
- Confidence intervals cho metrics
- P-values

---

### **4.8. Ablation study (nếu có)**

#### **4.8.1. Component contributions**
- Bảng 9: Ablation results
- Contribution của từng component:
  - Without preprocessing
  - Without hyperparameter tuning
  - Without XAI explanations
- Performance degradation analysis

---

### **4.9. Kết quả đánh giá ngoài (External Validation)**

#### **4.9.1. Cross-validation results**
- Bảng 10: 5-fold CV results
- Mean ± std cho các metrics
- Consistency across folds

#### **4.9.2. Generalization analysis**
- Train vs Test performance
- Overfitting detection
- Bias-variance trade-off

---

### **4.10. Bàn luận kết quả (Results Discussion)**

#### **4.10.1. Key findings**
- XGBoost hoặc Random Forest achieves best performance
- ROC-AUC > 0.82 indicates good discrimination
- Class imbalance significantly affects metrics
- SHAP provides meaningful clinical insights

#### **4.10.2. Clinical implications**
- Top features align with clinical knowledge
- Model can assist in early AKI detection
- XAI explanations enhance trust

#### **4.10.3. Limitations**
- Imbalanced dataset (5% positive class)
- Limited to single center data (VitalDB)
- Need for external validation
- Postoperative AKI only (7 days window)

#### **4.10.4. Future work**
- Larger datasets
- Multi-center validation
- Real-time deployment
- Integration with EHR systems

---

## ĐỊNH DẠNG HÌNH VÀ BẢNG

### **Figures cần thiết:**

**Hình 1.** Overview flowchart (có trong phần Method)

**Hình 2.** Class distribution trong dataset

**Hình 3.** Feature correlation heatmap với AKI

**Hình 4.** ROC curves comparison cho tất cả models

**Hình 5.** Precision-Recall curves comparison

**Hình 6.** Confusion matrices cho từng model

**Hình 7.** SHAP summary plot (beeswarm)

**Hình 8.** SHAP waterfall plot (individual prediction)

**Hình 9.** SHAP force plot

**Hình 10.** Feature interaction plot (optional)

### **Tables cần thiết:**

**Bảng 1.** Descriptive statistics

**Bảng 2.** Model performance comparison

**Bảng 3.** Hyperparameter search space

**Bảng 4.** Optimal hyperparameters

**Bảng 5.** Before/After tuning comparison

**Bảng 6.** Class balancing results

**Bảng 7.** Feature importance from SHAP

**Bảng 8.** Baseline comparison

**Bảng 9.** Ablation study results (optional)

**Bảng 10.** Cross-validation results

---

## METRICS CHÍNH ĐỂ BÁO CÁO

### **Primary Metrics:**
1. **ROC-AUC**: Main discrimination metric
2. **AUPRC**: Better for imbalanced data
3. **Sensitivity (Recall)**: Important for medical diagnosis
4. **Specificity**: True negative rate
5. **PPV (Precision)**: Clinical relevance

### **Secondary Metrics:**
- Accuracy (less informative due to imbalance)
- F1-Score
- NPV (Negative Predictive Value)

---

## TIPS CHO TÁC GIẢ

### **Viết Results:**
1. **Be objective**: Report results without interpretation
2. **Use past tense**: "The model achieved..."
3. **Include confidence intervals**: "ROC-AUC: 0.82 (95% CI: 0.78-0.86)"
4. **Show both visualizations and numbers**: Tables + Figures
5. **Highlight key findings**: Bold hoặc italicize important results

### **Figures:**
- High resolution (300 DPI minimum)
- Clear labels and legends
- Consistent color scheme với Method section
- Professional appearance

### **Tables:**
- Round to 3-4 decimal places
- Highlight best performers
- Use proper formatting
- Include sample sizes

### **Citations:**
- Reference previous similar studies
- Compare với state-of-the-art
- Cite evaluation metric definitions if needed

---

## ĐỘ DÀI ĐỀ XUẤT

- **Total**: 10-15 pages (tùy journal)
- **Each subsection**: 1-2 pages
- **Figures**: 10-12 figures
- **Tables**: 8-10 tables
- **Results text**: Concise, data-driven
- **Discussion**: Separate section (nếu journal yêu cầu)

