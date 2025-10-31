# ĐỀ CƯƠNG PHẦN "THỰC NGHIỆM VÀ KẾT QUẢ" (SIMPLIFIED)
## Cho Small School-Level Conference

---

## SO SÁNH VỚI VERSION ĐẦY ĐỦ

| Aspect | Full Version | Simplified Version (Conference) |
|--------|--------------|--------------------------------|
| **Sections** | 10 subsections | 4 subsections |
| **Length** | 10-15 pages | 2-3 pages |
| **Figures** | 10 figures | 3-4 figures |
| **Tables** | 10 tables | 2 tables |
| **Detail Level** | Comprehensive | Essential only |
| **Writing Time** | 1-2 weeks | 2-3 days |
| **Target** | Journal paper | Small conference |

---

## CẤU TRÚC PHẦN THỰC NGHIỆM VÀ KẾT QUẢ (RÚT GỌN)

### **4.1. Môi trường thực nghiệm (Experimental Setup)**

**Một đoạn văn duy nhất bao gồm:**
- Dataset: 3,989 cases từ VitalDB, 43 features
- Train/test split: 80/20 với StratifiedKFold (5-fold)
- Class imbalance: ~5% AKI positive, 95% negative
- Models: Logistic Regression, Random Forest, XGBoost, SVM
- Evaluation: ROC-AUC, AUPRC, Accuracy, Precision, Recall, F1-Score
- Tools: Python, scikit-learn, XGBoost, SHAP

---

### **4.2. Kết quả đánh giá mô hình (Model Evaluation Results)**

#### **4.2.1. So sánh performance các mô hình**
- **Bảng 1:** Performance comparison table
  - 4 models (LR, RF, XGBoost, SVM)
  - Các metrics chính: ROC-AUC, AUPRC, Accuracy, Precision, Recall, F1-Score
  - Highlight best performer (XGBoost hoặc Random Forest)

**Expected Results:**
- XGBoost: ROC-AUC: 0.82-0.84, Accuracy: 0.94
- Random Forest: ROC-AUC: 0.78-0.83, Accuracy: 0.94
- Logistic Regression: ROC-AUC: 0.71-0.79, Accuracy: 0.74
- SVM: Similar to Logistic Regression

#### **4.2.2. Visualization (2 figures only)**
- **Hình 2:** ROC curves comparison cho 4 models
- **Hình 3:** SHAP summary plot (beeswarm) showing top 10 features

---

### **4.3. Phân tích tính giải thích được (Interpretability Analysis)**

#### **4.3.1. Feature importance từ SHAP**
- **Bảng 2:** Top 10 features quan trọng nhất
- Mean absolute SHAP values
- Clinical interpretation ngắn gọn

#### **4.3.2. Individual prediction example**
- **Hình 4:** SHAP waterfall plot cho 1-2 cases ví dụ
- Giải thích tại sao model dự đoán AKI/không AKI

---

### **4.4. Kết quả và bàn luận (Results and Discussion)**

**Một đoạn ngắn về:**
- Best model: XGBoost (ROC-AUC > 0.82)
- Key features align với clinical knowledge
- Class imbalance impact
- Clinical potential: Assist early AKI detection
- Limitations: Small dataset, single center, need external validation

---

## ĐỊNH DẠNG HÌNH VÀ BẢNG (TỐI GIẢN)

### **Figures (chỉ 3-4 hình):**
- **Hình 1.** Flowchart (từ phần Method)
- **Hình 2.** ROC curves comparison
- **Hình 3.** SHAP summary plot
- **Hình 4.** SHAP waterfall plot (optional)

### **Tables (chỉ 2 bảng):**
- **Bảng 1.** Model performance comparison
- **Bảng 2.** Top 10 feature importance from SHAP

---

## LỊCH TRÌNH VIẾT (TIMELINE)

### **Timeline cho Small Conference Paper:**

#### **Version 1: Draft (1-2 ngày)**
- Viết 4.1: Setup (1 paragraph)
- Viết 4.2: Model comparison với Bảng 1
- Tạo Hình 2: ROC curves
- Viết 4.3: SHAP analysis với Bảng 2
- Tạo Hình 3: SHAP plot
- Viết 4.4: Discussion (1 paragraph)

#### **Version 2: Final (1 ngày)**
- Review và polish
- Add Hình 4 (optional waterfall plot)
- Final check formatting
- Add confidence intervals nếu có

---

## FORMATTING GUIDE CHO SMALL CONFERENCE

### **Độ dài đề xuất:**
- **Total Results section:** 2-3 pages
- **Setup:** 0.5 page
- **Model Evaluation:** 1 page
- **Interpretability:** 0.5 page
- **Discussion:** 0.5 page

### **Writing style:**
- Concise và to-the-point
- Focus on key findings
- Less technical jargon
- More accessible language
- Clear visualizations

### **Figures:**
- Resolution: 200-300 DPI
- Simple và clear
- Consistent colors với flowchart
- Large fonts cho readability trong presentation

### **Tables:**
- Round to 2-3 decimal places
- Bold best performers
- Simple formatting
- Fit on half page each

---

## KEY POINTS TO HIGHLIGHT

### **Must include:**
1. ✅ Best model performs well (ROC-AUC > 0.80)
2. ✅ XAI provides meaningful explanations
3. ✅ Top features align with clinical knowledge
4. ✅ System can assist clinical decision-making

### **Don't include:**
1. ❌ Detailed hyperparameter tuning (mention briefly only)
2. ❌ Ablation studies
3. ❌ Complex statistical tests
4. ❌ Multiple baseline comparisons
5. ❌ Extensive cross-validation details

---

## TEMPLATE PARA ĐƠN GIẢN

### **Setup Paragraph Template:**
```
We conducted experiments on VitalDB dataset containing 3,989 surgical cases 
with 43 features. The dataset was split into 80% training and 20% testing 
using stratified sampling to maintain class distribution (~5% AKI). We 
evaluated four ML models: Logistic Regression, Random Forest, XGBoost, and 
SVM using ROC-AUC as the primary metric.
```

### **Results Paragraph Template:**
```
Table 1 shows the performance comparison of all models. XGBoost achieved the 
best performance with ROC-AUC of 0.82-0.84 and accuracy of 0.94. Figure 2 
displays the ROC curves for all models, demonstrating XGBoost's superior 
discrimination ability. Random Forest showed comparable performance (ROC-AUC: 
0.78-0.83), while Logistic Regression and SVM achieved lower scores.
```

### **Interpretability Paragraph Template:**
```
SHAP analysis revealed that [top 3 features] were the most important 
predictors of AKI. Table 2 lists the top 10 features ranked by mean absolute 
SHAP values. Figure 3 visualizes the SHAP contributions, where positive 
values indicate increased AKI risk. The feature importance aligns with 
clinical knowledge about AKI risk factors.
```

### **Discussion Paragraph Template:**
```
Our results demonstrate that the AXKI system achieves good discrimination 
performance (ROC-AUC > 0.80) for predicting postoperative AKI. The XAI 
component provides interpretable explanations that can enhance clinical 
decision-making. Key limitations include dataset size and the need for 
external validation. Future work will focus on larger multi-center 
datasets and real-time deployment.
```

---

## CHECKLIST CHO SUBMISSION

- [ ] Results section: 2-3 pages
- [ ] 2-3 tables formatted properly
- [ ] 3-4 figures with good quality
- [ ] All key metrics reported
- [ ] Best model clearly highlighted
- [ ] SHAP analysis included
- [ ] Discussion mentions limitations
- [ ] Writing is clear and concise
- [ ] No technical jargon overload
- [ ] Figures match flowchart colors
- [ ] Ready for small conference presentation

---

## PRESENTATION TIPS

### **For Conference Presentation:**
1. **Slide 1:** Title + Overview flowchart
2. **Slide 2:** Setup (1 slide)
3. **Slide 3:** Results table (Bảng 1)
4. **Slide 4:** ROC curves (Hình 2)
5. **Slide 5:** SHAP plot (Hình 3)
6. **Slide 6:** Discussion + Conclusion

**Total:** 6 slides cho Results section

---

## SAMPLE TABLE FORMATS

### **Bảng 1: Model Performance Comparison**

| Model | ROC-AUC | AUPRC | Accuracy | Precision | Recall | F1-Score |
|-------|---------|-------|----------|-----------|--------|----------|
| **XGBoost** | **0.82** | **0.47** | **0.94** | **0.49** | **0.45** | **0.47** |
| Random Forest | 0.78 | 0.42 | 0.94 | 0.45 | 0.38 | 0.41 |
| Logistic Regression | 0.71 | 0.31 | 0.74 | 0.13 | 0.35 | 0.19 |
| SVM | 0.68 | 0.28 | 0.73 | 0.12 | 0.32 | 0.17 |

### **Bảng 2: Top 10 Feature Importance**

| Rank | Feature | SHAP Value | Clinical Meaning |
|------|---------|------------|------------------|
| 1 | preop_cr | 0.45 | Pre-operative creatinine |
| 2 | age | 0.32 | Patient age |
| 3 | ... | ... | ... |

---

**NOTE:** This simplified version focuses on core results that are achievable and suitable for a small school-level conference.

