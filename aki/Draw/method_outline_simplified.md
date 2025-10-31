# ĐỀ CƯƠNG PHẦN "PHƯƠNG PHÁP ĐỀ XUẤT" (SIMPLIFIED)
## Cho Small School-Level Conference
## Sections 3.3 onwards (Sau phần Input Data)

---

## SO SÁNH METHOD SECTIONS

| Aspect | Full Version | Simplified Version |
|--------|--------------|-------------------|
| **3.3 Preprocessing** | 5 subsections | 1 paragraph |
| **3.4 Prediction** | 2 subsections (detailed) | 1 paragraph |
| **3.5 XAI** | 5 subsections | 1 paragraph |
| **3.6 Clinical Decision** | 3 subsections | 1 paragraph |
| **3.7 Summary** | Long bullet points | 1 paragraph |
| **Total Length** | 5-7 pages | 1.5-2 pages |
| **Technical Detail** | Comprehensive | Essential only |

---

## CẤU TRÚC PHẦN PHƯƠNG PHÁP (TỪ 3.3 TRỞ ĐI)

### **3.3. Tiền xử lý dữ liệu (Preprocessing)**

**Viết thành 1 đoạn văn bao gồm:**
- Gộp và làm sạch: loại bỏ biến phân loại, features không liên quan
- Cân bằng lớp: StratifiedKFold, train/test split 80/20
- Xử lý giá trị thiếu: mean imputation cho RF và XGBoost
- Chuẩn hóa: StandardScaler cho LR và SVM
- Dataset: 3,989 cases, 43 features, class imbalance ~18:1

---

### **3.4. Quy trình dự đoán (Prediction Process)**

**Viết thành 1 đoạn văn bao gồm:**
- 4 mô hình ML: Logistic Regression, Random Forest, XGBoost, SVM
- Brief description của từng model (1 câu)
- Training: GridSearchCV + StratifiedKFold (5-fold)
- Metrics: ROC-AUC (chính), AUPRC, Accuracy, Precision, Recall, F1-Score
- Best model selection based on ROC-AUC

---

### **3.5. Quy trình lựa chọn với XAI (Selection Process with XAI)**

**Viết thành 1 đoạn văn bao gồm:**
- SHAP framework cho interpretability
- SHAP values dương = tăng nguy cơ AKI, âm = giảm nguy cơ
- Các loại explainers: TreeExplainer, LinearExplainer, KernelExplainer
- Visualizations: SHAP summary plot, waterfall plots
- Clinical applications: giúp bác sĩ hiểu predictions

---

### **3.6. Quy trình quyết định lâm sàng (Clinical Decision Process)**

**Viết thành 1 đoạn văn bao gồm:**
- Output: binary prediction, risk score, SHAP contributions, alerts
- Combine với clinical judgment
- SHAP explanations cho final diagnosis
- Future: EHR integration, real-time monitoring

---

### **3.7. Tóm tắt**

**Viết thành 1 đoạn văn bao gồm:**
- Tóm tắt 5 giai đoạn chính
- Tính đổi mới: tích hợp ML + XAI
- 4 ưu điểm chính:
  - Tính toàn diện
  - Tính thực tiễn
  - Tính giải thích được
  - Khả năng mở rộng

---

## WRITING GUIDELINES

### **Structure for Each Section:**
- Start each section with a topic sentence
- List key points in logical order
- Use connecting words: "sau đó", "tiếp theo", "cuối cùng"
- Keep paragraph under 150 words
- End with summary sentence

### **Key Points to Include:**

**3.3 Preprocessing:**
- 4 main steps
- Mention specific techniques (StratifiedKFold, StandardScaler)
- Include dataset statistics (3,989 cases, 43 features, ~18:1 ratio)

**3.4 Prediction:**
- 4 models with brief description
- Training method (GridSearchCV, StratifiedKFold)
- Evaluation metrics
- Best model selection criterion

**3.5 XAI:**
- SHAP framework
- Positive/negative values meaning
- Types of explainers
- Visualizations
- Clinical benefit

**3.6 Clinical Decision:**
- System outputs (4 types)
- Integration with clinical judgment
- SHAP for diagnosis
- Future applications

**3.7 Summary:**
- 5 main stages recap
- Innovation point
- 4 key advantages

---

## SUMMARY: ENTIRE METHOD SECTION STRUCTURE

### **Complete Method Section (Simplified):**

1. **3.1. Tổng quan phương pháp** ✅ (Already written as paragraph)
2. **3.2. Dữ liệu đầu vào** ✅ (Already written as paragraph)
3. **3.3. Tiền xử lý dữ liệu** → 1 paragraph
4. **3.4. Quy trình dự đoán** → 1 paragraph
5. **3.5. Quy trình lựa chọn với XAI** → 1 paragraph
6. **3.6. Quy trình quyết định lâm sàng** → 1 paragraph
7. **3.7. Tóm tắt** → 1 paragraph

**Total:** ~2 pages

---

## COMPLETE METHOD SECTION WITH FIGURES

### **Flowchart Required:**
- **Hình 1:** AXKI framework flowchart (từ phần đầu)
- Insert vào đầu section 3.1

### **No Additional Figures Needed:**
- Flowchart là hình duy nhất cho Method section
- Tất cả visualization được show trong Results section

---

## TIMELINE CHO METHOD SECTION

### **Writing Time:**
- **3.1-3.2:** Already done ✅
- **3.3-3.7:** 1-2 hours
- **Review & Polish:** 30 minutes
- **Total:** 2-3 hours to complete Method section

---

## CHECKLIST

- [ ] Flowchart inserted at beginning of section 3.1
- [ ] Section 3.1: Overview paragraph ✅
- [ ] Section 3.2: Input data paragraph ✅
- [ ] Section 3.3: Preprocessing paragraph
- [ ] Section 3.4: Prediction process paragraph
- [ ] Section 3.5: XAI paragraph
- [ ] Section 3.6: Clinical decision paragraph
- [ ] Section 3.7: Summary paragraph
- [ ] All paragraphs are concise (under 150 words each)
- [ ] Technical terms explained briefly
- [ ] Flow matches flowchart exactly
- [ ] Ready for small conference

---

## MODIFICATIONS TO ORIGINAL OUTLINE

### **What to Keep from Full Version:**
- ✅ Section 3.1: Overview paragraph
- ✅ Section 3.2: Input data paragraph with references
- ✅ Flowchart (Hình 1)
- ✅ Key technical concepts (StratifiedKFold, GridSearchCV, SHAP)

### **What to Simplify:**
- ❌ Multiple subsections → Single paragraph each
- ❌ Detailed technical explanations → Brief mentions
- ❌ Separate feature lists → Integrated in text
- ❌ Extensive hyperparameter details → Mention only key ones
- ❌ Long bullet points → Flowing paragraphs

### **What to Remove:**
- ❌ Pseudocode (save for full journal version)
- ❌ Detailed explainer descriptions
- ❌ Ablation studies
- ❌ Complex comparisons
- ❌ Extensive future work discussions

---

## FINAL STRUCTURE COMPARISON

### **Before (Full Version):**
- Section 3.3: 5 subsections (detailed)
- Section 3.4: 2 subsections (detailed)
- Section 3.5: 5 subsections (detailed)
- Section 3.6: 3 subsections (detailed)
- Section 3.7: Multiple bullet points
- **Total:** 5-7 pages

### **After (Simplified Version):**
- Section 3.3: 1 paragraph
- Section 3.4: 1 paragraph
- Section 3.5: 1 paragraph
- Section 3.6: 1 paragraph
- Section 3.7: 1 paragraph
- **Total:** 1.5-2 pages

---

**NOTE:** This simplified version reduces Method section from 5-7 pages to 1.5-2 pages while maintaining all essential information about your AXKI framework. Perfect for small school-level conference submission!

