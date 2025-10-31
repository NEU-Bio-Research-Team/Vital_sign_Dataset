# ĐỀ CƯƠNG PHẦN "PHƯƠNG PHÁP ĐỀ XUẤT"
## AXKI: Hệ thống chấm điểm AI có thể giải thích được để đánh giá rủi ro tổn thương thận cấp tính

---

## CẤU TRÚC PHẦN PHƯƠNG PHÁP ĐỀ XUẤT

### **Hình X: Sơ đồ khái quát phương pháp AXKI**
_[Chèn flowchart vào đây]_

---

### **3.1. Tổng quan phương pháp**

Trong nghiên cứu này, chúng tôi đề xuất AXKI (Hệ thống chấm điểm AI có thể giải thích được để đánh giá rủi ro tổn thương thận cấp tính), một framework tích hợp Machine Learning và Explainable AI (XAI) nhằm dự đoán rủi ro tổn thương thận cấp tính sau phẫu thuật sử dụng các sinh hiệu lâm sàng từ VitalDB. Như minh họa trong Hình 1.1, hệ thống AXKI hoạt động theo năm giai đoạn chính: (1) đầu vào dữ liệu sinh hiệu lâm sàng, (2) tiền xử lý bao gồm cân bằng lớp và chia dữ liệu, (3) quy trình dự đoán với mô hình machine learning được huấn luyện và đánh giá, (4) quy trình lựa chọn sử dụng XAI để giải thích các dự đoán, và cuối cùng là (5) quy trình quyết định lâm sàng tạo ra chẩn đoán cuối cùng về khả năng suy thận. Mỗi thành phần trong pipeline được thiết kế để hỗ trợ lẫn nhau, trong đó mô hình ML cung cấp khả năng dự đoán chính xác, còn XAI đảm bảo tính minh bạch và giải thích được của các quyết định, từ đó tăng độ tin cậy cho các bác sĩ lâm sàng trong việc đưa ra quyết định điều trị.

---

### **3.2. Dữ liệu đầu vào (Input Data)**

Hệ thống AXKI sử dụng dữ liệu từ dataset VitalDB [1], một bộ dữ liệu lớn chứa các sinh hiệu lâm sàng được thu thập trong quá trình phẫu thuật tại nhiều bệnh viện. Dataset này bao gồm thông tin từ hàng nghìn cases phẫu thuật với các loại sinh hiệu đa dạng được ghi lại liên tục theo thời gian thực. Các sinh hiệu lâm sàng được sử dụng trong nghiên cứu này bao gồm: điện tim (ECG) để theo dõi nhịp tim và các bất thường về nhịp; plethysmography (PPG) cung cấp nhịp tim mạch (PLETH_HR) và độ bão hòa oxy máu (PLETH_SPO2); áp lực động mạch đo huyết áp tâm thu (ART_SBP) và huyết áp tâm trương (ART_DBP); capnography (ECO2) đo nồng độ CO2 cuối kỳ thở ra (ECO2_ETCO2). Ngoài ra, hệ thống còn tích hợp dữ liệu phẫu thuật bao gồm thông tin bệnh nhân (tuổi, giới tính, BMI), thông tin phẫu thuật (loại phẫu thuật, thời gian), và đặc biệt là các xét nghiệm máu quan trọng như creatinine trước và sau phẫu thuật. Để xác định sự xuất hiện của AKI, nghiên cứu sử dụng tiêu chuẩn KDIGO Stage I [2] với công thức `AKI = (postop_cr > preop_cr × 1.5)`, nghĩa là khi creatinine sau phẫu thuật tăng hơn 1.5 lần so với giá trị trước phẫu thuật. Tiêu chuẩn này được lựa chọn vì đây là định nghĩa phổ biến nhất trong lâm sàng và phù hợp với các nghiên cứu về AKI sau phẫu thuật [3].

---

### **3.3. Tiền xử lý dữ liệu (Preprocessing)**

#### **3.3.1. Gộp và làm sạch dữ liệu**
- Gộp dữ liệu từ nhiều nguồn (cases, labs, vitals)
- Xử lý các biến phân loại
- Loại bỏ các features không liên quan (caseid, subjectid)

#### **3.3.2. Cân bằng lớp (Class Balancing)**
- **Vấn đề:** Class imbalance nghiêm trọng (~18:1)
- **Giải pháp:**
  - Sử dụng StratifiedKFold để đảm bảo phân bố đồng đều
  - Hoặc áp dụng SMOTE/undersampling nếu cần
- Lý do quan trọng của việc cân bằng lớp

#### **3.3.3. Chia dữ liệu (Data Splitting)**
- Tỷ lệ train/test: 80/20
- Stratified splitting để giữ tỷ lệ AKI
- Cross-validation với StratifiedKFold (5-fold)

#### **3.3.4. Xử lý giá trị thiếu và chuẩn hóa**
- **Missing Value Imputation:** Mean imputation cho các features số
- **Standardization:** Sử dụng StandardScaler cho Logistic Regression và SVM
- Công thức và lý do sử dụng từng phương pháp

---

### **3.4. Quy trình dự đoán (Prediction Process)**

#### **3.4.1. Lựa chọn mô hình Machine Learning**
Giải thích việc sử dụng 4 mô hình ML:

**a) Logistic Regression**
- Mô hình đơn giản, dễ giải thích
- Sử dụng dữ liệu đã chuẩn hóa
- Baseline model

**b) Random Forest**
- Ensemble learning với nhiều decision trees
- Xử lý tốt non-linear relationships
- Robust với outliers
- Sử dụng dữ liệu imputed

**c) XGBoost**
- Gradient boosting mạnh mẽ
- Hiệu suất cao trên structured data
- Sử dụng dữ liệu imputed

**d) Support Vector Machine (SVM)**
- Khả năng phân loại tốt với kernel RBF
- Hiệu quả với high-dimensional data
- Sử dụng dữ liệu đã chuẩn hóa

#### **3.4.2. Huấn luyện và đánh giá mô hình**
- **Hyperparameter Tuning:** GridSearchCV với StratifiedKFold (5-fold)
- **Metrics chính:** ROC-AUC, AUPRC, Accuracy, Precision, Recall, F1-Score, PPV, NPV, Specificity
- **Validation Strategy:** Cross-validation để tránh overfitting
- So sánh performance giữa các mô hình
- Lựa chọn best model dựa trên ROC-AUC

---

### **3.5. Quy trình lựa chọn với XAI (Selection Process with XAI)**

#### **3.5.1. Tầm quan trọng của Explainable AI trong y học**
- Yêu cầu giải thích trong clinical decision support
- Tăng độ tin cậy và khả năng chấp nhận của bác sĩ
- Đảm bảo tính minh bạch và công bằng

#### **3.5.2. SHAP Framework cho Model Interpretability**
- Giới thiệu SHAP (SHapley Additive exPlanations)
- Giải thích giá trị SHAP:
  - Dương: Tăng nguy cơ AKI
  - Âm: Giảm nguy cơ AKI
- Tại sao SHAP phù hợp cho medical AI

#### **3.5.3. Các loại SHAP Explainers**
- **TreeExplainer:** Cho Random Forest và XGBoost
- **LinearExplainer:** Cho Logistic Regression
- **KernelExplainer:** Cho SVM
- Lý do lựa chọn từng explainer

#### **3.5.4. Phân tích Feature Importance**
- SHAP summary plot (beeswarm plot)
- Xác định features quan trọng nhất
- Giải thích cách các features ảnh hưởng đến prediction
- Ví dụ cụ thể về interpretability

#### **3.5.5. Individual Prediction Explanations**
- Waterfall plots cho từng case
- Force plots để visualize contributions
- Giúp bác sĩ hiểu lý do prediction cho từng bệnh nhân

---

### **3.6. Quy trình quyết định lâm sàng (Clinical Decision Process)**

#### **3.6.1. Chẩn đoán cuối cùng về khả năng suy thận**
- Kết hợp output của model với clinical judgment
- Hỗ trợ decision-making mà không thay thế bác sĩ
- Dựa trên SHAP explanations để đưa ra chẩn đoán

#### **3.6.2. Output của hệ thống**
- **Dự đoán nhị phân:** Có AKI hoặc không có AKI
- **Risk score:** Xác suất AKI (0-1)
- **Feature contributions:** SHAP values giúp giải thích
- **Alerts:** Cảnh báo khi risk cao

#### **3.6.3. Tích hợp với workflow lâm sàng**
- Real-time monitoring với time-series vitals
- Continuous risk assessment
- Integration với EHR systems
- Telemedicine và remote monitoring applications
- Feedback loop để cải thiện model

---

### **3.7. Tóm tắt**

- Tóm tắt lại năm giai đoạn chính của phương pháp AXKI
- Nhấn mạnh tính đổi mới: tích hợp ML và XAI cho AKI prediction
- Ưu điểm chính: 
  - Tính toàn diện: sử dụng nhiều nguồn sinh hiệu và ensemble ML models
  - Tính thực tiễn: dữ liệu dễ thu thập, real-time predictions
  - Tính giải thích được: SHAP visualization dễ hiểu cho clinicians
  - Khả năng mở rộng: có thể thêm features và cập nhật model
- Kết nối với mục tiêu ban đầu
- Dẫn vào phần Experiments/Results

---

## GHI CHÚ CHO TÁC GIẢ

### **Chi tiết kỹ thuật nên bao gồm:**
1. Pseudocode cho thuật toán chính
2. Tables với hyperparameters đã tune
3. Số liệu cụ thể về dataset (N, features, distribution)
4. Ví dụ SHAP plots và interpretations
5. Comparison với baseline methods

### **Định dạng nên có:**
- Flowchart ở đầu phần (Figure 1)
- Sub-figures cho từng component quan trọng
- Tables để tổng hợp thông tin
- Pseudocode trong Appendix

### **Trích dẫn liên quan:**
- VitalDB dataset
- KDIGO criteria
- SHAP paper
- ML model documentation
- Related AKI prediction studies

### **Độ dài đề xuất:**
- Tổng cộng: 8-12 trang (tùy journal)
- Mỗi subsection: 1-2 trang
- Figures và tables: 2-3 trang

---

## TÀI LIỆU THAM KHẢO CHO PHẦN 3.2

Để sử dụng trong Word document, định dạng citations như sau:

### **Reference [1] - VitalDB Dataset**
**APA Format:**
```
Lee, H. C., Park, Y., Yoon, S. B., Yang, S. M., Park, D., & Jung, C. W. (2018). VitalDB, a high-fidelity multi-signal vital signs database for intensive care unit patients. Scientific Data, 5, 180003. https://doi.org/10.1038/sdata.2018.3
```

**Vietnam Format (cho bài báo tiếng Việt):**
```
Lee H. C., Park Y., Yoon S. B., Yang S. M., Park D., Jung C. W. (2018). VitalDB, a high-fidelity multi-signal vital signs database for intensive care unit patients. Scientific Data, 5, 180003. https://doi.org/10.1038/sdata.2018.3
```

### **Reference [2] - KDIGO Guidelines**
**APA Format:**
```
Kidney Disease: Improving Global Outcomes (KDIGO) Acute Kidney Injury Work Group. (2012). KDIGO clinical practice guideline for acute kidney injury. Kidney International Supplements, 2(1), 1-138. https://doi.org/10.1038/kisup.2012.1
```

**Vietnam Format:**
```
Kidney Disease: Improving Global Outcomes (KDIGO) Acute Kidney Injury Work Group. (2012). KDIGO clinical practice guideline for acute kidney injury. Kidney International Supplements, 2(1), 1-138. https://doi.org/10.1038/kisup.2012.1
```

### **Reference [3] - Postoperative AKI Studies**
**APA Format (example):**
```
Meersch, M., Schmidt, C., Hoffmeier, A., Van Aken, H., Wempe, C., Gerss, J., & Zarbock, A. (2017). Prevention of cardiac surgery-associated AKI by implementing the KDIGO guidelines in high-risk patients identified by biomarkers: the PrevAKI randomized controlled trial. Intensive Care Medicine, 43(11), 1551-1561. https://doi.org/10.1007/s00134-016-4670-3
```

**Vietnam Format:**
```
Meersch M., Schmidt C., Hoffmeier A., Van Aken H., Wempe C., Gerss J., Zarbock A. (2017). Prevention of cardiac surgery-associated AKI by implementing the KDIGO guidelines in high-risk patients identified by biomarkers: the PrevAKI randomized controlled trial. Intensive Care Medicine, 43(11), 1551-1561. https://doi.org/10.1007/s00134-016-4670-3
```

### **Hướng dẫn sử dụng trong Word:**

1. **Tạo Bibliography:**
   - References tab → Insert Citation → Add New Source
   - Chọn Type: Journal Article
   - Điền thông tin từ các references trên

2. **Insert In-text Citation:**
   - Đặt cursor sau "VitalDB" → References → Insert Citation → chọn Reference [1]
   - Tương tự cho KDIGO [2] và AKI studies [3]

3. **Format:**
   - Style: APA (hoặc Vietnamese format tùy journal)
   - Font: Times New Roman, 12pt
   - Spacing: Double (hoặc theo yêu cầu journal)

