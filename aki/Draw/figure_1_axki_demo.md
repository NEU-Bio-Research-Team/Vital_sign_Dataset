# Figure 1: AXKI Medical Dashboard

## Figure Caption (Legend) - Tiêu đề hình trong paper

### Vietnamese:

**Hình 1: Giao diện Hệ thống Hỗ trợ Quyết định AXKI.** Hệ thống dashboard y khoa AXKI với ba panel chính: (Trái) Giám sát dấu hiệu sinh tồn theo thời gian thực với các biểu đồ đường cho huyết áp tâm thu, huyết áp tâm trương, nhịp tim và SpO2; (Giữa) Dự đoán nguy cơ AKI với thông tin bệnh nhân, lựa chọn mô hình ML, và hiển thị kết quả dự đoán (96.1% nguy cơ cao); (Phải) Trợ lý AI lâm sàng cung cấp tóm tắt nguy cơ, các yếu tố góp phần (giải thích SHAP), và khuyến nghị lâm sàng.

### English:

**Figure 1: AXKI Medical Decision Support System Interface.** The AXKI medical dashboard with three main panels: (Left) Real-time vital signs monitoring with line charts for systolic/diastolic blood pressure, heart rate, and SpO2; (Center) AKI risk prediction with patient information, ML model selection, and prediction results display (96.1% high risk); (Right) AI clinical assistant providing risk summary, contributing factors (SHAP explanation), and clinical recommendations.

---

## Explanation in Paper Text - Giải thích trong nội dung paper

### Section: "System Implementation" hoặc "Dashboard Design"

### Vietnamese Version:

Hệ thống AXKI được triển khai dưới dạng dashboard web tương tác để hỗ trợ các bác sĩ trong việc ra quyết định lâm sàng. Dashboard được chia thành ba panel chính, mỗi panel phục vụ một mục đích cụ thể trong quy trình đánh giá và dự đoán nguy cơ AKI sau phẫu thuật.

**Panel Trái (Giám Sát Dấu Hiệu Sinh Tồn):** Panel này hiển thị các dấu hiệu sinh tồn theo thời gian thực của bệnh nhân dưới dạng biểu đồ đường. Hệ thống theo dõi các chỉ số quan trọng bao gồm huyết áp tâm thu (SBP), huyết áp tâm trương (DBP), nhịp tim (HR), và độ bão hòa oxy máu (SpO2). Mỗi dấu hiệu được mã màu riêng theo bảng màu medical để dễ dàng phân biệt. Biểu đồ cho phép bác sĩ theo dõi diễn biến lâm sàng của bệnh nhân và phát hiện các dấu hiệu bất thường.

**Panel Giữa (Dự Đoán Nguy Cơ):** Panel trung tâm cung cấp giao diện để chạy dự đoán nguy cơ AKI. Bác sĩ có thể xem thông tin bệnh nhân (ID, tên, tuổi, giới tính, loại phẫu thuật) và lựa chọn một trong năm mô hình dự đoán: Điểm AKI Truyền thống (KDIGO), Logistic Regression, Random Forest, XGBoost hoặc SVM. Sau khi nhấn nút "Dự Đoán Nguy Cơ AKI", hệ thống hiển thị xác suất nguy cơ (dưới dạng phần trăm), phân loại nguy cơ (Thấp/Trung bình/Cao), khoảng tin cậy, các yếu tố nguy cơ chính, và hiệu suất của mô hình được chọn.

**Panel Phải (Trợ Lý AI Lâm Sàng):** Panel này cung cấp trợ lý AI tự động để giải thích kết quả dự đoán và đưa ra khuyến nghị lâm sàng. Sau mỗi lần dự đoán, trợ lý AI tự động hiển thị bốn thông điệp: (1) Tóm tắt đánh giá nguy cơ với xác suất và phân loại nguy cơ; (2) Danh sách các yếu tố nguy cơ chính kèm biểu đồ waterfall SHAP để giải thích đóng góp của từng yếu tố; (3) Khuyến nghị lâm sàng cụ thể dựa trên mức độ nguy cơ; (4) Ghi chú về tương lai sẽ tích hợp LLM để tăng cường khả năng trả lời câu hỏi và tư vấn.

Dashboard được thiết kế với giao diện tối (dark mode) và ngôn ngữ tiếng Việt để phù hợp với môi trường lâm sàng Việt Nam. Hệ thống sử dụng dữ liệu tổng hợp (synthetic data) cho mục đích trình diễn nhưng có thể dễ dàng tích hợp với cơ sở dữ liệu bệnh nhân thực tế.

---

### English Version:

The AXKI system is implemented as an interactive web dashboard to support clinicians in clinical decision-making. The dashboard is divided into three main panels, each serving a specific purpose in the AKI risk assessment and prediction workflow.

**Left Panel (Vital Signs Monitoring):** This panel displays patient vital signs in real-time as line charts. The system monitors critical parameters including systolic blood pressure (SBP), diastolic blood pressure (DBP), heart rate (HR), and oxygen saturation (SpO2). Each vital sign is color-coded using a medical color scheme for easy identification. The charts allow physicians to monitor the patient's clinical progression and detect abnormal patterns.

**Center Panel (Risk Prediction):** The central panel provides the interface for running AKI risk predictions. Physicians can view patient information (ID, name, age, sex, surgery type) and select one of five prediction models: Traditional AKI Score (KDIGO), Logistic Regression, Random Forest, XGBoost, or SVM. After clicking the "Predict AKI Risk" button, the system displays the risk probability (as percentage), risk classification (Low/Medium/High), confidence interval, key risk factors, and model performance metrics.

**Right Panel (AI Clinical Assistant):** This panel provides an automated AI assistant to explain prediction results and deliver clinical recommendations. After each prediction, the AI assistant automatically displays four messages: (1) Risk assessment summary with probability and risk classification; (2) List of key risk factors accompanied by a SHAP waterfall chart explaining each factor's contribution; (3) Specific clinical recommendations based on risk level; (4) Note about future LLM integration to enhance question-answering and advisory capabilities.

The dashboard features a dark mode interface and Vietnamese language support to align with Vietnamese clinical environments. The system uses synthetic data for demonstration purposes but can be easily integrated with real patient databases.

---

## LaTeX Code for Including the Figure

```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=\textwidth]{images/AXKI_demo.png}
    \caption{Giao diện Hệ thống Hỗ trợ Quyết định AXKI. Hệ thống dashboard y khoa AXKI với ba panel chính: (Trái) Giám sát dấu hiệu sinh tồn theo thời gian thực với các biểu đồ đường cho huyết áp tâm thu, huyết áp tâm trương, nhịp tim và SpO2; (Giữa) Dự đoán nguy cơ AKI với thông tin bệnh nhân, lựa chọn mô hình ML, và hiển thị kết quả dự đoán (96.1\% nguy cơ cao); (Phải) Trợ lý AI lâm sàng cung cấp tóm tắt nguy cơ, các yếu tố góp phần (giải thích SHAP), và khuyến nghị lâm sàng.}
    \label{fig:axki_dashboard}
\end{figure}
```

## Suggested Section Structure

### Vietnamese Paper Structure:

```
3. Hệ thống AXKI (AXKI System)
   ...
   
4. Triển khai và Giao diện (Implementation and Interface)
   4.1. Thiết kế Dashboard
       [Giải thích về 3 panels]
   Hình 1: [Caption]

   4.2. Quy trình Sử dụng
       [Hướng dẫn cách sử dụng]

5. Kết quả và Đánh giá (Results and Evaluation)
```

### English Paper Structure:

```
3. AXKI System
   ...
   
4. Implementation and Dashboard Design
   4.1. Dashboard Layout
       [Explain 3 panels]
   Figure 1: [Caption]

   4.2. Workflow
       [Usage instructions]

5. Results and Evaluation
```

---

## Keywords for Paper

- Medical dashboard
- Decision support system
- Real-time vital signs monitoring
- SHAP interpretability
- AI clinical assistant
- Risk prediction interface
- Human-computer interaction in healthcare

