# ĐỀ CƯƠNG PHẦN "GIỚI THIỆU" (SIMPLIFIED)
## Cho Small School-Level Conference

---

## SO SÁNH INTRODUCTION SECTION

| Aspect | Full Version | Simplified Version |
|--------|--------------|-------------------|
| **Sections** | 5-6 subsections | 3-4 subsections |
| **Length** | 2-3 pages | 1-1.5 pages |
| **Related Work** | Extensive review | Brief mention |
| **Problem Statement** | Detailed | Concise |
| **Contributions** | Listed in detail | Summary |
| **Writing Time** | 1 week | 1-2 days |

---

## CẤU TRÚC PHẦN GIỚI THIỆU

### **1.1. Background và Motivation**

**Viết thành 1 đoạn văn bao gồm:**
- Acute Kidney Injury (AKI) là vấn đề quan trọng sau phẫu thuật
- Tỷ lệ AKI sau phẫu thuật: ~5-15% cases
- AKI làm tăng mortality, length of stay, healthcare costs
- Cần phát hiện sớm để can thiệp kịp thời
- Machine Learning có thể hỗ trợ dự đoán AKI

---

### **1.2. Problem Statement**

**Viết thành 1 đoạn văn bao gồm:**
- Vấn đề: Khó dự đoán AKI trong giai đoạn sớm
- Challenges: Multiple risk factors, complex interactions
- Existing methods: Không đủ accurate hoặc không giải thích được
- Need: Accurate prediction + explainable AI
- Goal: Predict postoperative AKI using vital signs + ML + XAI

---

### **1.3. Proposed Solution và Contributions**

**Viết thành 1 đoạn văn bao gồm:**
- Đề xuất: AXKI framework tích hợp ML và XAI
- Data: VitalDB dataset với vital signs và clinical data
- Models: 4 ML models (LR, RF, XGBoost, SVM)
- Explainability: SHAP framework cho interpretations
- Contributions:
  - Novel framework combining ML and XAI
  - Comprehensive evaluation trên VitalDB
  - Clinical decision support với explainable predictions
  - Best ROC-AUC > 0.82

---

### **1.4. Paper Organization** (Optional)

**Một đoạn ngắn (~50 words):**
- Section 2: Related work
- Section 3: Proposed method (AXKI framework)
- Section 4: Experiments and results
- Section 5: Discussion and conclusion

---

## WRITING GUIDELINES

### **Structure for Each Section:**
- Start with broad context
- Narrow down to specific problem
- Present solution and contributions
- Preview paper structure

### **Key Points to Include:**

**1.1 Background:**
- AKI importance in postoperative care
- Statistics about AKI prevalence/impact
- Early detection importance
- ML potential

**1.2 Problem:**
- Prediction challenges
- Current method limitations
- Explainability requirement
- Research gap

**1.3 Contributions:**
- AXKI framework
- Methods used
- Key achievements
- Practical value

**1.4 Organization:**
- Brief overview of paper sections
- What each section covers

---

## TÀI LIỆU THAM KHẢO CHI TIẾT CHO TỪNG SECTION

### **1.1. Background Section:**

#### **AKI Definition và Guidelines:**
**[1] Kidney Disease: Improving Global Outcomes (KDIGO) Acute Kidney Injury Work Group** (2012). KDIGO clinical practice guideline for acute kidney injury. *Kidney International Supplements*, 2(1), 1-138. https://doi.org/10.1038/kisup.2012.1

**Vietnamese format:**
```
KDIGO Acute Kidney Injury Work Group. (2012). KDIGO clinical practice guideline 
for acute kidney injury. Kidney International Supplements, 2(1), 1-138.
```

#### **AKI Prevalence sau Phẫu thuật:**
**[2] Hoste, E. A., Kellum, J. A., Katz, N. M., Rosner, M. H., Haase, M., & Ronco, C.** (2014). Epidemiology of acute kidney injury. *Critical Care Medicine*, 42(6), 1478-1485. https://doi.org/10.1097/CCM.0000000000000286

**Vietnamese format:**
```
Hoste E. A., Kellum J. A., Katz N. M., Rosner M. H., Haase M., Ronco C. (2014). 
Epidemiology of acute kidney injury. Critical Care Medicine, 42(6), 1478-1485.
```

#### **AKI Impact lên Mortality và Costs:**
**[3] Rewa, O., & Bagshaw, S. M.** (2014). Acute kidney injury-epidemiology, outcomes and economics. *Nature Reviews Nephrology*, 10(4), 193-207. https://doi.org/10.1038/nrneph.2013.282

**Vietnamese format:**
```
Rewa O., Bagshaw S. M. (2014). Acute kidney injury-epidemiology, outcomes and 
economics. Nature Reviews Nephrology, 10(4), 193-207.
```

#### **ML Potential trong Healthcare:**
**[4] Topol, E. J.** (2019). High-performance medicine: the convergence of human and artificial intelligence. *Nature Medicine*, 25(1), 44-56. https://doi.org/10.1038/s41591-018-0300-7

**Vietnamese format:**
```
Topol E. J. (2019). High-performance medicine: the convergence of human and 
artificial intelligence. Nature Medicine, 25(1), 44-56.
```

---

### **1.2. Problem Statement Section:**

#### **ML cho AKI Prediction:**
**[5] Koyner, J. L., Carey, K. A., Edelson, D. P., & Churpek, M. M.** (2018). The development of a machine learning inpatient acute kidney injury prediction model. *Critical Care Medicine*, 46(7), 1070-1077. https://doi.org/10.1097/CCM.0000000000003123

**Vietnamese format:**
```
Koyner J. L., Carey K. A., Edelson D. P., Churpek M. M. (2018). The development 
of a machine learning inpatient acute kidney injury prediction model. Critical 
Care Medicine, 46(7), 1070-1077.
```

#### **Limitations của Existing Methods:**
**[6] Tomasev, N., Glorot, X., Rae, J. W., Zielinski, M., Askham, H., Saraiva, A., Mottram, A., Meyer, C., Ravuri, S., Protsyuk, I., Connell, A., Hughes, C. O., Karthikesalingam, A., Cornebise, J., Montgomery, H., Rees, G., Laing, C., Baker, C. R., Peterson, K., ... Suleyman, M.** (2019). A clinically applicable approach to continuous prediction of future acute kidney injury. *Nature*, 572(7767), 116-119. https://doi.org/10.1038/s41586-019-1390-1

**Vietnamese format:**
```
Tomasev N., Glorot X., Rae J. W., Zielinski M., Askham H., Saraiva A., Mottram A., 
Meyer C., Ravuri S., Protsyuk I., Connell A., Hughes C. O., Karthikesalingam A., 
Cornebise J., Montgomery H., Rees G., Laing C., Baker C. R., Peterson K., ... 
Suleyman M. (2019). A clinically applicable approach to continuous prediction 
of future acute kidney injury. Nature, 572(7767), 116-119.
```

#### **Explainability Requirement trong Medical AI:**
**[7] Amann, J., Blasimme, A., Vayena, E., Frey, D., & Madai, V. I.** (2020). Explainability for artificial intelligence in healthcare: a multidisciplinary perspective. *BMC Medical Informatics and Decision Making*, 20(1), 310. https://doi.org/10.1186/s12911-020-01332-6

**Vietnamese format:**
```
Amann J., Blasimme A., Vayena E., Frey D., Madai V. I. (2020). Explainability 
for artificial intelligence in healthcare: a multidisciplinary perspective. 
BMC Medical Informatics and Decision Making, 20(1), 310.
```

---

### **1.3. Contributions Section:**

#### **SHAP Framework:**
**[8] Lundberg, S. M., & Lee, S. I.** (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30, 4765-4774.

**Vietnamese format:**
```
Lundberg S. M., Lee S. I. (2017). A unified approach to interpreting model 
predictions. Advances in Neural Information Processing Systems, 30, 4765-4774.
```

#### **VitalDB Dataset:**
**[9] Lee, H. C., Park, Y., Yoon, S. B., Yang, S. M., Park, D., & Jung, C. W.** (2018). VitalDB, a high-fidelity multi-signal vital signs database for intensive care unit patients. *Scientific Data*, 5, 180003. https://doi.org/10.1038/sdata.2018.3

**Vietnamese format:**
```
Lee H. C., Park Y., Yoon S. B., Yang S. M., Park D., Jung C. W. (2018). VitalDB, 
a high-fidelity multi-signal vital signs database for intensive care unit 
patients. Scientific Data, 5, 180003.
```

#### **Postoperative AKI Prediction:**
**[10] Meersch, M., Schmidt, C., Hoffmeier, A., Van Aken, H., Wempe, C., Gerss, J., & Zarbock, A.** (2017). Prevention of cardiac surgery-associated AKI by implementing the KDIGO guidelines in high-risk patients identified by biomarkers: the PrevAKI randomized controlled trial. *Intensive Care Medicine*, 43(11), 1551-1561. https://doi.org/10.1007/s00134-016-4670-3

**Vietnamese format:**
```
Meersch M., Schmidt C., Hoffmeier A., Van Aken H., Wempe C., Gerss J., Zarbock A. 
(2017). Prevention of cardiac surgery-associated AKI by implementing the KDIGO 
guidelines in high-risk patients identified by biomarkers: the PrevAKI randomized 
controlled trial. Intensive Care Medicine, 43(11), 1551-1561.
```

---

## CÁCH SỬ DỤNG CÁC TÀI LIỆU THAM KHẢO

### **Trong đoạn văn:**

**Ví dụ cho Background:**
```
Acute Kidney Injury (AKI) is a serious postoperative complication, affecting 
5-15% of surgical patients [1,2]. AKI significantly increases mortality rates, 
prolongs hospital stays, and escalates healthcare costs [3]. Early detection 
of AKI is crucial for timely intervention and improved patient outcomes. 
Machine learning has shown great potential in predicting clinical outcomes 
across various medical domains [4].
```

**Ví dụ cho Problem:**
```
While machine learning models have been developed for AKI prediction [5,6], 
existing methods often lack interpretability, limiting their clinical adoption. 
Explainable AI has emerged as a critical requirement for medical applications, 
as clinicians need to understand the reasoning behind predictions [7].
```

**Ví dụ cho Contributions:**
```
We propose AXKI, an explainable AI framework that integrates machine learning 
with SHAP-based interpretability [8] for predicting postoperative AKI using 
vital signs from the VitalDB dataset [9]. Our approach addresses the need for 
both accurate prediction and clinical interpretability in postoperative AKI 
risk assessment [10].
```

---

## CITATION FORMAT CHO WORD DOCUMENT

### **In-text Citations:**
- Single reference: [1]
- Multiple references: [1,2] hoặc [1-3]
- Author-date (optional): "Lee et al. (2018)" hoặc "KDIGO (2012)"

### **Reference List Format (Vietnamese Style):**
```
References

1. KDIGO Acute Kidney Injury Work Group. (2012). KDIGO clinical practice 
   guideline for acute kidney injury. Kidney International Supplements, 2(1), 
   1-138.

2. Hoste E. A., Kellum J. A., Katz N. M., Rosner M. H., Haase M., Ronco C. 
   (2014). Epidemiology of acute kidney injury. Critical Care Medicine, 42(6), 
   1478-1485.

... (tiếp tục theo thứ tự)
```

### **Hướng dẫn trong Word:**
1. References tab → Manage Sources
2. Add each reference from the list above
3. Insert citations: References → Insert Citation
4. Bibliography: References → Bibliography → Works Cited

---

## COMPLETE INTRODUCTION STRUCTURE

### **Simplified Version:**

1. **1.1. Background và Motivation** → 1 paragraph
2. **1.2. Problem Statement** → 1 paragraph
3. **1.3. Proposed Solution và Contributions** → 1 paragraph
4. **1.4. Paper Organization** → 1 short paragraph (optional)

**Total:** 1-1.5 pages

---

## WRITING TIPS

### **Do's:**
- ✅ Start broad, end specific
- ✅ Use statistics to support claims
- ✅ Clearly state the problem
- ✅ Highlight contributions clearly
- ✅ Keep concise for conference

### **Don'ts:**
- ❌ Too much background detail
- ❌ Extensive literature review
- ❌ Too many citations
- ❌ Over-promise contributions
- ❌ Technical jargon overload

---

## CHECKLIST

- [ ] Introduction: 1-1.5 pages
- [ ] 3-4 paragraphs total
- [ ] Background motivates the work
- [ ] Problem clearly stated
- [ ] Contributions highlighted
- [ ] Paper organization mentioned
- [ ] Appropriate citations (5-8 references)
- [ ] Writing is clear and concise
- [ ] No unnecessary technical details
- [ ] Ready for small conference

---

## COMPARISON WITH FULL VERSION

### **Full Version (Journal):**
- 5-6 subsections
- Extensive related work review
- Detailed problem analysis
- Comprehensive contributions list
- Multiple citations (20-30)
- 2-3 pages

### **Simplified Version (Conference):**
- 3-4 subsections
- Brief background mention
- Concise problem statement
- Key contributions summary
- Essential citations (5-8)
- 1-1.5 pages

---

## SECTION BREAKDOWN EXAMPLE

### **1.1 Background (~150 words):**
Start: "Acute Kidney Injury (AKI) is a serious complication..."
Include: Prevalence stats, impact on outcomes, need for early detection
End: "...machine learning can assist in early AKI prediction."

### **1.2 Problem (~100 words):**
Start: "Early prediction of postoperative AKI remains challenging..."
Include: Current limitations, explainability gap
End: "...requires both accurate prediction and interpretable explanations."

### **1.3 Contributions (~150 words):**
Start: "We propose AXKI, an explainable AI framework..."
Include: Framework components, methods, key results
End: "...achieving ROC-AUC > 0.82 with interpretable predictions."

### **1.4 Organization (~50 words):**
"The rest of this paper is organized as follows: Section 2 reviews related work, Section 3 presents the proposed AXKI framework, Section 4 shows experimental results, and Section 5 concludes."

---

## TIMELINE

### **Writing Time:**
- Background: 2-3 hours
- Problem statement: 1 hour
- Contributions: 1-2 hours
- Organization: 30 minutes
- **Total:** 4-6 hours (1 day)

---

## KEY STATISTICS TO INCLUDE

### **AKI Background:**
- Postoperative AKI: 5-15% of surgical cases
- Increases mortality by 2-5x
- Hospital stay: +5-10 days
- Healthcare costs: Significant increase

### **ML Potential:**
- Can analyze multiple features simultaneously
- Identifies complex patterns
- Provides probability scores
- XAI enhances trust

---

## CITATION STRATEGY

### **Minimal Required Citations:**
1. KDIGO guidelines (already have)
2. VitalDB dataset (already have)
3. 1-2 AKI prevalence studies
4. 1-2 ML for AKI papers
5. SHAP paper
6. 1-2 medical AI interpretability papers

**Total:** 7-8 citations (essential only)

---

## OPENING SENTENCE OPTIONS

### **Option 1 (Clinical Focus):**
"Acute Kidney Injury (AKI) is a serious postoperative complication affecting 5-15% of surgical patients, leading to increased mortality, prolonged hospital stays, and higher healthcare costs."

### **Option 2 (ML Focus):**
"Machine learning has shown great potential in predicting postoperative complications, with explainable AI emerging as a critical requirement for clinical acceptance."

### **Option 3 (Problem Focus):**
"Early prediction of postoperative Acute Kidney Injury remains challenging due to the complex interplay of multiple risk factors and the lack of interpretable prediction models."

---

## CLOSING SENTENCE FOR INTRO

### **Suggested Closing:**
"The rest of this paper presents our proposed AXKI framework, demonstrates its effectiveness on the VitalDB dataset, and discusses its potential for clinical decision support in postoperative AKI prediction."

---

---

## HƯỚNG DẪN TÌM KIẾM VÀ COPY REFERENCES TRÊN GOOGLE SCHOLAR

### **Step 1: Truy cập Google Scholar**
- Vào website: https://scholar.google.com
- Hoặc tìm "Google Scholar" trên Google search

---

### **Step 2: Search Papers**

#### **A. Basic Search:**
- Nhập keywords vào search box:
  - `"acute kidney injury" "machine learning" prediction`
  - `"AKI" postoperative "explainable AI"`
  - `"SHAP" interpretability healthcare`
- Nhấn Enter hoặc click search icon

#### **B. Advanced Search:**
- Click "☰" menu ở góc trái trên → chọn "Advanced search"
- Điền các trường:
  - **with all of the words:** `acute kidney injury`
  - **with the exact phrase:** `machine learning`
  - **Return articles:** `since 2018` (chọn papers gần đây)
- Click "Search"

---

### **Step 3: Copy Citation (Không Cần Download)**

#### **Method 1: Copy từ Google Scholar (Đơn giản nhất)**

1. Tìm paper bạn muốn trên Google Scholar
2. Click icon **"Cite"** (quotation marks icon) dưới mỗi result
3. Chọn format:
   - **APA** → Copy
   - **MLA** → Copy  
   - **Chicago** → Copy
4. Right-click → Copy hoặc Ctrl+C
5. Paste vào Word document

**Example:**
```
Bước 1: Click "Cite" icon
Bước 2: Chọn "APA" format
Bước 3: Copy citation
Bước 4: Paste vào Word
```

#### **Method 2: Copy Citation Number**

1. Trên Google Scholar, mỗi paper có citation số
2. Copy format đơn giản: `[1] Author et al. (Year) Title`
3. Paste vào document
4. Format lại sau nếu cần

---

### **Step 4: Convert sang Vietnamese Format**

#### **Thay đổi Format:**

**Original (Google Scholar APA):**
```
Lee, H. C., Park, Y., Yoon, S. B., Yang, S. M., Park, D., & Jung, C. W. (2018). 
VitalDB, a high-fidelity multi-signal vital signs database for intensive care 
unit patients. Scientific Data, 5, 180003.
```

**Vietnamese Format (Sau khi convert):**
```
Lee H. C., Park Y., Yoon S. B., Yang S. M., Park D., Jung C. W. (2018). VitalDB, 
a high-fidelity multi-signal vital signs database for intensive care unit 
patients. Scientific Data, 5, 180003.
```

#### **Các thay đổi cần thiết:**
- ❌ **Lee, H. C.** → ✅ **Lee H. C.** (remove comma)
- ❌ **& Jung, C. W.** → ✅ **Jung C. W.** (remove & and comma)
- Giữ nguyên: (Year), Title, Journal

---

### **Step 5: Paste vào Word Document**

#### **Trong Word:**

1. **In-text Citation:**
   - Paste: `[1]`
   - Hoặc: `Lee et al. (2018)`

2. **Reference List:**
   - Paste full citation vào References section
   - Format: Times New Roman, 12pt
   - Numbering: 1, 2, 3...

3. **Format Reference List:**
   ```
   References
   
   1. Lee H. C., Park Y., Yoon S. B., Yang S. M., Park D., Jung C. W. (2018). 
      VitalDB, a high-fidelity multi-signal vital signs database for intensive 
      care unit patients. Scientific Data, 5, 180003.
   
   2. Lundberg S. M., Lee S. I. (2017). A unified approach to interpreting 
      model predictions. Advances in Neural Information Processing Systems, 
      30, 4765-4774.
   ```

---

### **Step 6: Search Keywords cho Project**

#### **Background Section:**
```
"acute kidney injury" "postoperative" prevalence
"AKI" "surgical" mortality complications
"acute kidney injury" epidemiology
```

#### **ML Methods:**
```
"acute kidney injury" "machine learning" prediction
"AKI" "random forest" "gradient boosting"
"kidney injury" "deep learning"
```

#### **Explainability:**
```
"explainable AI" "machine learning" medical
"SHAP" interpretability healthcare
"model interpretability" clinical
```

#### **Related Work:**
```
"AKI prediction" "vital signs"
"postoperative complications" prediction
"SHAP" "acute kidney injury"
```

---

### **Step 7: Filter Results**

#### **Trên Google Scholar:**
- Click **"Since 2018"** → chỉ papers gần đây
- Sort by **"Cited by"** → papers popular nhất
- Chọn **"Articles"** → journal articles only

#### **Check Quality:**
- Journal name: Nature, PLOS ONE, Science, etc.
- Citation count: cao = papers quan trọng
- Publication year: ưu tiên 2018-2024

---

### **Step 8: Organize References**

#### **Trong Word Document:**

1. **Create Reference List:**
   - Section cuối paper
   - Title: "References" hoặc "Tài liệu tham khảo"
   - Numbering: 1, 2, 3...

2. **Format References:**
   - Font: Times New Roman
   - Size: 12pt
   - Spacing: 1.5 hoặc double
   - Indent: second line onwards (hanging indent)

3. **Example Format:**
   ```
   1. Author A., Author B., Author C. (Year). Title of paper. Journal Name, 
      Volume(Issue), Page-Page.
   
   2. Author D., Author E. (Year). Title of paper. Journal Name, Volume(Issue), 
      Page-Page.
   ```

---

### **Step 9: Tips và Tricks**

#### **✅ Do's:**
- Use specific keywords
- Copy citation immediately khi tìm được
- Save citations vào Word ngay
- Convert to Vietnamese format
- Check paper relevance đọc abstract

#### **❌ Don'ts:**
- Don't use generic keywords
- Don't skip saving citations
- Don't forget to convert format
- Don't copy without reading abstract
- Don't use too old papers (before 2015)

---

### **Step 10: Complete Workflow**

```
1. Google Scholar → Search keywords
2. Filter: Since 2018, Sort by citations
3. Read abstracts of top results
4. Click "Cite" icon for relevant papers
5. Copy APA format
6. Convert to Vietnamese format (remove commas)
7. Paste vào Word Reference List
8. Number references: [1], [2], [3]...
9. Use in-text citations: [1] or Author et al. (Year)
10. Format properly in Word
```

---

### **Example: Complete Process**

#### **Search Query:**
```
Google Scholar → "acute kidney injury" "machine learning" prediction
```

#### **Found Paper:**
```
Lee et al. (2018) - VitalDB dataset paper
```

#### **Copy Citation:**
```
Click "Cite" → Select "APA" → Copy
```

#### **Convert to Vietnamese:**
```
Lee H. C., Park Y., Yoon S. B., Yang S. M., Park D., Jung C. W. (2018). 
VitalDB, a high-fidelity multi-signal vital signs database for intensive 
care unit patients. Scientific Data, 5, 180003.
```

#### **Paste vào Word:**
```
References

1. Lee H. C., Park Y., Yoon S. B., Yang S. M., Park D., Jung C. W. (2018). 
   VitalDB, a high-fidelity multi-signal vital signs database for intensive 
   care unit patients. Scientific Data, 5, 180003.
```

#### **Use in Text:**
```
We use the VitalDB dataset [1] for our experiments.
```

---

**NOTE:** Hướng dẫn này giúp bạn tìm và copy references trực tiếp từ Google Scholar mà không cần download .bib files, hoàn hảo cho Vietnamese paper!


