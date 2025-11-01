# Project Context Backup - Vital Sign Dataset Projects

**Last Updated:** 2024-12-01 (Evening Session)
**AI Readiness:** 100%

## Repository Overview

This repository contains **two independent medical AI research projects** based on the VitalDB dataset:

1. **AKI Prediction** (`aki/`) - Postoperative Acute Kidney Injury prediction
2. **Arrhythmia Classification** (`arrdb/`) - Multi-level cardiac arrhythmia detection

Both projects are self-contained with their own source code, notebooks, data, and documentation.

---

## Project 1: AKI Prediction

### Project Overview

**Project Name:** AKI Prediction from Vital Signs Dataset  
**Purpose:** Predict postoperative Acute Kidney Injury (AKI) using vital signs and clinical data from VitalDB  
**Technology Stack:** Python, scikit-learn, XGBoost, SHAP, pandas, numpy, matplotlib, Dash

## Current Repository Structure

```
Vital_sign_Dataset/
│
├── aki/                           # Project 1: AKI Prediction
│   ├── src/                       # Source code package
│   │   ├── __init__.py
│   │   ├── utils.py              # Data loading and preprocessing
│   │   ├── train.py              # Model training and hyperparameter tuning
│   │   ├── evaluate.py           # Model evaluation and metrics
│   │   ├── visualization.py      # Plotting and visualization
│   │   └── shap_explainer.py     # SHAP-based interpretability
│   ├── notebooks/                 # Jupyter notebooks
│   │   ├── Pat_*.ipynb           # Patient-level experiments
│   │   │   ├── Pat_dl_examination.ipynb      # DL vs ML comparison
│   │   │   ├── Pat_aki_prediction.ipynb      # Original AKI prediction
│   │   │   ├── Pat_aki_pred_hyper.ipynb      # Hyperparameter tuning
│   │   │   ├── Pat_dataset_analysis.ipynb    # Dataset analysis
│   │   │   ├── Pat_data_vis.ipynb            # Data visualization
│   │   │   ├── Pat_example_train.ipynb       # Training example
│   │   │   └── Pat_example_eval.ipynb        # Evaluation example
│   │   ├── Win_*.ipynb           # Window-level experiments
│   │   │   └── Win_windowaki_examine.ipynb   # Temporal data exploration
│   │   └── Com_*.ipynb           # Combined experiments
│   │       └── Com_temporal_features_aki.ipynb  # Temporal feature extraction & comparison
│   ├── dashboard/                 # Medical Dashboard (Dash)
│   │   ├── app.py                # Main dashboard application
│   │   ├── components/           # Dashboard components
│   │   ├── utils/                # Utility functions
│   │   ├── assets/               # Styles and assets
│   │   └── requirements_dashboard.txt
│   ├── paper/                     # LaTeX research paper
│   │   ├── main.tex              # Main document
│   │   ├── sections/             # Paper sections
│   │   ├── references.bib        # Bibliography
│   │   └── out/                  # Compiled PDF
│   ├── Draw/                      # Flowchart and paper outlines
│   │   ├── vitaldb_framework.jpg
│   │   ├── color_codes.txt
│   │   └── *_outline*.md
│   ├── examples/                  # Python script examples
│   ├── processed/                 # Processed temporal features
│   │   ├── .gitignore            # Exclude large CSV files
│   │   └── temporal_features_aki.csv  # Extracted features (not in git)
│   ├── shap_plots/                # SHAP visualization outputs
│   ├── Notes.md                   # Research notes and findings
│   └── README.md                  # AKI project documentation
│
├── arrdb/                         # Project 2: Arrhythmia Classification
│   ├── src/                       # Source code package
│   │   ├── __init__.py
│   │   ├── data_loader.py        # Load VitalDB annotation files
│   │   ├── feature_extractor.py  # HRV feature extraction
│   │   ├── preprocess.py         # Data preprocessing and windowing
│   │   ├── models.py             # PyTorch DL architectures
│   │   ├── train_models.py       # Training functions
│   │   ├── train_models_simple.py # Simplified ML training
│   │   ├── evaluate_models.py    # Evaluation and metrics
│   │   └── visualization.py      # Visualization utilities
│   ├── notebooks/                 # Experiment notebooks
│   │   ├── beat_dl.ipynb         # CNN for beat classification
│   │   ├── beat_lstm.ipynb       # LSTM for beat classification
│   │   ├── rhythm_dl.ipynb       # CNN for rhythm classification
│   │   ├── rhythm_lstm.ipynb     # LSTM for rhythm classification
│   │   ├── trad_ml.ipynb         # Traditional ML for both tasks
│   │   ├── classification_visualization.ipynb  # DL visualization
│   │   ├── ml_visualization.ipynb # ML visualization
│   │   └── general_evaluation.ipynb # Comprehensive comparison
│   ├── experiments/
│   │   └── results/
│   │       ├── predictions/      # Saved model predictions
│   │       ├── metrics/          # Performance metrics (CSV)
│   │       └── plots/            # Visualization figures
│   ├── LabelFile/                 # ECG annotations and metadata
│   │   ├── metadata.csv
│   │   └── Annotation_Files_250907/  # 482 patient annotation files
│   ├── EXP_GUIDE.md               # Step-by-step execution guide
│   ├── Notes.md                   # Research notes and paper draft
│   └── requirements_arrdb.txt     # Project-specific dependencies
│
├── requirements.txt               # Common Python dependencies
├── backup-context.md              # This file
└── README.md                      # Main repository README
```

## Current Process Workflow

### 1. Data Loading & Preprocessing (utils.py)
- **Load VitalDB data** from API (cases and labs)
- **Process creatinine levels** (preop and postop within 7 days)
- **Create AKI label** using KDIGO Stage I definition (postop_cr > 1.5 × preop_cr)
- **Remove categorical variables** and prepare numerical features
- **Handle missing values** using SimpleImputer (mean strategy)
- **Split data** into train/test (80/20) with stratification
- **Create two preprocessing pipelines**:
  - `scaled`: StandardScaler for models requiring normalization (SVM, Logistic Regression)
  - `imputed`: Mean imputation for tree-based models (Random Forest, XGBoost)

### 2. Model Training (train.py)
- **Default model configurations**:
  - Logistic Regression (C, solver, max_iter, class_weight)
  - Random Forest (n_estimators, max_depth, min_samples_split, class_weight)
  - XGBoost (n_estimators, max_depth, learning_rate, subsample, scale_pos_weight)
  - SVM (C, gamma, kernel, class_weight)
- **Hyperparameter tuning** using GridSearchCV with 5-fold cross-validation
- **Flexible training options**:
  - Train all models
  - Train specific models from default configs
  - Train with custom parameter grids
  - Train single model
  - Conditional training scenarios (fast/balanced/comprehensive)

### 3. Model Evaluation (evaluate.py)
- **Comprehensive metrics**:
  - ROC-AUC, AUPRC
  - Accuracy, Precision, Recall, F1-Score
  - PPV (Positive Predictive Value), NPV, Specificity
  - Confusion matrix components (TP, FP, TN, FN)
- **Auto-detection** of data type required for each model
- **Model ranking** by ROC-AUC
- **Save best model** and evaluation results

### 4. Visualization (visualization.py)
- **ROC curves** comparison across models
- **Precision-Recall curves** comparison
- **Model comparison** bar charts (multiple metrics)
- **Confusion matrices** for all models
- **Enhanced styling** with legends, grids, and performance summaries

### 5. Model Interpretability (shap_explainer.py)
- **SHAP explanations** for different model types:
  - TreeExplainer for XGBoost and Random Forest
  - LinearExplainer for Logistic Regression
  - KernelExplainer for SVM
- **Feature importance analysis**
- **Coefficient analysis** for Logistic Regression
- **Save SHAP values** for later analysis

## Key Features

### Strengths
1. **Modular architecture** - Clean separation of concerns
2. **Flexible training** - Multiple options for model selection
3. **Comprehensive evaluation** - Multiple metrics for thorough assessment
4. **Model interpretability** - SHAP integration for feature importance
5. **Well-documented** Example notebooks and README
6. **Auto-detection** - Intelligent data type selection for models
7. **Imbalanced dataset handling** - Using class_weight and scale_pos_weight

### Current Dataset Characteristics
- **Total samples:** 3,989 records
- **Features:** 43 numerical features (after preprocessing)
- **AKI cases:** 210 (5.26% positive class - highly imbalanced)
- **AKI definition:** KDIGO Stage I (postop creatinine > 1.5 × preop creatinine)

### Model Performance (Example Results)
- **XGBoost:** ROC-AUC 0.8244, AUPRC 0.4744, F1-Score 0.4706
- **Logistic Regression:** ROC-AUC 0.7875, AUPRC 0.3110, F1-Score 0.2097
- **Random Forest:** ROC-AUC 0.7849, AUPRC 0.3060, F1-Score 0.1818
- **SVM:** ROC-AUC 0.6316, AUPRC 0.2760, F1-Score 0.0000

## Current Status

### Completed
- ✅ Data loading and preprocessing pipeline
- ✅ Model training with hyperparameter tuning
- ✅ Comprehensive evaluation framework
- ✅ Visualization tools
- ✅ SHAP interpretability
- ✅ Example notebooks
- ✅ README documentation
- ✅ Import testing script
- ✅ Data visualization notebook with time-series vital signs
- ✅ Flowchart design and color scheme
- ✅ Research paper outlines (Introduction, Method, Results)
- ✅ LaTeX project (compiled PDF, 9 pages)
- ✅ Medical Dashboard (Dash) - Complete implementation

### Missing Components
- ❌ `best_models/` directory (not created yet)
- ❌ `results/` directory (not created yet)
- ❌ No automated data validation
- ❌ No model versioning system
- ❌ No unit tests (except import test)
- ❌ No continuous integration
- ❌ No logging system
- ❌ No performance monitoring

## Dependencies (requirements.txt)
- pandas >= 1.5.0
- numpy >= 1.21.0
- matplotlib >= 3.5.0
- seaborn >= 0.11.0
- scikit-learn >= 1.1.0
- xgboost >= 1.6.0
- shap >= 0.41.0
- jupyter >= 1.0.0
- ipykernel >= 6.0.0
- joblib >= 1.1.0
- plotly >= 5.0.0 (optional)
- pytest >= 7.0.0 (dev)
- black >= 22.0.0 (dev)
- flake8 >= 5.0.0 (dev)

## Change Log

### 2024-10-26 - Session 2: LaTeX Compilation Success
- Successfully resolved local LaTeX compilation issues
- Installed texlive-full on the system
- Updated `.vscode/settings.json` to use system TeX Live
- Modified `main.tex` to use natbib instead of biblatex for better compatibility
- Compiled complete research paper to PDF (9 pages)
- Output: `paper/out/main.pdf` (149KB)
- All sections included:
  - Introduction
  - Method (AXKI framework)
  - Results
  - Discussion
  - Conclusion
  - Bibliography with 10 references
- LaTeX project now fully functional locally
- Can edit `.tex` files and view compiled PDF side-by-side in VS Code

### 2024-12-01 - Evening Session: Comprehensive Granularity Documentation
- **Major update to `aki/Notes.md`**: Added comprehensive explanations of patient-level vs window-level approaches
  - **New Section 1.3**: "Understanding Granularity Levels"
    - Patient-Level Approach: Definition, characteristics, real-world example with clinical workflow
    - Window-Level Approach: Definition, characteristics, temporal feature extraction example
    - Combined Approach: Hybrid method with comprehensive clinical example
    - Comparison table: 8 aspects (granularity, features, timing, use cases, etc.)
  - **Expanded Section 8**: "Clinical Implications and Deployment Scenarios"
    - **Section 8.1**: Practical Applications by Granularity (6 detailed scenarios)
      - Patient-level: Preoperative screening, resource allocation
      - Window-level: Real-time alerts, hypotension pattern recognition
      - Combined: Comprehensive risk management, early warning systems
    - **Section 8.2**: Deployment Considerations (infrastructure, advantages, limitations, phased strategy)
  - **Added real-world examples** with specific patient data, predictions, and clinical actions
  - **Line count**: Expanded from 569 to 917 lines (+348 lines of new content)
- **Updated change log** in Notes.md with latest documentation additions

### 2024-12-01 - Session: Processed Features Folder & Save/Load Functionality
- **Created `aki/processed/` folder** for temporal feature storage:
  - Added `.gitignore` to exclude large CSV files from git
  - Folder ready to store extracted features
  - Prevents re-extraction of temporal features (saves 2.5+ hours)
- **Added save/load functionality to `Com_temporal_features_aki.ipynb`**:
  - **Cell 7**: Saves extracted temporal features to `processed/temporal_features_aki.csv` after extraction
  - **Cell 9**: Optional cell to load pre-saved features and skip 2.5-hour extraction
  - Features automatically saved: 130 temporal features for 3,989 cases
  - Workflow: Extract once → Save → Load on subsequent runs
- **Updated repository structure**: Added processed/ folder to backup-context.md documentation

### 2024-11-01 - Session: Temporal Features Analysis & Documentation
- **Created comprehensive research notes**: `aki/Notes.md`
  - Complete research overview and methodology
  - Experimental results: Tabular vs Temporal vs Combined features
  - Key findings: Temporal features improve ROC-AUC by 3.9-15% when combined
  - AXKI framework documentation
  - Clinical implications and future directions
- **Renamed notebooks with granularity prefixes**:
  - `Pat_*`: Patient-level experiments (7 notebooks)
  - `Win_*`: Window-level experiments (1 notebook)
  - `Com_*`: Combined experiments (1 notebook)
- **Temporal features experiment results**:
  - Best model: XGBoost on Combined Features (ROC-AUC: 0.7873)
  - Combined features: 173 features (43 tabular + 130 temporal)
  - Signal coverage: PLETH_HR (99.9%), PLETH_SPO2 (100%), ART_MBP (70.6%)
  - Key finding: Temporal features alone insufficient, but beneficial when combined
- **Updated documentation**:
  - README.md: Added Notes.md reference, updated notebook structure
  - backup-context.md: Updated project structure and recent findings

### 2024-12-19 - Session 1: Created Data Visualization Notebook
- Created `notebooks/data_vis.ipynb` with comprehensive VitalDB dataset visualizations
- Features include:
  - Dataset loading and AKI target creation
  - Comprehensive visualization dashboard (10 subplots):
    - Class distribution (bar and pie charts)
    - Missing values analysis
    - Age, creatinine, BMI, and sex distributions
    - Preoperative vs postoperative creatinine scatter plot
    - Feature statistics summary
  - Feature correlation analysis with heatmap
  - Statistical summary table
  - Key insights and recommendations
  - **NEW**: Time-series signal visualization section:
    - Generate realistic synthetic vital signs time-series data
    - Multi-signal comparison view with moving averages
    - Individual detailed signal views with smoothing and IQR bands
    - Vital signs correlation matrix
    - Statistical summaries for each vital sign
    - Note: VitalDB time-series API requires authentication; using synthetic data for demo
- Fixed KeyError in correlation analysis: Added check to include 'aki' column as numeric type
- Applied flowchart color scheme to time-series visualizations:
  - Cell 14: Multi-signal comparison view with medical color palette
  - Cell 16: Individual detailed view with flowchart colors
  - Background colors, text colors, grid colors aligned with medical theme
  - Color assignment: ART_SBP (Red), ART_DBP (Orange), PLETH_HR (Blue), PLETH_SPO2 (Teal), ECO2_ETCO2 (Purple)
- Created flowchart guide and color reference for research paper:
  - Created `Draw/flowchart_guide.md` with complete ML-focused flowchart structure
  - Created `Draw/color_codes.txt` with ready-to-use color codes
  - High-level overview showing: Input → Preprocessing → Data Pipeline → ML Models → Evaluation → SHAP → Output
  - Professional color scheme aligned with medical/clinical theme
  - Includes draw.io implementation instructions and best practices
- Proposed method name: **AXKI** (eXplainable AI and Machine learning for Acute Kidney Injury)
- Created research paper outline:
  - Created `Draw/method_outline_vietnamese.md` with comprehensive Vietnamese outline for "Proposed Method" section
  - Created `Draw/method_outline_english.md` with English version
  - Based on flowchart analysis (AXKI framework with 5 main stages matching the flowchart)
  - Reorganized structure to match flowchart exactly:
    1. Đầu vào dữ liệu sinh hiệu lâm sàng (Section 3.2)
    2. Tiền xử lý dữ liệu (Section 3.3)
    3. Quy trình dự đoán (Section 3.4)
    4. Quy trình lựa chọn với XAI (Section 3.5)
    5. Quy trình quyết định lâm sàng (Section 3.6)
  - Includes technical details, citations, and formatting guidelines
- Wrote paragraph version of Section 3.2 (Input Data):
  - Converted bullet points into coherent paragraph
  - Added 3 key references: VitalDB dataset [1], KDIGO guidelines [2], AKI studies [3]
  - Included citation instructions for Word document
  - Provided both APA and Vietnamese citation formats
  - Added bibliography creation guide
- Created Experiments and Results outline:
  - Created `Draw/experiments_results_outline.md `with comprehensive results section structure (full version for journal)
  - Created `Draw/experiments_results_outline_simplified.md` (condensed version for small conference)
  - Full version: 10 subsections, 10-15 pages, 10 figures, 10 tables
  - Simplified version: 4 subsections, 2-3 pages, 3-4 figures, 2 tables
  - Based on actual results from notebooks (XGBoost ROC-AUC: 0.82-0.84, Random Forest: 0.78-0.83)
  - Includes formatting guidelines, writing tips, templates, and timeline
  - Comparison table showing differences between versions
- Created Method section simplified outline:
  - Created `Draw/method_outline_simplified.md` for sections 3.3 onwards
  - Reduced from 5-7 pages to 1.5-2 pages
  - Sections 3.3-3.7 condensed to single paragraphs each
  - Includes complete paragraph templates for each section
  - Comparison table: full vs simplified structure
  - Timeline: 2-3 hours to write
  - Checklist for submission
  - Ready-to-use templates for rapid writing
- Created Introduction section simplified outline:
  - Created `Draw/introduction_outline_simplified.md`
  - 3-4 subsections: Background, Problem Statement, Contributions, Organization
  - Reduced from 2-3 pages to 1-1.5 pages
  - Bullet-point outlines (not templates) for each section
  - Detailed references with Vietnamese citation formats:
    - 10 specific references covering AKI, ML, explainability, and datasets
    - Each reference includes Vietnamese format for easy copy-paste
    - Example usage paragraphs showing how to cite in text
    - Word document citation guide
    - Google Scholar search guide for finding papers
  - Key statistics for AKI background
  - Writing guidelines and tips
  - Timeline: 4-6 hours (1 day)
  - Opening and closing sentence options
- Created LaTeX project simulating Overleaf locally:
  - Created `paper/` folder structure with complete LaTeX project
  - Main file: `paper/main.tex` with complete structure
  - Section files: Introduction, Method, Results, Discussion, Conclusion
  - VS Code configuration: `.vscode/settings.json` for LaTeX Workshop
  - Compilation config: `paper/.latexmkrc`
  - README with instructions for VS Code usage
  - All content based on simplified outlines
  - Ready to compile and view PDF side-by-side
  - All sections with proper citations from outlines
- Updated backup-context.md with session details
- Created VS Code LaTeX Workshop configuration (`.vscode/settings.json`)
- Switched from biblatex to natbib for compatibility
- Successfully compiled LaTeX paper to PDF (9 pages, 149KB)
- Fixed compilation issues by using system TeX Live instead of conda texlive-core
- Generated final PDF output at `paper/out/main.pdf`
- **NEW: Created AXKI Medical Dashboard**
  - Built complete Dash (Plotly) dashboard for AKI prediction
  - Location: `dashboard/` folder
  - Three-panel layout:
    - Left: Real-time vital signs visualization (Plotly)
    - Center: ML model selection and prediction results
    - Right: AI chatbot with SHAP explanations
  - Features:
    - 5 ML models + Traditional AKI Score
    - 4 patient scenarios (low/medium/high risk)
    - Synthetic vital signs data generator
    - Mock prediction engine with realistic probabilities
    - SHAP waterfall plot generation
    - Pre-scripted AI responses with clinical recommendations
    - Medical-grade professional UI
  - Files created:
    - `dashboard/app.py` - Main application
    - `dashboard/components/` - UI components
    - `dashboard/utils/` - Data and prediction utilities
    - `dashboard/assets/styles.css` - Custom medical styling
    - `dashboard/README.md` - Complete documentation
    - `dashboard/requirements_dashboard.txt` - Dependencies
  - Technology: Dash 2.14+, Plotly, Bootstrap Components
  - Color scheme: Uses AXKI flowchart colors
  - Status: Fully implemented, ready to run
  - Run with: `cd dashboard && python app.py`
  - Access at: http://localhost:8050

### 2024-12-19 - Initial Context Capture
- Created backup-context.md with comprehensive project overview
- Documented complete workflow from data loading to model interpretability
- Identified strengths and missing components
- Documented dataset characteristics and model performance

## Recommendations for Future Improvements

1. **Create output directories** (`best_models/`, `results/`) with .gitkeep files
2. **Add logging** for better debugging and tracking
3. **Implement data validation** checks before preprocessing
4. **Add unit tests** for each module
5. **Implement model versioning** for production deployment
6. **Add performance monitoring** for deployed models
7. **Create data pipeline** for automated retraining
8. **Add configuration file** (YAML/JSON) for parameters
9. **Implement early stopping** for hyperparameter tuning
10. **Add more evaluation metrics** (calibration curves, Brier score)

## Next Steps

1. Review current process ✅
2. Implement missing components
3. Add comprehensive tests
4. Set up CI/CD pipeline
5. Deploy for production use

---

## Project 2: Arrhythmia Classification (ARRDB)

### Project Overview

**Project Name:** Multi-Level Arrhythmia Classification from ECG Signals  
**Purpose:** Classify cardiac arrhythmias at beat-level and rhythm-level using deep learning and traditional ML  
**Technology Stack:** Python, PyTorch, scikit-learn, XGBoost, pandas, numpy, matplotlib

### Key Features

**Classification Tasks:**
- **Beat-level**: 4 classes (N=Normal, S=Supraventricular, V=Ventricular, U=Unknown)
- **Rhythm-level**: Multiple rhythm types (AFIB/AFL, SR, etc.)

**Models:**
- **Deep Learning**: 1D-CNN, LSTM (PyTorch)
- **Traditional ML**: XGBoost, Random Forest, Logistic Regression

**Data Processing:**
- Window-based approach (60-beat windows, stride=30)
- Patient-level splits (60/20/20) to prevent data leakage
- HRV feature extraction for ML models
- Raw RR sequence input for DL models

### Current Dataset Characteristics

- **Total Patients**: 482 cases
- **Input Format**: RR intervals from ECG R-peak annotations
- **Window Parameters**: window_size=60, stride=30 (50% overlap)
- **Evaluation**: Window-level for fair comparison

### Model Performance

**Beat Classification:**
- **CNN**: Accuracy 88.21%, F1-Macro 51.95%, AUROC 86.65%
- **XGBoost**: Accuracy 76.17%, F1-Macro 44.24%, AUROC 91.53%
- **LSTM**: Accuracy 75.18%, F1-Macro 34.11%, AUROC 69.94%

**Rhythm Classification:**
- **CNN**: Accuracy 70.82%, F1-Macro 50.04%, AUROC 93.07%
- **LogisticRegression**: Accuracy 70.48%, F1-Macro 30.46%, AUROC 88.43%
- **LSTM**: Accuracy 62.25%, F1-Macro 31.82%, AUROC 82.42%

### Current Status

**Completed:**
- ✅ Complete data preprocessing pipeline (windowing, filtering)
- ✅ Deep learning models (CNN, LSTM) for both tasks
- ✅ Traditional ML models (XGBoost, RF, LR) for both tasks
- ✅ Window-level feature extraction and evaluation
- ✅ Comprehensive evaluation metrics (9 metrics per model)
- ✅ Model comparison framework
- ✅ Visualization notebooks (DL and ML)
- ✅ Execution guide (EXP_GUIDE.md)
- ✅ Research notes and paper draft (Notes.md)
- ✅ Saved model predictions and metrics

**Project Structure:**
- All source code in `arrdb/src/`
- 9 experiment notebooks in `arrdb/notebooks/`
- Results saved in `arrdb/experiments/results/`
- Documentation: EXP_GUIDE.md, Notes.md

### Execution Workflow

1. **Phase 1: Model Training** (sequential)
   - `beat_dl.ipynb` - Train CNN for beat classification
   - `beat_lstm.ipynb` - Train LSTM for beat classification
   - `rhythm_dl.ipynb` - Train CNN for rhythm classification
   - `rhythm_lstm.ipynb` - Train LSTM for rhythm classification
   - `trad_ml.ipynb` - Train ML models for both tasks

2. **Phase 2: Visualization** (optional, can run in parallel)
   - `classification_visualization.ipynb` - DL model visualizations
   - `ml_visualization.ipynb` - ML model visualizations

3. **Phase 3: Comprehensive Evaluation**
   - `general_evaluation.ipynb` - Compare all models

### Key Technical Details

- **Data Splitting**: Patient-level (random_state=42) ensures no data leakage
- **Feature Extraction**: HRV features (10 features) for ML, raw sequences for DL
- **Class Handling**: Class weights for imbalance, filtering of rare classes
- **Evaluation**: Window-level metrics for fair comparison across all models
- **Model Persistence**: Saved predictions, models, and encoders in `experiments/results/predictions/`

---

## Notes

- Both projects are well-structured and follow ML/DL best practices
- Current focus is on research and development
- AKI project: Ready for production deployment after adding missing components
- ARRDB project: Complete experimental framework, ready for publication
- Strong documentation and examples available for both projects

