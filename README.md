# Vital Sign Dataset Projects

This repository contains multiple medical AI research projects based on the VitalDB dataset, focusing on predictive healthcare analytics using machine learning and deep learning approaches.

---

## 📋 Projects Overview

This repository hosts two independent research projects:

1. **AKI Prediction** (`aki/`) - Acute Kidney Injury prediction from vital signs
2. **Arrhythmia Classification** (`arrdb/`) - Cardiac arrhythmia detection from ECG signals

Both projects are self-contained with their own data, source code, notebooks, and documentation.

---

## 🔬 Project 1: AKI Prediction

### Overview
Predict postoperative Acute Kidney Injury (AKI) using vital signs and clinical data from VitalDB surgical patients.

### Key Features
- **Task**: Binary classification (AKI vs No-AKI)
- **Dataset**: 3,989 surgical patients, 43 features, 5.26% positive class (highly imbalanced)
- **Models**: Logistic Regression, Random Forest, XGBoost, SVM
- **Special Features**:
  - SHAP-based model interpretability
  - Interactive medical dashboard (Dash/Plotly)
  - Comprehensive evaluation metrics
  - Hyperparameter tuning framework

### Project Structure
```
aki/
├── src/                    # Source code package
│   ├── utils.py           # Data loading and preprocessing
│   ├── train.py           # Model training and hyperparameter tuning
│   ├── evaluate.py        # Model evaluation and metrics
│   ├── visualization.py   # Plotting and visualization
│   └── shap_explainer.py  # SHAP-based interpretability
├── notebooks/             # Jupyter notebooks
│   ├── Pat_*.ipynb       # Patient-level experiments
│   ├── Win_*.ipynb       # Window-level experiments
│   └── Com_*.ipynb       # Combined (patient + window) experiments
├── dashboard/             # Interactive medical dashboard
│   ├── app.py            # Main dashboard application
│   ├── components/       # UI components
│   └── utils/            # Dashboard utilities
├── paper/                # LaTeX research paper
│   ├── main.tex         # Main document
│   └── sections/        # Paper sections
├── shap_plots/          # SHAP visualization outputs
├── Notes.md             # Research notes and findings
└── README.md            # Detailed AKI project documentation
```

### Getting Started

**1. Install Dependencies:**
```bash
cd aki
pip install -r requirements.txt  # Check if exists, otherwise use root requirements.txt
```

**2. Run Data Visualization:**
```bash
jupyter notebook notebooks/data_vis.ipynb
```

**3. Train Models:**
```bash
jupyter notebook notebooks/example_train.ipynb
```

**4. Launch Interactive Dashboard:**
```bash
cd dashboard
pip install -r requirements_dashboard.txt
python app.py
# Access at: http://localhost:8050
```

### Key Results
- **Best Model (Combined Features)**: XGBoost (ROC-AUC: 0.7873, PR-AUC: 0.2282)
- **Temporal Features Impact**: Combined features improve ROC-AUC by 3.9-15% vs tabular-only
- **Model Performance**: Patient-level models evaluated; temporal features enhance baseline
- **SHAP Interpretability**: Feature importance analysis for all models

### Documentation
- **Research Notes**: `aki/Notes.md` - Complete research summary, findings, and methodology
- See `aki/README.md` for detailed project documentation
- Research paper: `aki/paper/main.tex` (compiled PDF available)

---

## ❤️ Project 2: Arrhythmia Classification (ARRDB)

### Overview
Multi-level arrhythmia classification from ECG signals using both deep learning and traditional machine learning approaches.

### Key Features
- **Tasks**:
  - **Beat-level Classification**: 4 classes (N=Normal, S=Supraventricular, V=Ventricular, U=Unknown)
  - **Rhythm-level Classification**: Multiple rhythm types (AFIB/AFL, SR, etc.)
- **Dataset**: 482 patients, 60-beat window sequences, window-level evaluation
- **Models**:
  - **Deep Learning**: 1D-CNN, LSTM (PyTorch)
  - **Traditional ML**: XGBoost, Random Forest, Logistic Regression
- **Special Features**:
  - Window-level feature extraction (HRV features for ML, raw RR sequences for DL)
  - Patient-level data splits for fair comparison
  - Comprehensive evaluation metrics (9 metrics per model)
  - Model comparison and visualization framework

### Project Structure
```
arrdb/
├── src/                           # Source code package
│   ├── data_loader.py            # Load VitalDB annotation files
│   ├── feature_extractor.py      # HRV feature extraction
│   ├── preprocess.py             # Data preprocessing and windowing
│   ├── models.py                 # PyTorch DL model architectures
│   ├── train_models.py           # Training functions
│   ├── train_models_simple.py    # Simplified ML training (no PyTorch)
│   └── evaluate_models.py        # Evaluation and metrics
├── notebooks/                     # Jupyter notebooks
│   ├── beat_dl.ipynb             # CNN for beat classification
│   ├── beat_lstm.ipynb           # LSTM for beat classification
│   ├── rhythm_dl.ipynb           # CNN for rhythm classification
│   ├── rhythm_lstm.ipynb         # LSTM for rhythm classification
│   ├── trad_ml.ipynb             # Traditional ML for both tasks
│   ├── classification_visualization.ipynb  # DL visualization
│   ├── ml_visualization.ipynb    # ML visualization
│   └── general_evaluation.ipynb  # Comprehensive model comparison
├── experiments/
│   └── results/
│       ├── predictions/          # Saved model predictions
│       ├── metrics/              # Performance metrics (CSV)
│       └── plots/                # Visualization figures
├── LabelFile/                    # ECG annotations and metadata
├── EXP_GUIDE.md                  # Step-by-step execution guide
├── Notes.md                      # Research notes and paper draft
└── requirements_arrdb.txt        # Project-specific dependencies
```

### Getting Started

**1. Install Dependencies:**
```bash
cd arrdb
pip install -r requirements_arrdb.txt
```

**2. Follow Execution Guide:**
```bash
# Read the experiment guide first
cat EXP_GUIDE.md
```

**3. Run Experiments (Sequential Order):**

**Phase 1: Model Training**
```bash
jupyter notebook notebooks/beat_dl.ipynb          # Train CNN for beats
jupyter notebook notebooks/beat_lstm.ipynb        # Train LSTM for beats
jupyter notebook notebooks/rhythm_dl.ipynb        # Train CNN for rhythm
jupyter notebook notebooks/rhythm_lstm.ipynb      # Train LSTM for rhythm
jupyter notebook notebooks/trad_ml.ipynb          # Train ML models
```

**Phase 2: Visualization (Optional)**
```bash
jupyter notebook notebooks/classification_visualization.ipynb  # DL viz
jupyter notebook notebooks/ml_visualization.ipynb              # ML viz
```

**Phase 3: Comprehensive Evaluation**
```bash
jupyter notebook notebooks/general_evaluation.ipynb  # Compare all models
```

### Key Results
- **Beat Classification Best**: CNN (Accuracy: 88.21%, F1-Macro: 51.95%)
- **Rhythm Classification Best**: CNN (Accuracy: 70.82%, F1-Macro: 50.04%)
- **Window-Level Evaluation**: All models evaluated at same granularity (60-beat windows)
- **Fair Comparison**: Identical patient splits (60/20/20) and window parameters

### Documentation
- **Execution Guide**: `arrdb/EXP_GUIDE.md` - Step-by-step notebook execution order
- **Research Notes**: `arrdb/Notes.md` - Complete research summary and paper draft
- **Results**: `arrdb/experiments/results/metrics/overall_performance_comparison.csv`

---

## 🗂️ Repository Structure

```
Vital_sign_Dataset/
├── aki/                    # Project 1: AKI Prediction
│   ├── src/               # Source code
│   ├── notebooks/         # Analysis notebooks
│   ├── dashboard/         # Interactive dashboard
│   ├── paper/             # Research paper
│   └── shap_plots/        # SHAP visualizations
│
├── arrdb/                 # Project 2: Arrhythmia Classification
│   ├── src/               # Source code
│   ├── notebooks/         # Experiment notebooks
│   ├── experiments/       # Results and outputs
│   ├── LabelFile/         # ECG data and annotations
│   ├── EXP_GUIDE.md       # Execution guide
│   └── Notes.md           # Research notes
│
├── requirements.txt       # Common Python dependencies
├── backup-context.md      # Project context backup
└── README.md             # This file
```

---

## 📦 Common Dependencies

Both projects share core dependencies (see `requirements.txt`):

```bash
pip install -r requirements.txt
```

**Core Libraries:**
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- jupyter, ipykernel
- joblib

**Project-Specific:**
- **AKI**: XGBoost, SHAP, plotly, dash
- **ARRDB**: PyTorch, XGBoost

---

## 🔬 Dataset Sources

Both projects use data from **VitalDB**:

### AKI Dataset
- **Source**: VitalDB surgical patient database
- **Type**: Clinical vital signs and laboratory values
- **Focus**: Postoperative AKI prediction
- **Access**: Requires VitalDB API access

### Arrhythmia Database
- **Source**: VitalDB Arrhythmia Database
- **Type**: ECG waveforms with R-peak annotations
- **Focus**: Beat-level and rhythm-level classification
- **Files**: Located in `arrdb/LabelFile/` (482 patient annotation files)

---

## 📊 Key Differences Between Projects

| Aspect | AKI Prediction | Arrhythmia Classification |
|--------|---------------|--------------------------|
| **Task Type** | Binary classification | Multi-class classification (2 tasks) |
| **Input Data** | Tabular vital signs | Time-series ECG signals (RR intervals) |
| **Models** | Traditional ML only | DL + Traditional ML |
| **Granularity** | Patient-level | Window-level (60-beat windows) |
| **Special Features** | SHAP, Dashboard | Window-level comparison, HRV features |
| **Evaluation** | Patient-level metrics | Window-level metrics |

---

## 🚀 Quick Start Examples

### AKI Prediction
```bash
# 1. Navigate to project
cd aki

# 2. Run data visualization
jupyter notebook notebooks/data_vis.ipynb

# 3. Train models
jupyter notebook notebooks/example_train.ipynb

# 4. Launch dashboard
cd dashboard && python app.py
```

### Arrhythmia Classification
```bash
# 1. Navigate to project
cd arrdb

# 2. Read execution guide
cat EXP_GUIDE.md

# 3. Train models (start with beat classification)
jupyter notebook notebooks/beat_dl.ipynb

# 4. Compare all models
jupyter notebook notebooks/general_evaluation.ipynb
```

---

## 📚 Documentation

### AKI Project
- **Main README**: `aki/README.md`
- **Dashboard Guide**: `aki/dashboard/README.md`
- **Paper**: `aki/paper/main.tex`

### ARRDB Project
- **Execution Guide**: `arrdb/EXP_GUIDE.md` (sequential notebook execution)
- **Research Notes**: `arrdb/Notes.md` (complete research summary)
- **Results**: `arrdb/experiments/results/` (metrics and visualizations)

---

## 🔮 Future Work

### Individual Projects
- **AKI**: Real-time monitoring integration, model versioning
- **ARRDB**: Ensemble methods, attention mechanisms, transfer learning

### Cross-Project
- Multi-task learning combining AKI and Arrhythmia predictions
- LLM-powered clinical decision support
- Integration with hospital EHR systems
- Real-time monitoring systems

---

## 📄 License

See individual project READMEs for license information.

---

## 👥 Contributing

Each project is independently maintained. Please refer to project-specific documentation for contribution guidelines.

---

## 📧 Contact

For questions about specific projects:
- **AKI Prediction**: See `aki/README.md`
- **Arrhythmia Classification**: See `arrdb/Notes.md` or `arrdb/EXP_GUIDE.md`
