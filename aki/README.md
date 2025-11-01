# AKI Prediction Project

Acute Kidney Injury (AKI) Risk Prediction System using Machine Learning and Explainable AI.

## Project Overview

This project predicts postoperative AKI using vital signs and clinical data from VitalDB. The system combines multiple ML models with SHAP-based explainability to provide interpretable predictions for clinical decision-making.

## Project Structure

```
aki/
├── src/                    # Source code package
│   ├── utils.py           # Data loading and preprocessing
│   ├── train.py            # Model training and hyperparameter tuning
│   ├── evaluate.py         # Model evaluation and metrics
│   ├── visualization.py   # Plotting and visualization functions
│   └── shap_explainer.py  # SHAP-based model interpretability
├── notebooks/              # Jupyter notebooks
│   ├── Pat_*.ipynb        # Patient-level experiments
│   │   ├── Pat_dl_examination.ipynb      # DL vs ML comparison
│   │   ├── Pat_aki_prediction.ipynb      # Original AKI prediction
│   │   ├── Pat_aki_pred_hyper.ipynb      # Hyperparameter tuning
│   │   ├── Pat_dataset_analysis.ipynb    # Dataset analysis
│   │   ├── Pat_data_vis.ipynb            # Data visualization
│   │   ├── Pat_example_train.ipynb       # Training example
│   │   └── Pat_example_eval.ipynb        # Evaluation example
│   ├── Win_*.ipynb        # Window-level experiments
│   │   └── Win_windowaki_examine.ipynb   # Temporal data exploration
│   └── Com_*.ipynb        # Combined experiments
│       └── Com_temporal_features_aki.ipynb  # Temporal feature extraction
├── examples/               # Python script examples
├── shap_plots/             # SHAP visualization outputs
├── paper/                  # LaTeX research paper
│   ├── main.tex
│   ├── sections/
│   ├── references.bib
│   └── out/main.pdf
├── dashboard/              # Interactive medical dashboard
│   ├── app.py
│   ├── components/
│   ├── utils/
│   └── assets/
├── Draw/                   # Flowcharts and paper outlines
│   ├── vitaldb_framework.jpg
│   ├── color_codes.txt
│   └── *_outline*.md
├── Notes.md                # Research notes and findings
└── README.md               # This file
```

## Key Features

- **ML Models**: Logistic Regression, Random Forest, XGBoost, SVM
- **Evaluation**: ROC-AUC, AUPRC, Precision, Recall, F1-Score
- **Explainability**: SHAP (SHapley Additive exPlanations)
- **Dashboard**: Interactive Dash application for clinical use
- **Dataset**: VitalDB with 3,989 samples, 5.26% AKI prevalence
- **Temporal Features**: Intraoperative vital sign analysis with window-level feature extraction
- **Multiple Approaches**: Patient-level, Window-level, and Combined feature analysis

## Model Performance

### Combined Features Approach (Best Performance)
| Model | ROC-AUC | PR-AUC | Accuracy | F1-Score |
|-------|---------|--------|----------|----------|
| XGBoost | **0.7873** | 0.2282 | 0.9411 | 0.0784 |
| Random Forest | **0.7778** | 0.1820 | 0.9424 | 0.0000 |
| Logistic Regression | **0.7562** | 0.2026 | 0.9424 | 0.1786 |

### Key Findings
- **Temporal Features Impact**: Combined features improve ROC-AUC by 3.9-15% vs tabular-only baseline
- **Best Configuration**: XGBoost on combined features (173 total: 43 tabular + 130 temporal)
- **Class Imbalance**: Remains challenging (high accuracy but low F1-scores due to 5.26% prevalence)

## Quick Start

```bash
# Install dependencies
pip install -r ../requirements.txt

# Run training example (Patient-level)
jupyter notebook notebooks/Pat_example_train.ipynb

# Explore temporal features (Combined)
jupyter notebook notebooks/Com_temporal_features_aki.ipynb

# Launch dashboard
cd dashboard && python app.py
```

## Methodology: AXKI Framework

**AXKI** (eXplainable AI and Machine learning for Acute Kidney Injury) is our proposed method combining:
1. Clinical vital signs data from VitalDB
2. Preprocessing (imputation, scaling, stratified splitting)
3. ML-based prediction with 5 models
4. XAI-based model selection using SHAP
5. Clinical decision support with explainable outputs

## Documentation

- **Research Notes**: `Notes.md` - Complete research summary, methodology, and findings
- **Temporal Features**: See `notebooks/Com_temporal_features_aki.ipynb` for window-level feature extraction
- **Patient-Level**: See `notebooks/Pat_*.ipynb` for traditional approaches
- **Window-Level**: See `notebooks/Win_windowaki_examine.ipynb` for temporal data exploration

## References

If using this work in your research, please cite:
- VitalDB Dataset: https://vitaldb.net/
- KDIGO Clinical Practice Guideline for Acute Kidney Injury
- AXKI Framework methodology

## License

See parent project "../README.md"

