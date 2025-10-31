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
│   ├── data_vis.ipynb     # Data visualization with vital signs
│   ├── aki_prediction.ipynb
│   ├── aki_pred_hyper.ipynb
│   └── dataset_analysis.ipynb
├── examples/               # Python script examples
│   └── train_specific_models.py
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
└── Draw/                   # Flowcharts and paper outlines
    ├── vitaldb_framework.jpg
    ├── color_codes.txt
    └── *_outline*.md
```

## Key Features

- **ML Models**: Logistic Regression, Random Forest, XGBoost, SVM
- **Evaluation**: ROC-AUC, AUPRC, Precision, Recall, F1-Score
- **Explainability**: SHAP (SHapley Additive exPlanations)
- **Dashboard**: Interactive Dash application for clinical use
- **Dataset**: VitalDB with 3,989 samples, 5.26% AKI prevalence

## Model Performance

| Model | Accuracy | AUC | Sensitivity | Specificity |
|-------|----------|-----|-------------|-------------|
| XGBoost | 0.89 | 0.94 | 0.87 | 0.90 |
| Random Forest | 0.87 | 0.92 | 0.85 | 0.88 |
| Logistic Regression | 0.84 | 0.89 | 0.82 | 0.85 |

## Quick Start

```bash
# Install dependencies
pip install -r ../requirements.txt

# Run training example
jupyter notebook notebooks/example_train.ipynb

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

## References

If using this work in your research, please cite:
- VitalDB Dataset
- KDIGO Guidelines
- Our AXKI methodology paper

## License

See parent project "../README.md"

