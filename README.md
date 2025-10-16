# AKI Prediction Project

A comprehensive machine learning pipeline for predicting postoperative Acute Kidney Injury (AKI) using vital signs and clinical data from VitalDB.

## 🏥 Project Overview

This project implements state-of-the-art machine learning models to predict the risk of developing Acute Kidney Injury (AKI) after surgery. AKI is a serious complication that can lead to increased mortality and healthcare costs. Early prediction can help clinicians take preventive measures.

### Key Features

- **Multiple ML Models**: Logistic Regression, Random Forest, XGBoost, SVM
- **Hyperparameter Tuning**: Automated optimization using GridSearchCV
- **Comprehensive Evaluation**: ROC-AUC, AUPRC, Precision, Recall, F1-Score, PPV
- **Model Interpretability**: SHAP explanations for understanding feature importance
- **Modular Design**: Clean, reusable code structure
- **Easy-to-use Examples**: Simple training and evaluation notebooks

## 📊 Dataset

The project uses data from [VitalDB](https://vitaldb.net/), a comprehensive database of vital signs and clinical information from surgical patients.

### Features
- **75+ clinical features** including demographics, vital signs, and laboratory values
- **AKI Definition**: KDIGO Stage I (postop creatinine > 1.5 × preop creatinine)
- **Class Distribution**: ~5.3% positive cases (imbalanced dataset)

## 🚀 Quick Start

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/NEU-Bio-Research-Team/Vital_sign_Dataset.git
   cd Vital_sign_Dataset
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run example training**
   ```bash
   jupyter notebook notebooks/example_train.ipynb
   ```

### Basic Usage

```python
# Import the package
import sys
import os
sys.path.append(os.path.abspath('src'))

# Import functions from individual modules
from utils import setup_plotting, load_vitaldb_data, preprocess_data, prepare_train_test_data
from train import get_default_model_configs, hyperparameter_tuning, save_best_model
from evaluate import evaluate_models
from shap_explainer import explain_model_with_shap

# Setup
setup_plotting()

# Load and preprocess data
df = load_vitaldb_data()
X, y, feature_names = preprocess_data(df)
data_dict = prepare_train_test_data(X, y)

# Train models
models_config = get_default_model_configs()
tuned_models = hyperparameter_tuning(models_config, data_dict['X_train_dict'], data_dict['y_train'])

# Evaluate and save best model
results_df = evaluate_models(tuned_models, data_dict['X_test_dict'], data_dict['y_test'])
best_model_name, best_model = save_best_model(tuned_models, data_dict['X_test_dict'], data_dict['y_test'])

# Generate SHAP explanations
explain_model_with_shap(best_model, data_dict['X_test_dict']['imputed'], feature_names)
```

### Training Specific Models

Instead of training all models, you can train only specific models:

#### Option 1: Select from Default Configurations
```python
# Get all default configs
all_configs = get_default_model_configs()

# Select only specific models
specific_models = {
    'LogisticRegression': all_configs['LogisticRegression'],
    'XGBoost': all_configs['XGBoost']
    # Add more models as needed
}

# Train only selected models
tuned_models = hyperparameter_tuning(specific_models, data_dict['X_train_dict'], data_dict['y_train'])
```

#### Option 2: Custom Model Configurations
```python
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# Create custom configurations with simplified parameters for faster training
custom_models = {
    'LogisticRegression_Fast': {
        'model': LogisticRegression(random_state=0),
        'params': {
            'C': [0.1, 1, 10],  # Reduced parameter grid
            'solver': ['lbfgs'],
            'class_weight': [None, 'balanced']
        },
        'data_type': 'scaled'
    },
    'XGBoost_Fast': {
        'model': XGBClassifier(random_state=0, eval_metric='logloss'),
        'params': {
            'n_estimators': [50, 100],
            'max_depth': [3, 6],
            'learning_rate': [0.1, 0.2]
        },
        'data_type': 'imputed'
    }
}

tuned_models = hyperparameter_tuning(custom_models, data_dict['X_train_dict'], data_dict['y_train'])
```

#### Option 3: Single Model Training
```python
# Train only XGBoost
single_model = {
    'XGBoost_Only': {
        'model': XGBClassifier(random_state=0, eval_metric='logloss'),
        'params': {
            'n_estimators': [100, 200],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.05, 0.1, 0.2]
        },
        'data_type': 'imputed'
    }
}

tuned_models = hyperparameter_tuning(single_model, data_dict['X_train_dict'], data_dict['y_train'])
```

#### Option 4: Conditional Training Scenarios
```python
# Define different training scenarios
scenarios = {
    'fast': ['LogisticRegression'],  # Quick training
    'balanced': ['LogisticRegression', 'XGBoost'],  # Balanced speed/performance
    'comprehensive': ['LogisticRegression', 'RandomForest', 'XGBoost', 'SVM']  # Full training
}

# Choose scenario
selected_scenario = 'balanced'  # Change as needed

# Build configuration
all_configs = get_default_model_configs()
selected_models = {model: all_configs[model] for model in scenarios[selected_scenario] if model in all_configs}

tuned_models = hyperparameter_tuning(selected_models, data_dict['X_train_dict'], data_dict['y_train'])
```

### Testing Imports

Test if all imports work correctly:
```bash
python test_imports.py
```

## 📁 Project Structure

```
Vital_sign_Dataset/
├── src/                           # Source code package
│   ├── __init__.py               # Package initialization
│   ├── utils.py                  # Data loading and preprocessing utilities
│   ├── train.py                  # Model training and hyperparameter tuning
│   ├── evaluate.py               # Model evaluation and metrics
│   ├── visualization.py          # Plotting and visualization functions
│   └── shap_explainer.py         # SHAP-based model interpretability
├── notebooks/                     # Jupyter notebooks
│   ├── example_train.ipynb       # Training example
│   └── example_eval.ipynb        # Evaluation example
├── examples/                      # Python script examples
│   └── train_specific_models.py  # Examples for training specific models
├── best_models/                   # Saved trained models
├── results/                       # Prediction results and outputs
├── requirements.txt               # Python dependencies
└── README.md                     # This file
```

## 🔧 API Reference

### Core Functions

#### Data Loading and Preprocessing
- `load_vitaldb_data()`: Load data from VitalDB
- `preprocess_data(df)`: Preprocess raw data for ML
- `prepare_train_test_data(X, y)`: Split data and prepare for different model types

#### Model Training
- `get_default_model_configs()`: Get predefined model configurations
- `hyperparameter_tuning(models_config, X_train, y_train)`: Train models with hyperparameter tuning
- `save_model(model, model_name)`: Save trained model
- `load_model(model_name)`: Load saved model

#### Model Evaluation
- `evaluate_models(models_dict, X_test, y_test)`: Evaluate multiple models
- `calculate_comprehensive_metrics(y_true, y_pred, y_pred_proba)`: Calculate all metrics
- `print_evaluation_summary(results_df)`: Print formatted results

#### Visualization
- `plot_roc_curves(models_dict, X_test, y_test)`: Plot ROC curves
- `plot_pr_curves(models_dict, X_test, y_test)`: Plot Precision-Recall curves
- `plot_model_comparison(results_df)`: Compare models across metrics
- `plot_confusion_matrices(models_dict, X_test, y_test)`: Plot confusion matrices

#### SHAP Explanations
- `explain_model_with_shap(model, X_test, feature_names)`: Generate SHAP explanations
- `explain_best_model_with_shap(models_dict, X_test, feature_names)`: Explain best model
- `analyze_logistic_regression_coefficients(model, feature_names)`: Analyze LR coefficients

## 📈 Model Performance

### Best Model Results (Example)
| Model | ROC-AUC | AUPRC | Accuracy | F1-Score | Precision | Recall |
|-------|---------|-------|----------|----------|-----------|--------|
| XGBoost | 0.8244 | 0.4744 | 0.9436 | 0.4706 | 0.4878 | 0.4545 |
| Logistic Regression | 0.7875 | 0.3110 | 0.7356 | 0.2097 | 0.1256 | 0.6364 |
| Random Forest | 0.7849 | 0.3060 | 0.9436 | 0.1818 | 0.1000 | 0.5000 |
| SVM | 0.6316 | 0.2760 | 0.9398 | 0.0000 | 0.0000 | 0.0000 |

### Key Insights
- **XGBoost** performs best with highest ROC-AUC and F1-Score
- **Class imbalance** affects precision but models maintain good discrimination (ROC-AUC)
- **SHAP analysis** reveals most important features for AKI prediction

## 🔍 Model Interpretability

The project includes comprehensive SHAP-based model interpretability:

- **Feature Importance**: Identify which clinical features are most predictive
- **Individual Predictions**: Understand why specific patients are predicted as high-risk
- **Model Comparison**: Compare feature importance across different models
- **Clinical Insights**: Translate ML findings into actionable clinical knowledge

### Example SHAP Features
- Preoperative creatinine levels
- Age and demographics
- Vital signs during surgery
- Laboratory values
- Surgical duration and complexity

## 🛠️ Development

### Adding New Models

1. **Add model configuration** in `src/train.py`:
   ```python
   'NewModel': {
       'model': NewModelClassifier(),
       'params': {'param1': [values], 'param2': [values]},
       'data_type': 'scaled'  # or 'imputed'
   }
   ```

2. **Update data mapping** in evaluation functions
3. **Test with example notebooks**

### Custom Metrics

Add new metrics in `src/evaluate.py`:
```python
def calculate_custom_metric(y_true, y_pred):
    # Your metric calculation
    return metric_value
```

### Custom Visualizations

Add new plots in `src/visualization.py`:
```python
def plot_custom_visualization(data, **kwargs):
    # Your visualization code
    plt.show()
```

## 📝 Usage Instructions

### For Researchers

1. **Training New Models**:
   ```bash
   jupyter notebook notebooks/example_train.ipynb
   ```

2. **Evaluating Existing Models**:
   ```bash
   jupyter notebook notebooks/example_eval.ipynb
   ```

3. **Custom Analysis**:
   ```python
   from src import *
   # Your custom code here
   ```

### For Developers

1. **Install in development mode**:
   ```bash
   pip install -e .
   ```

2. **Run tests**:
   ```bash
   pytest tests/
   ```

3. **Code formatting**:
   ```bash
   black src/
   flake8 src/
   ```

## 🤝 Contributing

We welcome contributions! Please follow these steps:

### 1. Fork the Repository
```bash
git clone https://github.com/YOUR_USERNAME/Vital_sign_Dataset.git
cd Vital_sign_Dataset
```

### 2. Create a Feature Branch
```bash
git checkout -b feature/new-feature
```

### 3. Make Changes
- Add new functionality
- Update documentation
- Add tests if applicable

### 4. Commit Changes
```bash
git add .
git commit -m "Add new feature: description"
```

### 5. Push and Create Pull Request
```bash
git push origin feature/new-feature
```

Then create a pull request on GitHub.

## 📋 Commit Guidelines

Please follow these commit message conventions:

- `feat:` New features
- `fix:` Bug fixes
- `docs:` Documentation updates
- `style:` Code formatting
- `refactor:` Code refactoring
- `test:` Adding tests
- `chore:` Maintenance tasks

Examples:
```bash
git commit -m "feat: add new SHAP visualization function"
git commit -m "fix: resolve dimension mismatch in SHAP explanation"
git commit -m "docs: update README with new API reference"
```

## 🚀 Deployment Instructions

### For Project Members

1. **Setup Development Environment**:
   ```bash
   git clone https://github.com/NEU-Bio-Research-Team/Vital_sign_Dataset.git
   cd Vital_sign_Dataset
   pip install -r requirements.txt
   ```

2. **Make Changes and Test**:
   ```bash
   jupyter notebook notebooks/example_train.ipynb
   ```

3. **Commit and Push**:
   ```bash
   git add .
   git commit -m "Your commit message"
   git push origin main
   ```

### For Production Use

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Load Pre-trained Models**:
   ```python
   from src import load_model
   best_model = load_model('XGBoost_best')
   ```

3. **Make Predictions**:
   ```python
   predictions = best_model.predict(new_data)
   ```

## 📊 Results and Outputs

The project generates several types of outputs:

- **Models**: Saved in `best_models/` directory
- **Predictions**: CSV files in `results/` directory
- **Visualizations**: PNG plots for ROC curves, PR curves, confusion matrices
- **SHAP Plots**: Feature importance visualizations
- **Evaluation Results**: Performance metrics in CSV format

## 🏆 Acknowledgments

- **VitalDB Team**: For providing the comprehensive vital signs database
- **NEU Bio Research Team**: For project development and maintenance
- **Open Source Community**: For the excellent ML libraries used in this project

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Contact

- **Project Maintainer**: NEU Bio Research Team
- **Email**: contact@example.com
- **GitHub**: [https://github.com/NEU-Bio-Research-Team/Vital_sign_Dataset](https://github.com/NEU-Bio-Research-Team/Vital_sign_Dataset)

## 🔗 Related Resources

- [VitalDB Documentation](https://vitaldb.net/)
- [KDIGO AKI Guidelines](https://kdigo.org/guidelines/acute-kidney-injury/)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)

---

**Note**: This project is for research purposes. Please consult with medical professionals before using predictions for clinical decision-making.
