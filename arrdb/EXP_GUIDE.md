# Experiment Guide: Step-by-Step Notebook Execution

This guide provides the correct sequential order for running all notebooks in the `arrdb/notebooks/` folder. Follow this order to ensure all dependencies are met and predictions/models are generated correctly.

---

## 📋 Overview

The notebooks are organized into **3 phases**:

1. **Phase 1: Model Training** - Train deep learning and traditional ML models
2. **Phase 2: Individual Visualizations** - Visualize results from specific model types
3. **Phase 3: Comprehensive Evaluation** - Compare all models together

---

## 🚀 Phase 1: Model Training

These notebooks train models and save predictions. **You must complete Phase 1 before running Phase 2 or 3**.

### Step 1.1: Train CNN for Beat Classification
**Notebook**: `beat_dl.ipynb`

**Purpose**: Train a 1D-CNN model for beat-level classification (4 classes: N, S, V, U)

**What it does**:
- Loads all patient cases from VitalDB
- Creates 60-beat window sequences (window_size=60, stride=30)
- Trains CNN model with class weights for imbalance handling
- Evaluates on test set
- Saves predictions to: `experiments/results/predictions/beat_classification_predictions.pkl`

**Output files**:
- `beat_classification_predictions.pkl`

**Estimated time**: ~10-20 minutes (depending on GPU)

---

### Step 1.2: Train LSTM for Beat Classification
**Notebook**: `beat_lstm.ipynb`

**Purpose**: Train an LSTM model for beat-level classification (4 classes: N, S, V, U)

**What it does**:
- Similar to `beat_dl.ipynb` but uses LSTM architecture
- Creates same 60-beat window sequences
- Trains LSTM model
- Evaluates on test set
- Saves predictions to: `experiments/results/predictions/beat_classification_lstm_predictions.pkl`

**Output files**:
- `beat_classification_lstm_predictions.pkl`

**Estimated time**: ~15-25 minutes (depending on GPU)

---

### Step 1.3: Train CNN for Rhythm Classification
**Notebook**: `rhythm_dl.ipynb`

**Purpose**: Train a 1D-CNN model for rhythm-level classification (multiple rhythm classes)

**What it does**:
- Loads all patient cases
- Creates 60-beat window sequences for rhythm classification
- Trains CNN model for rhythm classification
- Evaluates on test set
- Saves predictions to: `experiments/results/predictions/rhythm_classification_predictions.pkl`

**Output files**:
- `rhythm_classification_predictions.pkl`

**Estimated time**: ~10-20 minutes (depending on GPU)

---

### Step 1.4: Train LSTM for Rhythm Classification
**Notebook**: `rhythm_lstm.ipynb`

**Purpose**: Train an LSTM model for rhythm-level classification

**What it does**:
- Similar to `rhythm_dl.ipynb` but uses LSTM architecture
- Creates same 60-beat window sequences
- Trains LSTM model for rhythm classification
- Evaluates on test set
- Saves predictions to: `experiments/results/predictions/rhythm_classification_lstm_predictions.pkl`

**Output files**:
- `rhythm_classification_lstm_predictions.pkl`

**Estimated time**: ~15-25 minutes (depending on GPU)

---

### Step 1.5: Train Traditional ML Models
**Notebook**: `trad_ml.ipynb`

**Purpose**: Train traditional ML models (XGBoost, RandomForest, LogisticRegression) for both beat and rhythm classification

**What it does**:
- Loads all patient cases
- Creates 60-beat window sequences (same as DL models)
- Extracts HRV features from each window
- Trains XGBoost, RandomForest, and LogisticRegression models for both tasks
- Evaluates all models on test set
- Saves predictions and trained models:
  - `beat_classification_ml_predictions.pkl`
  - `rhythm_classification_ml_predictions.pkl`
  - Individual model files: `beat_classification_*_model.pkl`, `beat_classification_*_encoder.pkl`
  - Individual model files: `rhythm_classification_*_model.pkl`, `rhythm_classification_*_encoder.pkl`

**Output files**:
- `beat_classification_ml_predictions.pkl`
- `rhythm_classification_ml_predictions.pkl`
- `beat_classification_xgboost_model.pkl`, `beat_classification_xgboost_encoder.pkl`
- `beat_classification_randomforest_model.pkl`, `beat_classification_randomforest_encoder.pkl`
- `beat_classification_logisticregression_model.pkl`, `beat_classification_logisticregression_encoder.pkl`
- `rhythm_classification_xgboost_model.pkl`, `rhythm_classification_xgboost_encoder.pkl`
- `rhythm_classification_randomforest_model.pkl`, `rhythm_classification_randomforest_encoder.pkl`
- `rhythm_classification_logisticregression_model.pkl`, `rhythm_classification_logisticregression_encoder.pkl`

**Estimated time**: ~5-10 minutes (CPU-based)

**Important**: This notebook must run after beat_dl and rhythm_dl notebooks to ensure same data splits, but can run before beat_lstm and rhythm_lstm (they use same splits).

---

## 📊 Phase 2: Individual Visualizations

These notebooks visualize results from specific model types. **Run these after Phase 1 is complete**. They can be run in any order.

### Step 2.1: Visualize CNN Results
**Notebook**: `classification_visualization.ipynb`

**Purpose**: Visualize inputs and outputs of CNN models for both beat and rhythm classification

**What it does**:
- Loads predictions from `beat_classification_predictions.pkl` (CNN beat)
- Loads predictions from `rhythm_classification_predictions.pkl` (CNN rhythm)
- Visualizes:
  - Input signal sequences with label distribution
  - Confusion matrices
  - ROC curves
  - Precision-Recall curves
  - Prediction confidence distributions
  - Per-class accuracy

**Dependencies**: Requires `beat_dl.ipynb` and `rhythm_dl.ipynb` to be completed first

**Estimated time**: ~2-5 minutes

---

### Step 2.2: Visualize LSTM Results
**Notebook**: `lstm_visualization.ipynb`

**Purpose**: Visualize inputs and outputs of LSTM models for both beat and rhythm classification

**What it does**:
- Loads predictions from `beat_classification_lstm_predictions.pkl`
- Loads predictions from `rhythm_classification_lstm_predictions.pkl`
- Similar visualizations as classification_visualization.ipynb but for LSTM models

**Dependencies**: Requires `beat_lstm.ipynb` and `rhythm_lstm.ipynb` to be completed first

**Estimated time**: ~2-5 minutes

---

### Step 2.3: Visualize ML Model Results
**Notebook**: `ml_visualization.ipynb`

**Purpose**: Visualize inputs and outputs of traditional ML models for both tasks

**What it does**:
- Loads predictions from `beat_classification_ml_predictions.pkl`
- Loads predictions from `rhythm_classification_ml_predictions.pkl`
- Visualizes:
  - HRV feature distributions
  - Confusion matrices
  - ROC curves
  - Precision-Recall curves
  - Sample input signals with feature importance
  - Output probabilities

**Dependencies**: Requires `trad_ml.ipynb` to be completed first

**Estimated time**: ~3-7 minutes

---

## 🎯 Phase 3: Comprehensive Evaluation

This notebook compares all models together. **Run this last, after all training and individual visualization notebooks are complete**.

### Step 3.1: Comprehensive Model Comparison
**Notebook**: `general_evaluation.ipynb`

**Purpose**: Comprehensive evaluation and comparison of ALL models (CNN, LSTM, XGBoost, RandomForest, LogisticRegression) for both tasks

**What it does**:
- Loads predictions from ALL models (DL and ML)
- Calculates comprehensive metrics for all models
- Creates comparison tables for both tasks
- Generates visualizations:
  - Sample input-output comparisons (CNN vs LSTM)
  - ROC and PR curves for all models
  - Confusion matrices
  - ML model sample visualizations
  - Overall performance summary charts
- Saves summary table to: `experiments/results/metrics/overall_performance_comparison.csv`
- Saves all figures to: `experiments/results/plots/`

**Dependencies**: Requires ALL Phase 1 notebooks to be completed:
- `beat_dl.ipynb`
- `beat_lstm.ipynb`
- `rhythm_dl.ipynb`
- `rhythm_lstm.ipynb`
- `trad_ml.ipynb`

**Output files**:
- `experiments/results/metrics/overall_performance_comparison.csv`
- `experiments/results/metrics/overall_performance_comparison.xlsx` (if openpyxl available)
- Multiple PNG figures in `experiments/results/plots/`

**Estimated time**: ~5-10 minutes

---

## 📝 Execution Checklist

Use this checklist to track your progress:

### Phase 1: Model Training
- [ ] Step 1.1: Run `beat_dl.ipynb`
- [ ] Step 1.2: Run `beat_lstm.ipynb`
- [ ] Step 1.3: Run `rhythm_dl.ipynb`
- [ ] Step 1.4: Run `rhythm_lstm.ipynb`
- [ ] Step 1.5: Run `trad_ml.ipynb`

### Phase 2: Individual Visualizations (Optional)
- [ ] Step 2.1: Run `classification_visualization.ipynb`
- [ ] Step 2.2: Run `lstm_visualization.ipynb`
- [ ] Step 2.3: Run `ml_visualization.ipynb`

### Phase 3: Comprehensive Evaluation
- [ ] Step 3.1: Run `general_evaluation.ipynb`

---

## ⚠️ Important Notes

### Data Consistency
- All notebooks use the **same patient splits** (60/20/20, random_state=42)
- All notebooks use the **same window parameters** (window_size=60, stride=30)
- This ensures fair comparison between models

### Parallel Execution
- **Phase 1**: Notebooks 1.1-1.4 (DL models) can be run in parallel if you have multiple GPUs
- **Phase 1**: Notebook 1.5 (ML models) can run in parallel with DL notebooks but uses same data splits
- **Phase 2**: All visualization notebooks can run in parallel after Phase 1 completes

### File Locations
All output files are saved to:
- **Predictions**: `arrdb/experiments/results/predictions/`
- **Metrics**: `arrdb/experiments/results/metrics/`
- **Plots**: `arrdb/experiments/results/plots/`

### Troubleshooting
If a visualization notebook fails to load predictions:
1. Check that the corresponding training notebook has completed successfully
2. Verify prediction files exist in `experiments/results/predictions/`
3. Restart the kernel and reload the notebook

---

## 🎓 Quick Start (Minimum Required)

If you only want the comprehensive evaluation:

**Minimum execution order**:
1. Run `beat_dl.ipynb` (CNN beat)
2. Run `beat_lstm.ipynb` (LSTM beat)
3. Run `rhythm_dl.ipynb` (CNN rhythm)
4. Run `rhythm_lstm.ipynb` (LSTM rhythm)
5. Run `trad_ml.ipynb` (ML models)
6. Run `general_evaluation.ipynb` (comprehensive comparison)

The individual visualization notebooks (Phase 2) are optional but provide detailed analysis for each model type.

---

## 📚 Additional Information

### Model Architectures
- **CNN**: Lightweight 1D-CNN with convolutional and pooling layers
- **LSTM**: Lightweight LSTM with bidirectional layers
- **XGBoost**: Gradient boosting with regularization
- **RandomForest**: Ensemble of decision trees
- **LogisticRegression**: Linear classifier with L2 regularization

### Evaluation Metrics
All models are evaluated using:
- Accuracy
- Precision (Macro, Weighted)
- Recall (Macro, Weighted)
- F1-Score (Macro, Weighted)
- AUROC (Macro)
- AUPRC (Macro)

### Window-Level Evaluation
All models are evaluated at the **window level** (60-beat windows) for fair comparison:
- ML models: Extract HRV features from each window
- DL models: Use raw RR interval sequences from each window
- Same windows, same patient splits → Fair comparison

---

## ✅ Completion

Once you've completed all steps, you will have:
- ✅ Trained models for all approaches (DL and ML)
- ✅ Comprehensive performance comparison table
- ✅ Visualizations comparing all models
- ✅ All figures saved for publication/reporting

**Good luck with your experiments! 🚀**

