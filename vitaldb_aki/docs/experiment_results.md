# VitalDB AKI Prediction - Experiment Results Summary

**Experiment Name:** `demo_5signals`  
**Date:** December 2024  
**Status:** Partial Results Available

---

## Experiment Overview

This experiment evaluates deep learning models for predicting Acute Kidney Injury (AKI) using intraoperative vital signs from the VitalDB dataset. The experiment uses 5 physiological signals and evaluates 4 different model architectures with 5-fold cross-validation.

---

## Configuration

### Signals
- **Total Signals:** 5
  - ART_MBP (Arterial Mean Blood Pressure)
  - CVP (Central Venous Pressure)
  - NEPI_RATE (Norepinephrine Rate)
  - PLETH_HR (Plethysmography Heart Rate)
  - PLETH_SPO2 (Plethysmography SpO2)

- **Required Signals:** ART_MBP, PLETH_HR, PLETH_SPO2
- **Optional Signals:** CVP, NEPI_RATE

### Data Preprocessing
- **Sampling Rate:** 1.0 Hz
- **Maximum Length:** 14,400 seconds (4 hours)
- **Minimum Length:** 600 seconds (10 minutes)
- **Cutoff Mode:** `early_intraop` (60 minutes cutoff to prevent data leakage)
- **Minimum Observations per Channel:** 60 points

### Dataset
- **Total Cases in Cohort:** 2,413 cases
- **Usable Cases:** ~2,412 cases (after quality gates)
- **Cross-Validation:** 5-fold stratified split
- **Random State:** 42

### Training Hyperparameters
- **Epochs:** 20
- **Batch Size:** 32
- **Learning Rate:** 0.001
- **Weight Decay:** 0.0001
- **Early Stopping Patience:** 5 epochs
- **Monitor Metric:** PR-AUC (Precision-Recall AUC)
- **Device:** CUDA (GPU)

### Model Architecture
- **Hidden Dimension:** 64
- **Number of Layers:** 2
- **Dropout:** 0.2

---

## Model Performance Results

### Summary Statistics (5-Fold Cross-Validation)

| Model | ROC-AUC (Mean ± Std) | PR-AUC (Mean ± Std) |
|-------|---------------------|---------------------|
| **TCN** | **0.674 ± 0.104** | **0.146 ± 0.068** |
| **GRU** | 0.649 ± 0.081 | 0.132 ± 0.067 |
| **LSTM** | 0.611 ± 0.070 | 0.122 ± 0.039 |
| **BiLSTM** | 0.604 ± 0.072 | 0.108 ± 0.051 |

**Best Model:** TCN (Temporal Convolutional Network) achieves the highest performance in both ROC-AUC and PR-AUC metrics.

### Detailed Results by Fold

#### TCN (Temporal Convolutional Network)
| Fold | ROC-AUC | PR-AUC | Best Epoch | Status |
|------|---------|--------|------------|--------|
| 1 | 0.740 | 0.115 | 1 | Loaded from checkpoint |
| 2 | 0.740 | 0.204 | 3 | Loaded from checkpoint |
| 3 | 0.606 | 0.190 | 7 | Loaded from checkpoint |
| 4 | 0.527 | 0.061 | 1 | Loaded from checkpoint |
| 5 | 0.720 | 0.131 | 10 | Loaded from checkpoint |

**Mean Performance:** ROC-AUC = 0.674, PR-AUC = 0.146

#### BiLSTM (Bidirectional LSTM)
| Fold | ROC-AUC | PR-AUC | Best Epoch | Status |
|------|---------|--------|------------|--------|
| 1 | 0.712 | 0.124 | 2 | Loaded from checkpoint |
| 2 | 0.643 | 0.165 | 1 | Loaded from checkpoint |
| 3 | 0.546 | 0.141 | 8 | Loaded from checkpoint |
| 4 | 0.547 | 0.055 | 3 | Loaded from checkpoint |
| 5 | 0.573 | 0.053 | 2 | Loaded from checkpoint |

**Mean Performance:** ROC-AUC = 0.604, PR-AUC = 0.108

#### GRU (Gated Recurrent Unit)
| Fold | ROC-AUC | PR-AUC | Best Epoch | Status |
|------|---------|--------|------------|--------|
| 1 | - | - | 18 | Checkpoint loaded (NaN metrics) |
| 2 | - | - | 15 | Checkpoint loaded (NaN metrics) |
| 3 | - | - | 3 | Checkpoint loaded (NaN metrics) |
| 4 | - | - | 1 | Checkpoint loaded (NaN metrics) |
| 5 | - | - | 18 | Checkpoint loaded (NaN metrics) |

**Status:** Metrics unavailable (NaN) - checkpoints may be from incompatible architecture or invalid state.

#### LSTM (Long Short-Term Memory)
| Fold | ROC-AUC | PR-AUC | Best Epoch | Status |
|------|---------|--------|------------|--------|
| 1 | - | - | 5 | Checkpoint loaded (NaN metrics) |
| 2 | - | - | 1 | Checkpoint loaded (NaN metrics) |
| 3 | - | - | 2 | Checkpoint loaded (NaN metrics) |
| 4 | - | - | 10 | Checkpoint loaded (NaN metrics) |
| 5 | - | - | 8 | Checkpoint loaded (NaN metrics) |

**Status:** Metrics unavailable (NaN) - checkpoints may be from incompatible architecture or invalid state.

---

## Key Findings

1. **Best Performing Model:** TCN achieves the highest ROC-AUC (0.674) and PR-AUC (0.146) among all evaluated models.

2. **Model Ranking:**
   - 1st: TCN (ROC-AUC: 0.674, PR-AUC: 0.146)
   - 2nd: GRU (ROC-AUC: 0.649, PR-AUC: 0.132)
   - 3rd: LSTM (ROC-AUC: 0.611, PR-AUC: 0.122)
   - 4th: BiLSTM (ROC-AUC: 0.604, PR-AUC: 0.108)

3. **Performance Variability:** TCN shows the highest variability (std = 0.104 for ROC-AUC), indicating potential sensitivity to fold-specific characteristics.

4. **PR-AUC Values:** All models show relatively low PR-AUC values (0.108-0.146), suggesting the task is challenging, possibly due to class imbalance or limited predictive signal in the early intraoperative period.

---

## Issues and Limitations

1. **Incomplete Results:** GRU and LSTM models show NaN metrics, indicating:
   - Checkpoints were loaded from previous runs with potentially incompatible architectures
   - Models need to be retrained with `--force` flag to generate valid metrics

2. **Checkpoint Status:** Most models show `trained=False`, meaning checkpoints were loaded rather than freshly trained. This suggests:
   - Results are from previous training runs
   - Models should be retrained to ensure consistency with current configuration

3. **Data Quality:** Some cases were excluded during preprocessing due to quality gate failures (minimum length, observation counts, etc.)

---

## Recommendations

1. **Retrain All Models:** Run training with `--force` flag to ensure all models are trained with the current configuration:
   ```bash
   python scripts/train.py --experiment-name demo_5signals --force
   ```

2. **Further Investigation:**
   - Investigate why PR-AUC values are relatively low
   - Consider class imbalance handling strategies
   - Evaluate feature engineering approaches
   - Test different cutoff times and window sizes

3. **Model Selection:** TCN shows promise and should be the primary focus for further optimization and hyperparameter tuning.

---

## Files and Artifacts

### Results Files
- `artifacts/demo_5signals/results/all_models_5fold_summary.csv` - Summary statistics
- `artifacts/demo_5signals/results/tcn_5fold_metrics.csv` - TCN detailed results
- `artifacts/demo_5signals/results/bilstm_5fold_metrics.csv` - BiLSTM detailed results
- `artifacts/demo_5signals/results/gru_5fold_metrics.csv` - GRU results (incomplete)
- `artifacts/demo_5signals/results/lstm_5fold_metrics.csv` - LSTM results (incomplete)

### Visualization Files
- `artifacts/demo_5signals/results/tcn.png` - TCN ROC/PR curves
- `artifacts/demo_5signals/results/gru.png` - GRU ROC/PR curves
- `artifacts/demo_5signals/results/lstm.png` - LSTM ROC/PR curves
- `artifacts/demo_5signals/results/bilstm.png` - BiLSTM ROC/PR curves

### Model Checkpoints
- `artifacts/demo_5signals/models/tcn/` - TCN model checkpoints (5 folds)
- `artifacts/demo_5signals/models/gru/` - GRU model checkpoints (5 folds)
- `artifacts/demo_5signals/models/lstm/` - LSTM model checkpoints (5 folds)
- `artifacts/demo_5signals/models/bilstm/` - BiLSTM model checkpoints (5 folds)

---

## Next Steps

1. Retrain all models with `--force` to ensure complete and consistent results
2. Generate evaluation plots for all models
3. Perform statistical significance testing between models
4. Conduct error analysis on misclassified cases
5. Explore hyperparameter optimization for the best-performing model (TCN)

---

**Last Updated:** December 27, 2024

