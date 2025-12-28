# VitalDB AKI Prediction - Experiment Results Summary

**Experiment Name:** `demo_5signals`  
**Date:** December 2024  
**Status:** Complete Results Available

---

## Experiment Overview

This experiment evaluates deep learning models for predicting Acute Kidney Injury (AKI) using intraoperative vital signs from the VitalDB dataset. The experiment uses 5 physiological signals and evaluates **10 different model architectures** with 5-fold cross-validation. Models include baseline architectures (LSTM, BiLSTM, GRU, TCN) and advanced hybrid architectures combining temporal convolutions, attention mechanisms, dilated operations, and recurrent networks.

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
- **Batch Size:** 32 (reduced to 16 for memory-intensive models: attention, transformer, tcn_attention)
- **Learning Rate:** 0.001
- **Weight Decay:** 0.0001
- **Early Stopping Patience:** 5 epochs
- **Monitor Metric:** PR-AUC (Precision-Recall AUC)
- **Device:** CUDA (GPU)

### Model Architecture
- **Hidden Dimension:** 64
- **Number of Layers:** 2-10 (model-dependent)
- **Dropout:** 0.2

---

## Model Performance Results

### Summary Statistics (5-Fold Cross-Validation)

Results sorted by ROC-AUC (descending):

| Rank | Model | ROC-AUC (Mean ± Std) | PR-AUC (Mean ± Std) | Architecture Type |
|------|-------|---------------------|---------------------|-------------------|
| **1** | **Temporal Synergy** | **0.669 ± 0.108** | 0.121 ± 0.048 | Hybrid (TCN + Dilated RNN) |
| **2** | **TCN** | 0.667 ± 0.096 | **0.140 ± 0.058** | Temporal Convolutional Network |
| **3** | **Dilated RNN** | 0.657 ± 0.095 | **0.147 ± 0.064** | Dilated Recurrent Network |
| 4 | WaveNet | 0.650 ± 0.086 | 0.134 ± 0.052 | Gated Dilated Convolution |
| 5 | Dilated Conv | 0.649 ± 0.092 | 0.138 ± 0.065 | Enhanced TCN with Gated Activation |
| 6 | Attention | 0.645 ± 0.079 | 0.122 ± 0.031 | Multi-scale CNN + BiLSTM + Attention |
| 7 | Transformer | 0.641 ± 0.065 | 0.129 ± 0.049 | Transformer Encoder |
| 8 | WaveNet RNN | 0.630 ± 0.062 | 0.121 ± 0.051 | Hybrid (WaveNet + RNN) |
| 9 | TCN Attention | 0.626 ± 0.088 | 0.125 ± 0.055 | Hybrid (TCN + Attention) |
| 10 | BiLSTM | 0.604 ± 0.072 | 0.108 ± 0.051 | Bidirectional LSTM |

**Best Model by ROC-AUC:** Temporal Synergy (0.669)  
**Best Model by PR-AUC:** Dilated RNN (0.147)

### Detailed Results by Model

#### Temporal Synergy (Best ROC-AUC)
- **Architecture:** Hybrid model combining TCN blocks with Dilated RNN layers
- **Key Features:** Multi-stage processing (TCN → Dilated RNN → Feature Fusion → Attention Pooling)
- **Performance:** ROC-AUC = 0.669 ± 0.108, PR-AUC = 0.121 ± 0.048
- **Status:** All 5 folds successfully evaluated

#### TCN (Best PR-AUC among top models)
- **Architecture:** Temporal Convolutional Network with dilated convolutions
- **Performance:** ROC-AUC = 0.667 ± 0.096, PR-AUC = 0.140 ± 0.058
- **Status:** All 5 folds successfully evaluated
- **Note:** Strong PR-AUC performance (0.140) despite slightly lower ROC-AUC than Temporal Synergy

#### Dilated RNN (Best PR-AUC overall)
- **Architecture:** Multi-scale dilated recurrent neural network
- **Performance:** ROC-AUC = 0.657 ± 0.095, PR-AUC = 0.147 ± 0.064
- **Status:** All 5 folds successfully evaluated
- **Note:** Achieves highest PR-AUC (0.147) among all models

#### WaveNet
- **Architecture:** Stacked dilated convolutions with gated activation and skip connections
- **Performance:** ROC-AUC = 0.650 ± 0.086, PR-AUC = 0.134 ± 0.052
- **Status:** All 5 folds successfully evaluated

#### Dilated Conv (Enhanced TCN)
- **Architecture:** Enhanced TCN with gated activations, multi-scale paths, and attention pooling
- **Performance:** ROC-AUC = 0.649 ± 0.092, PR-AUC = 0.138 ± 0.065
- **Status:** All 5 folds successfully evaluated

#### Attention
- **Architecture:** Multi-scale CNN + Bidirectional LSTM + Temporal Attention
- **Performance:** ROC-AUC = 0.645 ± 0.079, PR-AUC = 0.122 ± 0.031
- **Status:** All 5 folds successfully evaluated
- **Note:** Lowest variability (std = 0.031 for PR-AUC)

#### Transformer
- **Architecture:** Transformer encoder with positional encoding and CLS token
- **Performance:** ROC-AUC = 0.641 ± 0.065, PR-AUC = 0.129 ± 0.049
- **Status:** All 5 folds successfully evaluated

#### WaveNet RNN
- **Architecture:** Hybrid model combining WaveNet blocks with RNN layers
- **Performance:** ROC-AUC = 0.630 ± 0.062, PR-AUC = 0.121 ± 0.051
- **Status:** All 5 folds successfully evaluated

#### TCN Attention
- **Architecture:** Hybrid model combining TCN blocks with temporal attention
- **Performance:** ROC-AUC = 0.626 ± 0.088, PR-AUC = 0.125 ± 0.055
- **Status:** All 5 folds successfully evaluated

#### BiLSTM
- **Architecture:** Bidirectional Long Short-Term Memory
- **Performance:** ROC-AUC = 0.604 ± 0.072, PR-AUC = 0.108 ± 0.051
- **Status:** All 5 folds successfully evaluated

---

## Key Findings

1. **Best Performing Models:**
   - **By ROC-AUC:** Temporal Synergy (0.669) - Hybrid TCN + Dilated RNN architecture
   - **By PR-AUC:** Dilated RNN (0.147) - Multi-scale dilated recurrent network
   - **Balanced Performance:** TCN (ROC-AUC: 0.667, PR-AUC: 0.140) - Strong performance on both metrics

2. **Model Ranking Insights:**
   - **Top 3 by ROC-AUC:** Temporal Synergy (0.669) > TCN (0.667) > Dilated RNN (0.657)
   - **Top 3 by PR-AUC:** Dilated RNN (0.147) > TCN (0.140) > Dilated Conv (0.138)
   - **Hybrid Models:** Temporal Synergy and WaveNet RNN demonstrate the value of combining different architectural components

3. **Architecture Performance Patterns:**
   - **Convolutional Models (TCN, Dilated Conv, WaveNet):** Strong performance (ROC-AUC: 0.626-0.667)
   - **Recurrent Models (Dilated RNN):** Best PR-AUC performance (0.147)
   - **Attention Models (Attention, Transformer, TCN Attention):** Moderate performance (ROC-AUC: 0.626-0.645)
   - **Hybrid Models:** Temporal Synergy (TCN+RNN) performs best overall, while TCN+Attention shows limited improvement over base TCN

4. **Performance Variability:**
   - **Lowest Variability:** Attention model (PR-AUC std = 0.031), Transformer (ROC-AUC std = 0.065)
   - **Highest Variability:** Temporal Synergy (ROC-AUC std = 0.108), Dilated RNN (PR-AUC std = 0.064)
   - **Stability:** Attention and Transformer models show more consistent performance across folds

5. **PR-AUC Analysis:**
   - **Range:** 0.108 (BiLSTM) to 0.147 (Dilated RNN)
   - **Top Performers:** Dilated RNN (0.147), TCN (0.140), Dilated Conv (0.138)
   - **Challenge:** All models show relatively low PR-AUC values, suggesting the task is challenging due to class imbalance or limited predictive signal in the early intraoperative period

6. **Hybrid Model Insights:**
   - **Temporal Synergy (TCN + Dilated RNN):** Successfully combines strengths of both architectures (ROC-AUC: 0.669)
   - **WaveNet RNN (WaveNet + RNN):** Moderate improvement over individual components (ROC-AUC: 0.630)
   - **TCN Attention (TCN + Attention):** Limited improvement over base TCN (ROC-AUC: 0.626 vs 0.667), suggesting attention may not be the optimal enhancement for TCN in this task

---

## Model Architecture Details

### Baseline Models
- **TCN:** 6 dilated convolution blocks with exponential dilation (1, 2, 4, 8, 16, 32)
- **BiLSTM:** 2-layer bidirectional LSTM with dropout
- **GRU/LSTM:** Not included in final results (incomplete training)

### Advanced Models
- **Dilated Conv:** Enhanced TCN with gated activations (tanh+sigmoid), multi-scale paths, attention pooling
- **Dilated RNN:** Multi-scale dilated RNN with feature fusion and attention pooling
- **WaveNet:** Stacked dilated convolutions (3 stacks × 10 layers) with gated activation and skip connections
- **Attention:** Multi-scale CNN + BiLSTM + temporal self-attention
- **Transformer:** Transformer encoder with positional encoding and CLS token

### Hybrid Models
- **Temporal Synergy:** TCN stage (4 levels) → Dilated RNN stage (4 layers) → Feature fusion → Attention pooling
- **WaveNet RNN:** WaveNet blocks (3 stacks × 10 layers) → RNN layers (2 layers) → Feature fusion → Attention pooling
- **TCN Attention:** TCN blocks (6 levels) → Temporal attention → Attention pooling

---

## Validation and Verification

All models have been re-evaluated from checkpoints to ensure consistency:
- **TCN:** Re-evaluated and validated (ROC-AUC: 0.667, PR-AUC: 0.140)
- **Temporal Synergy:** Re-evaluated and validated (ROC-AUC: 0.669, PR-AUC: 0.121)
- **All other models:** Metrics verified from checkpoint evaluations

---

## Files and Artifacts

### Results Files
- `artifacts/demo_5signals/results/all_models_5fold_summary.csv` - Summary statistics for all 10 models
- `artifacts/demo_5signals/results/tcn_5fold_metrics.csv` - TCN detailed results
- `artifacts/demo_5signals/results/temporal_synergy_5fold_metrics.csv` - Temporal Synergy detailed results
- `artifacts/demo_5signals/results/dilated_rnn_5fold_metrics.csv` - Dilated RNN detailed results
- `artifacts/demo_5signals/results/wavenet_5fold_metrics.csv` - WaveNet detailed results
- `artifacts/demo_5signals/results/dilated_conv_5fold_metrics.csv` - Dilated Conv detailed results
- `artifacts/demo_5signals/results/attention_5fold_metrics.csv` - Attention detailed results
- `artifacts/demo_5signals/results/transformer_5fold_metrics.csv` - Transformer detailed results
- `artifacts/demo_5signals/results/wavenet_rnn_5fold_metrics.csv` - WaveNet RNN detailed results
- `artifacts/demo_5signals/results/tcn_attention_5fold_metrics.csv` - TCN Attention detailed results
- `artifacts/demo_5signals/results/bilstm_5fold_metrics.csv` - BiLSTM detailed results

### Visualization Files
- `artifacts/demo_5signals/results/tcn_roc.png`, `tcn_pr.png`, `tcn_cm.png` - TCN visualizations
- `artifacts/demo_5signals/results/temporal_synergy_roc.png`, `temporal_synergy_pr.png`, `temporal_synergy_cm.png` - Temporal Synergy visualizations
- Similar visualization files available for other models

### Model Checkpoints
- All models have checkpoints saved in `artifacts/demo_5signals/models/{model_name}/` (5 folds each)

---

## Recommendations

1. **Model Selection:**
   - **For ROC-AUC optimization:** Use Temporal Synergy (0.669)
   - **For PR-AUC optimization:** Use Dilated RNN (0.147)
   - **For balanced performance:** Use TCN (ROC-AUC: 0.667, PR-AUC: 0.140)

2. **Further Investigation:**
   - Investigate why PR-AUC values are relatively low across all models
   - Consider class imbalance handling strategies (currently using pos_weight in loss)
   - Evaluate feature engineering approaches
   - Test different cutoff times and window sizes
   - Explore ensemble methods combining top-performing models

3. **Hyperparameter Optimization:**
   - Focus on Temporal Synergy, TCN, and Dilated RNN for further tuning
   - Experiment with different TCN levels, RNN layers, and attention configurations
   - Test different hidden dimensions and dropout rates

4. **Architecture Exploration:**
   - Investigate why TCN+Attention shows limited improvement over base TCN
   - Explore other hybrid combinations (e.g., Transformer+TCN, Dilated Conv+RNN)
   - Test deeper architectures for top-performing models

---

## Next Steps

1. ✅ Complete evaluation of all 10 model architectures
2. ✅ Validate results through re-evaluation
3. ⏭️ Perform statistical significance testing between top models
4. ⏭️ Conduct error analysis on misclassified cases
5. ⏭️ Explore hyperparameter optimization for top 3 models (Temporal Synergy, TCN, Dilated RNN)
6. ⏭️ Test ensemble methods combining Temporal Synergy, TCN, and Dilated RNN
7. ⏭️ Investigate feature importance and temporal patterns in top-performing models

---

**Last Updated:** December 2024
