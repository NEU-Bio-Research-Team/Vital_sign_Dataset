# VitalDB AKI Prediction – Experiment Results Summary (v2)

**Source files (do not edit):**
- `docs/experiment_results.md` (older summary)

**This file:** Updated summary based on the current project state (post-clone changes) and artifacts in `artifacts/5_signal_exp/`.

**Experiment Name:** `5_signal_exp`  
**Last Updated:** Jan 24, 2026  
**Status:** Multi-seed 5-fold results available

---

## 1) Experiment Overview

This experiment evaluates deep learning models for predicting Acute Kidney Injury (AKI) using intraoperative vital signs from the VitalDB dataset.

Key differences vs the older `demo_5signals` write-up:
- Results here are for `artifacts/5_signal_exp/`.
- Training/evaluation supports **multi-seed** runs and an **OOF seed-ensemble** (`ensemble_mean`).
- Model set differs slightly (e.g. LSTM/GRU are fully evaluated here; Attention/TCN-Attention are not present in this artifact).

---

## 2) Configuration (from `artifacts/5_signal_exp/config.json`)

### Signals
- **Total signals:** 5
  - ART_MBP (Arterial Mean Blood Pressure)
  - CVP (Central Venous Pressure)
  - NEPI_RATE (Norepinephrine Rate)
  - PLETH_HR (Plethysmography Heart Rate)
  - PLETH_SPO2 (Plethysmography SpO2)
- **Required signals:** ART_MBP, PLETH_HR, PLETH_SPO2
- **Optional signals:** CVP, NEPI_RATE
- **include_optional_signals:** `true`

### Data preprocessing
- **Sampling rate:** 1.0 Hz
- **Max length:** 14,400 sec (4h)
- **Min length:** 600 sec (10m)
- **Cutoff mode:** `early_intraop`
- **Cutoff time:** `t_cut_sec = 3600` (60m)
- **Min observations per channel:** 60 points
- **Inputs to the model:** signals + masks → feature dimension = `2 × (#signals)`

### Folds
- **Cross-validation:** 5-fold stratified
- **Random state:** 42
- **Stratification extras (implemented, config-controlled):**
  - `fold_stratify_use_cr_margin_bin = true` (enabled)
  - `fold_stratify_use_n_postop_labs = false` (disabled)

### Training hyperparameters
- **Epochs:** 40
- **Batch size:** 32
- **LR:** 0.001
- **Weight decay:** 0.0001
- **Early stopping patience:** 15
- **Monitor metric:** PR-AUC
- **Device:** CUDA (if available)

---

## 3) Post-clone pipeline changes that affect results

This summary is aligned with the current codebase:

1) **Multi-seed training + seed ensemble OOF**
- CLI supports `--seeds` and generates:
  - per-seed fold metrics and OOF predictions (`*_seed{n}.csv`)
  - **ensemble OOF** by averaging probabilities across seeds (`*_ensemble_mean.csv`)

2) **Preprocessing robustness / reproducibility**
- Cached NPZ handling is stricter (detect corrupted caches and rebuild).
- Fold stratification can optionally incorporate **AKI × CR-margin bin** and/or **AKI × n_postop_labs bins**.

3) **Per-fold scalers + robust scaling for NEPI_RATE**
- Per-fold scalers are fit from training folds.
- NEPI_RATE uses robust stats (median/IQR) rather than mean/std.

4) **Artifact path resolution**
- Artifact paths are resolved relative to project root for consistent outputs.

---

## 4) Model Performance Results

### 4.1 Updated report (from CSV summary)
These are taken from `artifacts/new_optional_exp/results/all_models_5fold_summary.csv`.

| Model | ROC-AUC (Mean ± Std) | PR-AUC (Mean ± Std) |
| --- | --- | --- |
| bilstm | 0.591 ± 0.068 | 0.109 ± 0.036 |
| lstm | 0.593 ± 0.066 | 0.105 ± 0.032 |
| mlp | 0.639 ± 0.075 | 0.111 ± 0.066 |
| wavenet | 0.670 ± 0.075 | 0.122 ± 0.066 |
| tcn | 0.675 ± 0.074 | 0.121 ± 0.064 |
| gru | 0.680 ± 0.067 | 0.116 ± 0.053 |
| dilated_conv | 0.692 ± 0.055 | 0.117 ± 0.058 |
| dilated_rnn | 0.700 ± 0.052 | 0.124 ± 0.055 |
| temporal_synergy | 0.691 ± 0.053 | 0.145 ± 0.082 |

**Best ROC-AUC:** Dilated RNN (0.700)  
**Best PR-AUC:** Temporal Synergy (0.145)

### 4.2 Seed-average report (for reference)
Not available in the provided CSV (only `all_models_5fold_summary.csv` was provided).

---

## 5) Files and Artifacts

- **Experiment config:** `artifacts/5_signal_exp/config.json`
- **Summary tables:**
  - `artifacts/new_optional_exp/results/all_models_5fold_summary.csv` (updated in this document)
  - `artifacts/5_signal_exp/results/all_models_5fold_summary.csv` (older reference; may differ)
- **Per-model metrics / OOF predictions:** See the corresponding experiment's `artifacts/<experiment>/results/` directory.

---

## 6) Recommendations

- Use **Temporal Synergy** if PR-AUC is the priority; use **Dilated Conv** if ROC-AUC is the priority.
- Keep reporting both:
  - **ensemble_mean** (primary, stable)
  - per-seed summaries (debug/variance analysis)
- If cross-fold instability persists, consider enabling lab-density stratification (`fold_stratify_use_n_postop_labs`) in future runs.
