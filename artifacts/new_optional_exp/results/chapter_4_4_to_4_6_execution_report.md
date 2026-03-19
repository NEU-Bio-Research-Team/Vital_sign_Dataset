# Chapter 4.4-4.6 Execution Report

Generated from experiment: new_optional_exp

## 1) Restore and execution status
- Project state restored to committed baseline before March (working tree cleaned from uncommitted March edits).
- Checklist execution performed against existing artifacts in new_optional_exp.

## 2) Section 4.4 results from existing data

### 4.4.1 Contribution of TCN branch (implicit ablation)
- SynerT ROC-AUC: 0.6908 +/- 0.0528
- Standalone TCN ROC-AUC: 0.6751 +/- 0.0736
- Delta (SynerT - TCN): +0.0157
- SynerT PR-AUC: 0.1452 vs TCN PR-AUC: 0.1213

### 4.4.2 Contribution of Dilated RNN branch (implicit ablation)
- Standalone Dilated RNN ROC-AUC: 0.7005 +/- 0.0524
- SynerT ROC-AUC: 0.6908 +/- 0.0528
- Delta (SynerT - Dilated RNN): -0.0096
- SynerT vs Dilated RNN PR-AUC delta: +0.0154, p-value=0.0241 (one-sided Wilcoxon)

### 4.4.3 and 4.4.4
- Not executable from current artifacts (no explicit no-attention or fusion-variant runs).
- Required new runs are listed in section 5 below.

## 3) Section 4.5 status
- 4.5.1 Observation windows (10/20/30/60 min): missing
- 4.5.2 Hyperparameter sensitivity sweep: missing
- 4.5.3 Missingness stress tests: missing
- 4.5.4 Clinical subgroup analysis: missing

## 4) Section 4.6 executed outputs
- Operating threshold chosen from OOF by max-F1: 0.5703
- TP count: 35
- TN count: 2029
- FP count: 283
- FN count: 66
- Representative cases exported to chapter_4_6_representative_cases.csv
- Error-pattern summary exported to chapter_4_6_error_patterns.csv

### 4.6.1 TP/TN representative cases
- TP caseid=3438, prob=0.8457, baseline_cr=0.55
- TN caseid=844, prob=0.0320, baseline_cr=0.82

### 4.6.2 False-positive patterns
- FP median baseline_cr=0.85
- FP median obs_frac_ART_MBP=0.998
- FP median art_mbp_zero_frac=0.010

### 4.6.3 False-negative patterns
- FN median baseline_cr=0.76
- FN median obs_frac_ART_MBP=0.999
- FN median art_mbp_zero_frac=0.035

### 4.6.4 Temporal attention interpretation
- Not executable from current artifacts (attention weights not exported).

## 5) Required additional runs (main repo)
1. True ablations: SynerT_no_attention, SynerT_no_tcn_branch, SynerT_no_rnn_branch.
2. Fusion variants: early_fusion, late_fusion, no_attention_fusion.
3. Window sensitivity: t_cut_sec in {600, 1200, 1800, 3600} with same fold/seed protocol.
4. Hyperparameter sweep: lr, dropout, hidden_dim, class-imbalance loss params.
5. Missingness stress test: channel dropout and random mask escalation.
6. Subgroup metrics: age/sex/surgery/risk bins with calibration and CI.
7. Attention export: per-case per-time-step attention weights for TP/FP/FN case narratives.