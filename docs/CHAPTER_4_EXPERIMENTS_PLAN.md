# Chapter 4.4–4.6 Experiments Execution Plan

**Date**: March 19, 2026  
**Status**: Planning & Implementation Phase  
**Baseline**: Project restored to pre-March committed state

---

## Executive Summary

This document maps the gap checklist from Chapter 4.4–4.6 to concrete, executable experiments ranked by **ROI (impact/time)**.

**Current Status**:
- ✅ 4.4.1, 4.4.2, 4.6.1–4.6.3 executed from existing artifacts
- ❌ 4.4.3, 4.4.4, 4.5.1–4.5.4, 4.6.4 require new training/analysis runs

**Critical Finding**: The gap is NOT in data or analysis; it's in **parametric sweeps and architectural ablations**. All support infrastructure exists.

---

## Tier 1: Must Have (Impact Crit, Time ~2 days)

### 4.4.3: True Ablation — Component Removal

**What we're testing**: Which component (TCN, RNN, Attention) contributes most to SynerT's PR-AUC improvement?

**Three variants**:

| Variant | Description | Config | Impact |
|---------|-------------|--------|--------|
| `SynerT_no_tcn` | Remove TCN stage, RNN processes raw input | `no_tcn_branch.yaml` | Isolate TCN contribution |
| `SynerT_no_rnn` | Remove RNN layers, fuse TCN output directly | `no_rnn_branch.yaml` | Isolate RNN contribution |
| `SynerT_no_attn` | Keep all branches, replace attention with mean pooling | `no_attention.yaml` | Isolate attention contribution |

**Execution**:
- Use `use_attention=False` parameter in config for no_attention variant
- For no_tcn/no_rnn: Need **source code modifications** (see Implementation section)
- Run standard 5×5 (5 folds × 5 random seeds) grid
- Compute paired Wilcoxon tests vs full SynerT on PR-AUC

**Expected Outcome**: Precise contribution percentages for each component. This is the **most impactful ablation**.

**Estimated Time**: 
- Config creation: 30 min
- Code mods: 30 min
- Training (5×5×3 models): 18–24 hours on GPU

---

### 4.4.4: Fusion Variant Comparison

**What we're testing**: How sensitive is SynerT's performance to fusion mechanism choice?

**Three variants**:

| Variant | Description | Config | Rationale |
|---------|-------------|--------|-----------|
| `SynerT_early_fusion` | Concat TCN + RNN features *before* attention | `early_fusion.yaml` | Raw feature fusion |
| `SynerT_late_fusion` | Predict from TCN, predict from RNN, concat logits | `late_fusion.yaml` | Soft prediction fusion |
| `SynerT_no_gate` | Current fusion but remove gating/weighting | `no_gate.yaml` | Ablate gating mechanism |

**Execution**:
- 5×5 grid for each variant
- Paired Wilcoxon vs full SynerT on PR-AUC
- Code mods needed for early/late fusion variants

**Expected Outcome**: Validates design choice of joint attention + gating (vs simpler alternatives).

**Estimated Time**: 
- Config creation: 30 min
- Code mods: 60 min
- Training (5×5×3 models): 18–24 hours on GPU

---

### 4.5.1: Observation Window Sensitivity

**What we're testing**: Is 60 min the optimal observation window? How much leeway do we have?

**Four window sizes**:

| Window (min) | t_cut_sec | Rationale |
|--------------|-----------|-----------|
| 10 | 600 | Minimal but practical |
| 20 | 1200 | Early alert potential |
| 30 | 1800 | Clinical consensus sweet spot |
| 60 | 3600 | Current setting (baseline) |

**Execution**:
- **NO code changes** — only config parameter change `t_cut_sec`
- Run SynerT + Dilated RNN (baseline comparator) at each window size
- 5×5 grid for each (window × model) pair = **8 configs × 5 folds × 5 seeds = 200 runs**
- Plot PR-AUC × window size curve with error bars

**Expected Outcome**: 
- If curve peaks at 60 min: validates our choice
- If curve peaks at 30 min: suggests earlier prediction risk acceptable
- Clinical relevance: shorter windows = faster interventions

**Estimated Time**: 
- Config creation: 30 min
- Training (8 configs × 5 folds × 5 seeds): 24–36 hours on GPU
- Post-processing: 30 min

---

## Tier 2: Should Have (High clinical value, Time ~2 days)

### 4.5.3: Missingness Stress Test

**What we're testing**: Model robustness to sensor failures (especially arterial line dropout).

**Two scenarios**:

| Scenario | Details | Method |
|----------|---------|--------|
| **Random channel dropout** | Mask random channels at inference time | p ∈ {0.1, 0.2, 0.3} (~10%, 20%, 30% channels zero-filled) |
| **ART_MBP dropout** | Simulate arterial line failure (most critical signal) | Apply to ART_MBP channel only |

**Execution**:
- **NO model retraining** — use existing OOF predictions
- Script applies mask at inference time, reruns forward pass
- Measure PR-AUC degradation vs clean input
- Repeat for 3–5 representative test cases

**Expected Outcome**: 
- Quantifies robustness claim
- Directly validates FN error analysis (many FNs linked to `art_mbp_zero_frac`)
- Important for clinical deployment confidence

**Estimated Time**: 
- Script creation: 60 min
- Execution: 2–3 hours

---

### 4.6.4: Temporal Attention Interpretation

**What we're testing**: Which time steps does the model focus on for TP vs FP vs FN cases?

**Execution**:
- Export per-timestep attention weights for SynerT
- Run inference on 4 representative cases:
  - TP (caseid=3438)
  - TN (caseid=844)
  - FP (top FP, caseid=?)
  - FN (top FN, caseid=?)
- Generate attention heatmaps: time × attention weight
- Plot alongside vital signs timeline

**Expected Outcome**: 
- Visual interpretation of what model "sees"
- Identify whether attention focuses on plausible clinical events
- TP vs FP: do they attend to different signals?

**Estimated Time**: 
- Hook implementation: 60 min
- Inference + plotting: 30 min

---

## Tier 3: Nice to Have (Diminishing returns, Time ~3+ days)

### 4.5.2: Hyperparameter Sensitivity Sweep

**Grid**: 3 × 3 × 3 = 27 configurations minimum

```
lr: [1e-4, 3e-4, 1e-3]
dropout: [0.1, 0.3, 0.5]
hidden_dim: [64, 128, 256]
```

**Execution**: 
- 5×5 folds/seeds per config
- **Total: 27 configs × 5 folds × 5 seeds = 675 training runs**

**Estimated Time**: 72+ hours (SKIP if deadline tight)

### 4.5.4: Clinical Subgroup Analysis

**Requires**: Rich metadata fields in `cohort_master.csv`
- Age (binary or quartile)
- Sex
- Surgery type
- Baseline Cr quartile

**Output**: Calibration, sensitivity, specificity per subgroup.

**Estimated Time**: 4–6 hours (if metadata available)

---

## Implementation Timeline

### **Week 1 (This week)**  — High ROI Quick Wins

| Task | Duration | Owner | Deliverable |
|------|----------|-------|-------------|
| 1. Create config files (12 configs) | 1 h | Agent | `.yaml` files ready |
| 2. Implement no_tcn/no_rnn variants | 1 h | Agent | Source code ready |
| 3. Run 4.4.3 ablations (75 models) | 18–24 h | User | `.csv` results |
| 4. Run 4.4.4 fusions (75 models) | 18–24 h | User | `.csv` results |
| 5. Run 4.5.1 windows (200 models) | 24–36 h | User | `.csv` results + plot |
| 6. Implement 4.5.3 script | 1 h | Agent | Ready to run |
| 7. Run 4.5.3 stress test | 2–3 h | User | `.csv` robustness report |
| 8. Implement 4.6.4 hook | 1 h | Agent | Ready to run |
| 9. Run 4.6.4 inference | 30 min | User | Attention heatmaps |
| **Total Training Time** | ~60–90 h | User | (Parallelizable) |

### **Week 2** (If needed) — Lower ROI

- 4.5.2: HPO grid (skip if deadline is this month)
- 4.5.4: Subgroup analysis (run if data available)

---

## File Inventory

### Config Files (to be created)

```
vitaldb_aki/configs/
├── no_tcn_branch.yaml          # 4.4.3 variant 1
├── no_rnn_branch.yaml          # 4.4.3 variant 2
├── no_attention.yaml           # 4.4.3 variant 3
├── early_fusion.yaml           # 4.4.4 variant 1
├── late_fusion.yaml            # 4.4.4 variant 2
├── no_gate.yaml                # 4.4.4 variant 3
├── window_600_sec.yaml         # 4.5.1 variant 1
├── window_1200_sec.yaml        # 4.5.1 variant 2
├── window_1800_sec.yaml        # 4.5.1 variant 3
├── window_3600_sec.yaml        # 4.5.1 variant 4 (baseline)
└── ... (2 more for Dilated RNN window variants)
```

### Scripts (to be created)

```
scripts/
├── run_ablation_4_4_3.py        # Batch runner for 4.4.3
├── run_fusion_4_4_4.py          # Batch runner for 4.4.4
├── run_window_sensitivity_4_5_1.py  # Batch runner for 4.5.1
├── test_missingness_stress_4_5_3.py  # 4.5.3 inference-time test
├── export_attention_4_6_4.py    # 4.6.4 attention hook
└── aggregate_results.py         # Combine all results into final report
```

### Source Code Modifications (if needed)

```
src/vitaldb_aki/models/architectures.py:
- Add variant flags for no_tcn / no_rnn / early_fusion / late_fusion
  (OR create separate model classes)
```

---

## Priority Decision Matrix

```
Impact     Time       Priority   Notes
─────────────────────────────────────────────────────────────
4.4.3      Crit       2–3 h code  Tier 1  MUST DO — core ablation
4.4.4      High       2–3 h code  Tier 1  MUST DO — validates fusion
4.5.1      High       1 h code    Tier 1  MUST DO — clinical credibility
4.5.3      High       1 h code    Tier 2  Important — validates robustness claim
4.6.4      Medium     1 h code    Tier 2  Interpretability bonus
4.5.2      Medium     48+ h code  Tier 3  SKIP if deadline < 2 weeks
4.5.4      Medium     2–3 h code  Tier 3  Do if metadata rich
```

---

## Next Steps

1. **Agent**: Create all config files (30 min)
2. **Agent**: Implement source code modifications for no_tcn/no_rnn/early/late fusion (90 min)
3. **Agent**: Create batch runner scripts (60 min)
4. **User**: Execute training batch for 4.4.3 (18–24 h)
5. **User**: Execute training batch for 4.4.4 (18–24 h)  
6. **User**: Execute training batch for 4.5.1 (24–36 h)
7. **Agent**: Post-process results, generate updated Chapter 4 report
8. **User**: Run 4.5.3 stress test (2–3 h)
9. **User**: Run 4.6.4 attention export (30 min)

---

## Expected Chapter 4 Impact

**Before experiments**: 
- 4.4: Implicit ablations only
- 4.5: No sensitivity analysis
- 4.6: Representative cases but no FP/FN patterns

**After Tier 1 completion**:
- 4.4: Explicit component ablations + fusion variants (strong Accept)
- 4.5: Window sensitivity curve (clinical credibility)
- 4.6: Robustness stress test + attention interpretation (high confidence)

**Outcome**: Chapter 4 shifts from "descriptive" → "analytically rigorous" → **conference-submittable quality**.

---

## Success Criteria

- [ ] All 12 configs created and validated
- [ ] Ablation models trained and parsed
- [ ] Window sensitivity curve shows monotonic or optimal trend
- [ ] Stress test shows <10% PR-AUC degradation under 30% channel dropout
- [ ] Attention weights show plausible temporal patterns
- [ ] Updated Chapter 4 report with all results
- [ ] Tables ready for manuscript submission

---

**Generated**: 2026-03-19  
**Last Updated**: 2026-03-19
