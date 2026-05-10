# Chapter 4.4–4.6 Experiments: Execution Checklist

**Last Updated**: 2026-03-19  
**Status**: Ready to Execute

---

## Quick Reference: Total Time Budget

| Tier | Experiment | Configs | Runs | GPU Hours | Priority |
|------|-----------|---------|------|-----------|----------|
| **1** | 4.4.3 Ablation | 3 | 75 | 18–24 | MUST |
| **1** | 4.4.4 Fusion | 3 | 75 | 18–24 | MUST |
| **1** | 4.5.1 Window | 8 | 200 | 48–72 | MUST |
| **2** | 4.5.3 Stress | 1 | 1 | 2–3 min | IMPORTANT |
| **2** | 4.6.4 Attention | 1 | 4 | 10–15 min | NICE |
| **TOTAL** | | 16 | 355 | **84–138 hours** | |

**Parallelization Note**: All training runs are independent. Run multiple configs in parallel on multiple GPUs if available.

---

## BEFORE YOU START: Verify Config Files

All 16 config files have been created in `vitaldb_aki/configs/`:

```bash
# List configs to verify creation
ls -la vitaldb_aki/configs/*.yaml | grep -E "(no_tcn|no_rnn|no_attention|early_fusion|late_fusion|no_gate|window.*sec)"
```

**Expected output**: 16 files
- ✓ no_tcn_branch.yaml
- ✓ no_rnn_branch.yaml
- ✓ no_attention.yaml
- ✓ early_fusion.yaml
- ✓ late_fusion.yaml
- ✓ no_gate.yaml
- ✓ window_600_sec.yaml, window_1200_sec.yaml, window_1800_sec.yaml, window_3600_sec.yaml
- ✓ dilated_rnn_window_600_sec.yaml, dilated_rnn_window_1200_sec.yaml, dilated_rnn_window_1800_sec.yaml, dilated_rnn_window_3600_sec.yaml

---

## TIER 1: High Priority (This Week)

### 4.4.3: True Ablation — Component Removal

**Objective**: Isolate contribution of TCN, RNN, and Attention components.

**Expected outcome**: 3 CSV files with 5-fold × 5-seed results each.

---

#### Option A: Sequential Execution (if single GPU)

```powershell
# Config 1: No TCN branch
cd c:\Users\LENOVO\Documents\PYTHON\BRT\31-10-2025\new_testing\Vital_sign_Dataset
python scripts\train.py --config vitaldb_aki/configs/no_tcn_branch.yaml --seed 42 --fold 0
python scripts\train.py --config vitaldb_aki/configs/no_tcn_branch.yaml --seed 43 --fold 0
python scripts\train.py --config vitaldb_aki/configs/no_tcn_branch.yaml --seed 44 --fold 0
python scripts\train.py --config vitaldb_aki/configs/no_tcn_branch.yaml --seed 45 --fold 0
python scripts\train.py --config vitaldb_aki/configs/no_tcn_branch.yaml --seed 46 --fold 0
# ... repeat for folds 1-4

# Config 2: No RNN branch
python scripts\train.py --config vitaldb_aki/configs/no_rnn_branch.yaml --seed 42 --fold 0
# ... (25 runs total)

# Config 3: No Attention
python scripts\train.py --config vitaldb_aki/configs/no_attention.yaml --seed 42 --fold 0
# ... (25 runs total)
```

**Total runtime**: ~18–24 hours on single GPU  
**Output files**:
- `artifacts/new_optional_exp/results/no_tcn_branch_5fold_summary.csv`
- `artifacts/new_optional_exp/results/no_rnn_branch_5fold_summary.csv`
- `artifacts/new_optional_exp/results/no_attention_5fold_summary.csv`

---

#### Option B: Batch Execution (Recommended if multi-GPU available)

```powershell
# Terminal 1: GPU 0
python scripts\train.py --config vitaldb_aki/configs/no_tcn_branch.yaml --device cuda:0 --batch-run true

# Terminal 2: GPU 1
python scripts\train.py --config vitaldb_aki/configs/no_rnn_branch.yaml --device cuda:1 --batch-run true

# Terminal 3: GPU 2
python scripts\train.py --config vitaldb_aki/configs/no_attention.yaml --device cuda:2 --batch-run true
```

**Runtime**: ~18–24 hours (parallel on 3 GPUs)

---

#### ✅ Post-4.4.3 Validation

After all runs complete:

```python
# Run this Python snippet to aggregate results
import pandas as pd
from pathlib import Path

root = Path("artifacts/new_optional_exp/results")
configs = ["no_tcn_branch", "no_rnn_branch", "no_attention"]
baseline = pd.read_csv(root / "all_models_5fold_summary.csv").query("model == 'temporal_synergy'")

for cfg in configs:
    df = pd.read_csv(root / f"{cfg}_5fold_summary.csv")
    print(f"\n{cfg}:")
    print(f"  PR-AUC: {df['pr_auc_mean'].values[0]:.4f} ± {df['pr_auc_std'].values[0]:.4f}")
    print(f"  ROC-AUC: {df['roc_auc_mean'].values[0]:.4f} ± {df['roc_auc_std'].values[0]:.4f}")
    pr_delta = baseline['pr_auc_mean'].values[0] - df['pr_auc_mean'].values[0]
    print(f"  Δ PR vs SynerT: {pr_delta:+.4f}")
```

---

### 4.4.4: Fusion Variant Comparison

**Objective**: Test alternative fusion mechanisms (early, late, no-gate).

**Expected outcome**: 3 CSV files with 5-fold × 5-seed results each.

---

#### ⚠️ IMPORTANT: Source Code Modification Required

The configs reference `model_type` values that may NOT exist yet:
- `temporal_synergy_early_fusion`
- `temporal_synergy_late_fusion`
- `temporal_synergy_no_gate`

**Before running these configs, you need to either:**

**Option 1**: Add conditional logic to `src/vitaldb_aki/models/architectures.py` (Recommended)

```python
# In get_model() function, add:
elif model_name == "temporal_synergy_early_fusion":
    return TemporalSynergyClassifier(
        input_dim,
        tcn_levels=tcn_levels,
        rnn_layers=rnn_layers,
        fusion_mode="early",  # Modified forward pass
        **synergy_kwargs,
    )
```

**Option 2**: Create wrapper model classes for each variant

**Option 3**: Defer these configs until Week 2 if time is critical

---

#### Execution (assuming source mods complete)

```powershell
# Config 1: Early Fusion
python scripts\train.py --config vitaldb_aki/configs/early_fusion.yaml --device cuda:0 --batch-run true

# Config 2: Late Fusion
python scripts\train.py --config vitaldb_aki/configs/late_fusion.yaml --device cuda:1 --batch-run true

# Config 3: No Gate
python scripts\train.py --config vitaldb_aki/configs/no_gate.yaml --device cuda:2 --batch-run true
```

**Runtime**: ~18–24 hours (parallel)

**Output files**:
- `artifacts/new_optional_exp/results/early_fusion_5fold_summary.csv`
- `artifacts/new_optional_exp/results/late_fusion_5fold_summary.csv`
- `artifacts/new_optional_exp/results/no_gate_5fold_summary.csv`

---

### 4.5.1: Observation Window Sensitivity

**Objective**: Test PR-AUC across 4 observation window sizes (10, 20, 30, 60 min).

**Execution Strategy**: SIMPLEST of all — just vary `t_cut_sec` in config (no code mods needed!)

---

#### Batch Execution for SynerT

```powershell
# Terminal 1: 10 min window
python scripts\train.py --config vitaldb_aki/configs/window_600_sec.yaml --device cuda:0 --batch-run true

# Terminal 2: 20 min window
python scripts\train.py --config vitaldb_aki/configs/window_1200_sec.yaml --device cuda:1 --batch-run true

# Terminal 3: 30 min window
python scripts\train.py --config vitaldb_aki/configs/window_1800_sec.yaml --device cuda:2 --batch-run true

# Terminal 4: 60 min window (baseline)
python scripts\train.py --config vitaldb_aki/configs/window_3600_sec.yaml --device cuda:0 --batch-run true
# (reuse GPU 0 after first config completes)
```

**Runtime**: ~24–36 hours for SynerT (4 windows × 5 folds × 5 seeds = 100 runs)

---

#### Batch Execution for Dilated RNN (Baseline Comparison)

```powershell
# After SynerT finishes (or in parallel on separate GPUs):

# Terminal 1: 10 min window
python scripts\train.py --config vitaldb_aki/configs/dilated_rnn_window_600_sec.yaml --device cuda:0 --batch-run true

# Terminal 2: 20 min window
python scripts\train.py --config vitaldb_aki/configs/dilated_rnn_window_1200_sec.yaml --device cuda:1 --batch-run true

# Terminal 3: 30 min window
python scripts\train.py --config vitaldb_aki/configs/dilated_rnn_window_1800_sec.yaml --device cuda:2 --batch-run true

# Terminal 4: 60 min window (baseline)
python scripts\train.py --config vitaldb_aki/configs/dilated_rnn_window_3600_sec.yaml --device cuda:0 --batch-run true
```

**Runtime**: ~24–36 hours for Dilated RNN

---

#### ✅ Post-4.5.1 Analysis

After all runs complete:

```python
# Generate window sensitivity curve
import pandas as pd
import matplotlib.pyplot as plt

results = []
for t_cut in [600, 1200, 1800, 3600]:
    synerty_df = pd.read_csv(f"artifacts/new_optional_exp/results/window_{t_cut}_sec_5fold_summary.csv")
    dilated_df = pd.read_csv(f"artifacts/new_optional_exp/results/dilated_rnn_window_{t_cut}_sec_5fold_summary.csv")
    
    results.append({
        "window_min": t_cut / 60,
        "synerty_pr": synerty_df['pr_auc_mean'].values[0],
        "dilated_pr": dilated_df['pr_auc_mean'].values[0],
    })

results_df = pd.DataFrame(results)
print(results_df)

# Plot
plt.figure(figsize=(10, 6))
plt.errorbar(results_df['window_min'], results_df['synerty_pr'], label='SynerT', marker='o')
plt.errorbar(results_df['window_min'], results_df['dilated_pr'], label='Dilated RNN', marker='s')
plt.xlabel('Observation Window (minutes)')
plt.ylabel('PR-AUC')
plt.title('4.5.1: Window Sensitivity')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('artifacts/new_optional_exp/results/chapter_4_5_1_window_sensitivity.png', dpi=150)
plt.show()
```

**Output file**: See `chapter_4_5_1_window_sensitivity.csv` and plot

---

## TIER 2: Important (Week 1–2)

### 4.5.3: Missingness Stress Test

**Objective**: Test model robustness to channel dropout (sensor failures).

**No retraining required!** ✨ Inference-time only.

---

#### Execution

```powershell
# Simply run the prepared script
python scripts\test_missingness_stress_4_5_3.py
```

**Runtime**: 2–3 minutes (inference-only)

**Output files**:
- `artifacts/new_optional_exp/results/chapter_4_5_3_missingness_stress_test.json`
- `artifacts/new_optional_exp/results/chapter_4_5_3_missingness_report.md`

---

#### Expected Output Example

```
Temporal Synergy:
  - Random 10% dropout: PR-AUC 0.1392 (−3.9%)
  - Random 20% dropout: PR-AUC 0.1331 (−8.1%)
  - Random 30% dropout: PR-AUC 0.1297 (−10.7%)
  - ART_MBP failure: PR-AUC 0.1231 (−15.2%)

Dilated RNN:
  - Random 10% dropout: PR-AUC 0.1398 (−0.5%)
  - ...
```

---

### 4.6.4: Temporal Attention Weights Export

**Objective**: Visualize what SynerT "attends to" for representative TP/TN/FP/FN cases.

**Partially inference-time!** Requires loading trained models + running forward pass with attention hook.

---

#### Simple Execution (Template/Planning)

```powershell
# Step 1: Pre-generate export plan
python scripts\export_attention_4_6_4.py
```

**Output**: 
- `artifacts/new_optional_exp/results/chapter_4_6_4_attention_export_plan.json`
- `artifacts/new_optional_exp/results/chapter_4_6_4_attention_export.md`

---

#### Full Implementation (requires coding)

This requires loading the 25 trained SynerT checkpoints and running forward passes with an attention hook. The template script provides structure; you'll need to:

1. Load each seed-fold checkpoint from training directory
2. Load corresponding test data for representative caseids
3. Run inference with `TemporalSynergyWithAttentionHook`
4. Average/aggregate attention weights across seeds
5. Generate heatmap visualizations

**Estimated additional time**: 1–2 hours of implementation + 30 min inference

---

## TIER 3: Nice to Have (Defer if Deadline Tight)

### 4.5.2: Hyperparameter Sensitivity (Skip for now)

**Why skip?**: 27 configs × 5 folds × 5 seeds = 675 training runs = 72+ GPU hours. Diminishing returns for deadline.

**Revisit only if**:
- Reviewers explicitly ask "why these hyperparameters?"
- You have 72+ hours spare GPU time

---

### 4.5.4: Clinical Subgroup Analysis (Conditional)

**Prerequisite**: Verify `cohort_master.csv` has rich metadata fields:
- Age (birth date or quartiles)
- Sex
- Surgery type
- Baseline creatinine quartile

---

## AGGREGATION & FINAL REPORTING

After all training runs complete (Tier 1 + 2):

---

#### Step 1: Aggregate Results

```python
# Run from project root
python scripts/aggregate_chapter_4_results.py
```

(Script to be created)

---

#### Step 2: Generate Updated Chapter 4 Report

```python
# Combine all new results with existing 4.6 findings
python -c "
import pandas as pd
import json

# Load all results
ablation_no_tcn = pd.read_csv('artifacts/new_optional_exp/results/no_tcn_branch_5fold_summary.csv')
ablation_no_rnn = pd.read_csv('artifacts/new_optional_exp/results/no_rnn_branch_5fold_summary.csv')
ablation_no_attn = pd.read_csv('artifacts/new_optional_exp/results/no_attention_5fold_summary.csv')

fusion_early = pd.read_csv('artifacts/new_optional_exp/results/early_fusion_5fold_summary.csv')
fusion_late = pd.read_csv('artifacts/new_optional_exp/results/late_fusion_5fold_summary.csv')
fusion_no_gate = pd.read_csv('artifacts/new_optional_exp/results/no_gate_5fold_summary.csv')

window_results = {
    600: pd.read_csv('artifacts/new_optional_exp/results/window_600_sec_5fold_summary.csv'),
    1200: pd.read_csv('artifacts/new_optional_exp/results/window_1200_sec_5fold_summary.csv'),
    1800: pd.read_csv('artifacts/new_optional_exp/results/window_1800_sec_5fold_summary.csv'),
    3600: pd.read_csv('artifacts/new_optional_exp/results/window_3600_sec_5fold_summary.csv'),
}

stress_results = json.load(open('artifacts/new_optional_exp/results/chapter_4_5_3_missingness_stress_test.json'))

print('✅ All results loaded successfully')
print(f'Ablations: TCN delta = {ablation_no_tcn[\"pr_auc_mean\"] - 0.1452:.4f}')
print(f'Window sensitivity: 600s={window_results[600][\"pr_auc_mean\"].values[0]:.4f}, 3600s={window_results[3600][\"pr_auc_mean\"].values[0]:.4f}')
"
```

---

## ❌ KNOWN LIMITATIONS & WORKAROUNDS

| Issue | Workaround | Priority |
|-------|-----------|----------|
| 4.4.3–4.4.4 variants need source code mods | Create wrapper classes OR modify architectures.py | HIGH |
| 4.6.4 full implementation needs ~2h coding | Use template, focus on TP/TN first | MEDIUM |
| Attention export depends on checkpoint availability | Ensure `artifacts/new_optional_exp/checkpoints/` has all 25 SynerT models | HIGH |

---

## SUCCESS CHECKLIST

- [ ] All 16 config files exist in `vitaldb_aki/configs/`
- [ ] 4.4.3 ablations trained (75 models), 3 CSV results files verified
- [ ] 4.4.4 fusion variants trained (75 models), 3 CSV results files verified *(if time permits)*
- [ ] 4.5.1 window sensitivity trained (200 models), curve plot generated
- [ ] 4.5.3 stress test executed, robustness report generated
- [ ] 4.6.4 attention export plan created, representative cases identified
- [ ] All results aggregated into single Excel/CSV file
- [ ] Updated `docs/CHAPTER_4_EXPERIMENTS_PLAN.md` with actual results
- [ ] Chapter 4 report updated with all new findings
- [ ] All outputs backed up to `/artifacts/new_optional_exp/results/`

---

## Timeline Estimate

```
Today (March 19):
  - Run 4.4.3 ablations overnight (Terminal 1–3)

Tomorrow (March 20):
  - Monitor 4.4.3 progress
  - Start 4.4.4 fusion variants when 4.4.3 finishes (or in parallel on GPU 3–5 if available)

Next 2 days (March 20–21):
  - While 4.4.4 runs: Start 4.5.1 window configs
  - Run 4.5.3 stress test (quick, 2 min)

By end of March 21:
  - All Tier 1 & 2 experiments complete
  - Aggregate results
  - Update Chapter 4 report
  - **READY FOR PEER SUBMISSION**
```

---

## Troubleshooting

**Q**: "Train script not found"  
**A**: Ensure `scripts/train.py` exists. If using different runner (e.g., `train.ipynb`), adapt commands.

**Q**: "CUDA out of memory"  
**A**: Reduce `batch_size` in config from 32 → 16, or run configs sequentially.

**Q**: "Models not converging"  
**A**: Check `monitor: pr_auc` in config is correct. Try `learning_rate: 0.0003`.

**Q**: "Results don't match baseline"  
**A**: Verify `random_state: 42` and `n_splits: 5` exactly match original config.

---

## Questions?

For detailed explanations, see:
- [CHAPTER_4_EXPERIMENTS_PLAN.md](../docs/CHAPTER_4_EXPERIMENTS_PLAN.md) — High-level strategy
- [Config file examples](vitaldb_aki/configs/) — Parameter reference
- [Chapter 4.4–4.6 original report](artifacts/new_optional_exp/results/chapter_4_4_to_4_6_execution_report.md) — Baseline findings

---

**Generated**: 2026-03-19  
**Last Update**: 2026-03-19  
**Status**: ✅ Ready to Execute
