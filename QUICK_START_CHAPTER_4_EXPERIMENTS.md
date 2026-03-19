# 🚀 Chapter 4.4–4.6 Experiments: Quick Start Guide

**Status**: ✅ All preparation complete — **Ready to Execute**  
**Date**: 2026-03-19  
**Estimated Completion**: March 21–22, 2026

---

## 📋 What's Prepared (Summary)

| Component | Status | Details |
|-----------|--------|---------|
| **Planning Doc** | ✅ | `docs/CHAPTER_4_EXPERIMENTS_PLAN.md` — Full strategy |
| **Config Files** | ✅ | 16 YAML files created in `vitaldb_aki/configs/` |
| **Tier 1 Scripts** | ✅ | 4.4.3 (ablations) ready; 4.5.1 (windows) ready |
| **Tier 2 Scripts** | ✅ | 4.5.3 (stress test) ready; 4.6.4 (attention) template ready |
| **Execution Guide** | ✅ | `scripts/EXECUTION_CHECKLIST_4_4_4_6.md` — Commands & timeline |

---

## 🎯 Your Action Items (In Order)

### Phase 1: IMMEDIATE (Today — March 19)

**1. Verify configs exist:**
```powershell
# Check all 16 configs were created
Get-ChildItem vitaldb_aki/configs/*.yaml | Where-Object {$_.Name -match "(no_tcn|no_rnn|no_attention|early_fusion|late_fusion|no_gate|window)" } | Measure-Object
# Expected: 16 items
```

**2. Decide on 4.4.4 Implementation (Choose ONE):**

| Option | Effort | Pros | Cons |
|--------|--------|------|------|
| **A: Modify architectures.py** | 60–90 min | Clean, reusable | Requires code understanding |
| **B: Create wrapper classes** | 90–120 min | Safe, isolated | More boilerplate |
| **C: Defer to Week 2** | 0 min | Immediate progress | Missing important ablation |

**Recommendation**: Start with **3 non-blocking configs** today while you implement Option A:

```powershell
# Can run IMMEDIATELY (no 4.4.4 code needed):
# 1. 4.4.3 ablations (no TCN, no RNN, no attention)
# 2. 4.5.1 window sensitivity (10, 20, 30, 60 min)
# 3. 4.5.3 stress test (takes 2 min)
```

---

### Phase 2: TRAINING RUNS (March 19–21)

**Copy the exact commands from here:**

#### Terminal 1: 4.4.3 No-TCN Ablation
```powershell
cd c:\Users\LENOVO\Documents\PYTHON\BRT\31-10-2025\new_testing\Vital_sign_Dataset

# Run full 5×5 grid (5 folds, 5 random seeds)
for ($fold = 0; $fold -lt 5; $fold++) {
    for ($seed = 42; $seed -lt 47; $seed++) {
        python scripts\train.py `
            --config vitaldb_aki/configs/no_tcn_branch.yaml `
            --seed $seed `
            --fold $fold
    }
}
```

**Expected duration**: 6–8 hours for 25 runs

---

#### Terminal 2: 4.4.3 No-RNN Ablation (start after above finishes, or use 2nd GPU)
```powershell
# Same as above but with no_rnn_branch.yaml
for ($fold = 0; $fold -lt 5; $fold++) {
    for ($seed = 42; $seed -lt 47; $seed++) {
        python scripts\train.py `
            --config vitaldb_aki/configs/no_rnn_branch.yaml `
            --seed $seed `
            --fold $fold
    }
}
```

**Expected duration**: 6–8 hours

---

#### Terminal 3: 4.4.3 No-Attention Ablation
```powershell
# Same as above but with no_attention.yaml
for ($fold = 0; $fold -lt 5; $fold++) {
    for ($seed = 42; $seed -lt 47; $seed++) {
        python scripts\train.py `
            --config vitaldb_aki/configs/no_attention.yaml `
            --seed $seed `
            --fold $fold
    }
}
```

**Expected duration**: 6–8 hours

---

#### Terminal 4: 4.5.1 Window Sensitivity (SynerT)
```powershell
# Run once per window config
# SIMPLEST because no code mods needed!

for ($window in "600", "1200", "1800", "3600") {
    Write-Host "Starting window_${window}_sec..."
    for ($fold = 0; $fold -lt 5; $fold++) {
        for ($seed = 42; $seed -lt 47; $seed++) {
            python scripts\train.py `
                --config vitaldb_aki/configs/window_${window}_sec.yaml `
                --seed $seed `
                --fold $fold
        }
    }
}
```

**Expected duration**: 24–36 hours (100 total runs for 4 windows)

---

#### Terminal 5: 4.5.1 Window Sensitivity (Dilated RNN Baseline)
```powershell
# After SynerT finishes, or in parallel if GPU available

for ($window in "600", "1200", "1800", "3600") {
    Write-Host "Starting dilated_rnn_window_${window}_sec..."
    for ($fold = 0; $fold -lt 5; $fold++) {
        for ($seed = 42; $seed -lt 47; $seed++) {
            python scripts\train.py `
                --config vitaldb_aki/configs/dilated_rnn_window_${window}_sec.yaml `
                --seed $seed `
                --fold $fold
        }
    }
}
```

**Expected duration**: 24–36 hours

---

### Phase 3: QUICK TESTS (March 21, after training runs)

**4.5.3: Missingness Stress Test (2 minutes)**
```powershell
python scripts\test_missingness_stress_4_5_3.py
```

**Outputs**:
- `artifacts/new_optional_exp/results/chapter_4_5_3_missingness_stress_test.json`
- `artifacts/new_optional_exp/results/chapter_4_5_3_missingness_report.md`

---

**4.6.4: Attention Export Plan (1 minute)**
```powershell
python scripts\export_attention_4_6_4.py
```

**Outputs**:
- `artifacts/new_optional_exp/results/chapter_4_6_4_attention_export_plan.json`
- `artifacts/new_optional_exp/results/chapter_4_6_4_attention_export.md`

---

### Phase 4: AGGREGATION (March 21–22)

**Combine all results:**
```python
# Open interactive Python REPL
python -i

# Once Python starts:
import pandas as pd
from pathlib import Path

root = Path("artifacts/new_optional_exp/results")

# Load ablations
ablation_no_tcn = pd.read_csv(root / "no_tcn_branch_5fold_summary.csv")
ablation_no_rnn = pd.read_csv(root / "no_rnn_branch_5fold_summary.csv")
ablation_no_attn = pd.read_csv(root / "no_attention_5fold_summary.csv")

# Load window results
windows = {}
for t in [600, 1200, 1800, 3600]:
    windows[int(t/60)] = pd.read_csv(root / f"window_{t}_sec_5fold_summary.csv")

# Print comparison
print("=" * 60)
print("4.4.3 ABLATION RESULTS")
print("=" * 60)
print(f"Baseline SynerT PR-AUC: 0.1452")
print(f"No TCN PR-AUC: {ablation_no_tcn['pr_auc_mean'].values[0]:.4f}")
print(f"No RNN PR-AUC: {ablation_no_rnn['pr_auc_mean'].values[0]:.4f}")
print(f"No Attention PR-AUC: {ablation_no_attn['pr_auc_mean'].values[0]:.4f}")

print("\n" + "=" * 60)
print("4.5.1 WINDOW SENSITIVITY RESULTS")
print("=" * 60)
for minutes, df in windows.items():
    print(f"{minutes:2d} min: PR-AUC = {df['pr_auc_mean'].values[0]:.4f}")
```

---

## 📊 Expected Results

### If Everything Works:

**4.4.3 Ablations**:
```
Component Contribution to PR-AUC:
  TCN Removal    → PR-AUC drops from 0.1452 to ~0.130 (−10%)
  RNN Removal    → PR-AUC drops from 0.1452 to ~0.125 (−14%)
  Attention Rmv. → PR-AUC drops from 0.1452 to ~0.140 (−3%)
  
Conclusion: RNN > TCN >> Attention in importance
```

**4.5.1 Window Sensitivity**:
```
Observation Window vs PR-AUC:
  10 min → 0.128
  20 min → 0.135
  30 min → 0.140
  60 min → 0.145 ← optimal

Conclusion: Longer observation = better but 30 min gives 96% of benefit
```

**4.5.3 Stress Test**:
```
Robustness under channel dropout:
  Random 10% dropout → PR drops 3–4%
  Random 30% dropout → PR drops 10–12%
  ART_MBP failure    → PR drops 15–20%
  
Conclusion: Model ROBUST to mild sensor loss, fragile to ART_MBP failure
```

---

## ⚠️ Important Notes

### Before You Start:

1. **Verify GPU availability**: Run `nvidia-smi` to check
   ```powershell
   nvidia-smi
   ```
   (Expected: CUDA-capable GPU with 8+ GB VRAM)

2. **Check disk space**: Need ~50 GB for checkpoint storage
   ```powershell
   Get-Volume | Select-Object DriveLetter, SizeRemaining | Format-Table
   ```

3. **Backup existing results**:
   ```powershell
   Copy-Item artifacts/new_optional_exp artifacts/new_optional_exp_backup_mar19
   ```

---

### Parallelization Tips:

If you have **multiple GPUs** (e.g., 4 GPUs), you can run everything in parallel:

```powershell
# Terminal 1: 4.4.3 No-TCN (GPU 0)
python scripts\train.py --config vitaldb_aki/configs/no_tcn_branch.yaml --device cuda:0 --batch-run true

# Terminal 2: 4.4.3 No-RNN (GPU 1)
python scripts\train.py --config vitaldb_aki/configs/no_rnn_branch.yaml --device cuda:1 --batch-run true

# Terminal 3: 4.4.3 No-Attention (GPU 2)
python scripts\train.py --config vitaldb_aki/configs/no_attention.yaml --device cuda:2 --batch-run true

# Terminal 4: 4.5.1 Windows (GPU 3)
python scripts\train.py --config vitaldb_aki/configs/window_600_sec.yaml --device cuda:3 --batch-run true
```

**Result**: All Tier 1 experiments finish in 18–24 hours instead of 80+ hours!

---

## 🛑 If Something Goes Wrong

| Error | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'vitaldb_aki'` | Add to PYTHONPATH: `$env:PYTHONPATH += ";.."` |
| `CUDA out of memory` | Reduce `batch_size` from 32 → 16 in config |
| `Config file not found` | Check path is relative to project root |
| `Test data file not found` | 4.5.3 uses synthetic fallback data (auto-generated) |
| `Results don't match baseline` | Verify `random_state: 42` and `n_splits: 5` exact match |

---

## 📌 Success Criteria

After all runs complete, you should have:

- [ ] ✅ 3 ablation CSV files in `artifacts/new_optional_exp/results/`
- [ ] ✅ 4 window sensitivity CSV files per model
- [ ] ✅ Stress test JSON + Markdown
- [ ] ✅ Attention export plan JSON + Markdown
- [ ] ✅ All results properly parsed (no errors)
- [ ] ✅ Chapter 4 updated with new findings

**When complete** → Ready to submit manuscript! 🎉

---

## 📞 Questions?

Refer to detailed docs:
- **Strategy**: `docs/CHAPTER_4_EXPERIMENTS_PLAN.md`
- **Operations**: `scripts/EXECUTION_CHECKLIST_4_4_4_6.md`
- **Config Reference**: `vitaldb_aki/configs/*.yaml`

---

**Timeline**: March 19–22, 2026  
**Effort**: Execute scripts (minimal coding required)  
**Impact**: Chapter 4 from "Good" → "Exceptional" ✨

**Ready? Let's go!** 🚀
