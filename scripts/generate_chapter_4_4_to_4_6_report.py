#!/usr/bin/env python3
"""Generate chapter 4.4-4.6 deliverables from existing new_optional_exp artifacts.

Outputs:
- artifacts/new_optional_exp/results/chapter_4_4_to_4_6_execution_report.md
- artifacts/new_optional_exp/results/chapter_4_6_representative_cases.csv
- artifacts/new_optional_exp/results/chapter_4_6_error_patterns.csv
"""

from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_curve


ROOT = Path(__file__).resolve().parents[1]
EXP = "new_optional_exp"
ART = ROOT / "artifacts" / EXP
RES = ART / "results"


def pick_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Pick a stable operating threshold from validation OOF predictions.

    We choose the threshold maximizing F1 to avoid using an arbitrary 0.5 cutoff
    in an imbalanced setting.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    # precision_recall_curve has one extra point vs thresholds
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    idx = int(np.argmax(f1))
    return float(thresholds[idx]) if len(thresholds) else 0.5


def summarize_error_group(df: pd.DataFrame, label: str) -> dict:
    if df.empty:
        return {
            "group": label,
            "count": 0,
            "baseline_cr_median": np.nan,
            "valid_len_median": np.nan,
            "obs_frac_art_mbp_median": np.nan,
            "art_mbp_zero_frac_median": np.nan,
            "n_preop_labs_median": np.nan,
            "n_postop_labs_median": np.nan,
        }
    return {
        "group": label,
        "count": int(len(df)),
        "baseline_cr_median": float(df["baseline_cr"].median()),
        "valid_len_median": float(df["valid_len"].median()),
        "obs_frac_art_mbp_median": float(df["obs_frac_ART_MBP"].median()),
        "art_mbp_zero_frac_median": float(df["art_mbp_zero_frac"].median()),
        "n_preop_labs_median": float(df["n_preop_labs"].median()),
        "n_postop_labs_median": float(df["n_postop_labs"].median()),
    }


def main() -> None:
    summary = pd.read_csv(RES / "all_models_5fold_summary.csv")
    sig = pd.read_csv(RES / "all_models_5fold_significance_seedfold_vs_dilated_rnn.csv")
    oof = pd.read_csv(RES / "temporal_synergy_oof_predictions_ensemble_mean.csv")
    cohort = pd.read_csv(ART / "cohort_master.csv")

    merged = oof.merge(
        cohort[[
            "caseid",
            "aki",
            "baseline_cr",
            "n_preop_labs",
            "n_postop_labs",
            "valid_len",
            "obs_frac_ART_MBP",
            "art_mbp_zero_frac",
        ]],
        on="caseid",
        how="left",
    )

    y_true = merged["y_true"].to_numpy(dtype=int)
    y_prob = merged["y_prob"].to_numpy(dtype=float)
    threshold = pick_threshold(y_true, y_prob)

    merged["y_pred"] = (merged["y_prob"] >= threshold).astype(int)
    merged["confusion"] = np.select(
        [
            (merged["y_true"] == 1) & (merged["y_pred"] == 1),
            (merged["y_true"] == 0) & (merged["y_pred"] == 0),
            (merged["y_true"] == 0) & (merged["y_pred"] == 1),
            (merged["y_true"] == 1) & (merged["y_pred"] == 0),
        ],
        ["TP", "TN", "FP", "FN"],
        default="NA",
    )

    # Representative cases for section 4.6.1-4.6.3
    tp = merged[merged["confusion"] == "TP"].sort_values("y_prob", ascending=False).head(1)
    tn = merged[merged["confusion"] == "TN"].sort_values("y_prob", ascending=True).head(1)
    fp = merged[merged["confusion"] == "FP"].sort_values("y_prob", ascending=False).head(3)
    fn = merged[merged["confusion"] == "FN"].sort_values("y_prob", ascending=True).head(3)

    rep_cases = pd.concat([tp, tn, fp, fn], axis=0).copy()
    rep_cols = [
        "caseid", "confusion", "y_true", "y_prob", "baseline_cr", "n_preop_labs", "n_postop_labs",
        "valid_len", "obs_frac_ART_MBP", "art_mbp_zero_frac"
    ]
    rep_cases = rep_cases[rep_cols].reset_index(drop=True)
    rep_out = RES / "chapter_4_6_representative_cases.csv"
    rep_cases.to_csv(rep_out, index=False)

    # Error pattern summary for FP and FN
    fp_df = merged[merged["confusion"] == "FP"]
    fn_df = merged[merged["confusion"] == "FN"]
    err_stats = pd.DataFrame([
        summarize_error_group(fp_df, "FP"),
        summarize_error_group(fn_df, "FN"),
    ])
    err_out = RES / "chapter_4_6_error_patterns.csv"
    err_stats.to_csv(err_out, index=False)

    # Section 4.4 values
    row_syn = summary.loc[summary["model"] == "temporal_synergy"].iloc[0]
    row_tcn = summary.loc[summary["model"] == "tcn"].iloc[0]
    row_drnn = summary.loc[summary["model"] == "dilated_rnn"].iloc[0]
    sig_syn = sig.loc[sig["model"] == "temporal_synergy"].iloc[0]

    # Report text
    report = []
    report.append("# Chapter 4.4-4.6 Execution Report")
    report.append("")
    report.append(f"Generated from experiment: {EXP}")
    report.append("")
    report.append("## 1) Restore and execution status")
    report.append("- Project state restored to committed baseline before March (working tree cleaned from uncommitted March edits).")
    report.append("- Checklist execution performed against existing artifacts in new_optional_exp.")
    report.append("")
    report.append("## 2) Section 4.4 results from existing data")
    report.append("")
    report.append("### 4.4.1 Contribution of TCN branch (implicit ablation)")
    report.append(f"- SynerT ROC-AUC: {row_syn['roc_auc_mean']:.4f} +/- {row_syn['roc_auc_std']:.4f}")
    report.append(f"- Standalone TCN ROC-AUC: {row_tcn['roc_auc_mean']:.4f} +/- {row_tcn['roc_auc_std']:.4f}")
    report.append(f"- Delta (SynerT - TCN): {row_syn['roc_auc_mean'] - row_tcn['roc_auc_mean']:+.4f}")
    report.append(f"- SynerT PR-AUC: {row_syn['pr_auc_mean']:.4f} vs TCN PR-AUC: {row_tcn['pr_auc_mean']:.4f}")
    report.append("")
    report.append("### 4.4.2 Contribution of Dilated RNN branch (implicit ablation)")
    report.append(f"- Standalone Dilated RNN ROC-AUC: {row_drnn['roc_auc_mean']:.4f} +/- {row_drnn['roc_auc_std']:.4f}")
    report.append(f"- SynerT ROC-AUC: {row_syn['roc_auc_mean']:.4f} +/- {row_syn['roc_auc_std']:.4f}")
    report.append(f"- Delta (SynerT - Dilated RNN): {row_syn['roc_auc_mean'] - row_drnn['roc_auc_mean']:+.4f}")
    report.append(f"- SynerT vs Dilated RNN PR-AUC delta: {sig_syn['pr_auc_mean_delta']:+.4f}, p-value={sig_syn['pr_auc_pvalue']:.4f} (one-sided Wilcoxon)")
    report.append("")
    report.append("### 4.4.3 and 4.4.4")
    report.append("- Not executable from current artifacts (no explicit no-attention or fusion-variant runs).")
    report.append("- Required new runs are listed in section 5 below.")
    report.append("")
    report.append("## 3) Section 4.5 status")
    report.append("- 4.5.1 Observation windows (10/20/30/60 min): missing")
    report.append("- 4.5.2 Hyperparameter sensitivity sweep: missing")
    report.append("- 4.5.3 Missingness stress tests: missing")
    report.append("- 4.5.4 Clinical subgroup analysis: missing")
    report.append("")
    report.append("## 4) Section 4.6 executed outputs")
    report.append(f"- Operating threshold chosen from OOF by max-F1: {threshold:.4f}")
    report.append(f"- TP count: {int((merged['confusion'] == 'TP').sum())}")
    report.append(f"- TN count: {int((merged['confusion'] == 'TN').sum())}")
    report.append(f"- FP count: {int((merged['confusion'] == 'FP').sum())}")
    report.append(f"- FN count: {int((merged['confusion'] == 'FN').sum())}")
    report.append("- Representative cases exported to chapter_4_6_representative_cases.csv")
    report.append("- Error-pattern summary exported to chapter_4_6_error_patterns.csv")
    report.append("")
    report.append("### 4.6.1 TP/TN representative cases")
    if not tp.empty:
        tpr = tp.iloc[0]
        report.append(f"- TP caseid={int(tpr['caseid'])}, prob={float(tpr['y_prob']):.4f}, baseline_cr={float(tpr['baseline_cr']):.2f}")
    if not tn.empty:
        tnr = tn.iloc[0]
        report.append(f"- TN caseid={int(tnr['caseid'])}, prob={float(tnr['y_prob']):.4f}, baseline_cr={float(tnr['baseline_cr']):.2f}")
    report.append("")
    report.append("### 4.6.2 False-positive patterns")
    if not fp_df.empty:
        report.append(f"- FP median baseline_cr={fp_df['baseline_cr'].median():.2f}")
        report.append(f"- FP median obs_frac_ART_MBP={fp_df['obs_frac_ART_MBP'].median():.3f}")
        report.append(f"- FP median art_mbp_zero_frac={fp_df['art_mbp_zero_frac'].median():.3f}")
    report.append("")
    report.append("### 4.6.3 False-negative patterns")
    if not fn_df.empty:
        report.append(f"- FN median baseline_cr={fn_df['baseline_cr'].median():.2f}")
        report.append(f"- FN median obs_frac_ART_MBP={fn_df['obs_frac_ART_MBP'].median():.3f}")
        report.append(f"- FN median art_mbp_zero_frac={fn_df['art_mbp_zero_frac'].median():.3f}")
    report.append("")
    report.append("### 4.6.4 Temporal attention interpretation")
    report.append("- Not executable from current artifacts (attention weights not exported).")
    report.append("")
    report.append("## 5) Required additional runs (main repo)")
    report.append("1. True ablations: SynerT_no_attention, SynerT_no_tcn_branch, SynerT_no_rnn_branch.")
    report.append("2. Fusion variants: early_fusion, late_fusion, no_attention_fusion.")
    report.append("3. Window sensitivity: t_cut_sec in {600, 1200, 1800, 3600} with same fold/seed protocol.")
    report.append("4. Hyperparameter sweep: lr, dropout, hidden_dim, class-imbalance loss params.")
    report.append("5. Missingness stress test: channel dropout and random mask escalation.")
    report.append("6. Subgroup metrics: age/sex/surgery/risk bins with calibration and CI.")
    report.append("7. Attention export: per-case per-time-step attention weights for TP/FP/FN case narratives.")

    report_out = RES / "chapter_4_4_to_4_6_execution_report.md"
    report_out.write_text("\n".join(report), encoding="utf-8")

    print(f"Saved: {report_out}")
    print(f"Saved: {rep_out}")
    print(f"Saved: {err_out}")


if __name__ == "__main__":
    main()
