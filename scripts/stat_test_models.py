#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Statistical comparison between two models.

Note: This script adds the repo's `src/` to `sys.path` at runtime, so static
analysis may not resolve imports unless your environment is configured.

Supports:
- (B, recommended) Paired test across matched (seed, fold) runs (n = seeds × folds)
- (A, supplementary) Bootstrap CI from OOF predictions (non-parametric)

Typical usage (B, hypothesis testing):
    # After training with multiple seeds (writes *_5fold_metrics_seed*.csv)
    python scripts/stat_test_models.py --model-a temporal_synergy --model-b dilated_rnn --metric auprc --pairing seed_fold

If you want (A) CI for deployment-style reporting:
    python scripts/evaluate.py --model temporal_synergy --seeds 1 2 3 4 5 --save-preds
    python scripts/evaluate.py --model dilated_rnn     --seeds 1 2 3 4 5 --save-preds
    python scripts/stat_test_models.py --model-a temporal_synergy --model-b dilated_rnn --tag ensemble_mean --pairing fold
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Add src to path for direct script execution
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vitaldb_aki.config import load_config
from vitaldb_aki.evaluation.statistics import (
    bootstrap_ci,
    get_metric_fn,
    paired_bootstrap_delta_ci,
    paired_test,
)
from vitaldb_aki.utils.paths import get_paths


MODEL_INPUT_ALIASES = {
    "synert": "temporal_synergy",
    "syner_t": "temporal_synergy",
}


def _normalize(name: str) -> str:
    key = (name or "").lower().strip()
    return MODEL_INPUT_ALIASES.get(key, key)


def _oof_path(results_dir: Path, model: str, tag: str) -> Path:
    if tag:
        return results_dir / f"{model}_oof_predictions_{tag}.csv"
    return results_dir / f"{model}_oof_predictions.csv"


def _metrics_path(results_dir: Path, model: str, tag: str) -> Path:
    suffix = f"_{tag}" if tag else ""
    return results_dir / f"{model}_5fold_metrics{suffix}.csv"


def _parse_seed_from_filename(path: Path) -> Optional[int]:
    name = path.name
    marker = "_seed"
    if marker not in name:
        return None
    tail = name.split(marker, 1)[1]
    digits = "".join(ch for ch in tail if ch.isdigit())
    return int(digits) if digits else None


def _load_seed_fold_metrics(results_dir: Path, model: str) -> pd.DataFrame:
    files = sorted(results_dir.glob(f"{model}_5fold_metrics_seed*.csv"))
    if not files:
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    for f in files:
        d = pd.read_csv(f)
        if "seed" not in d.columns:
            s = _parse_seed_from_filename(f)
            if s is not None:
                d["seed"] = int(s)
        frames.append(d)
    return pd.concat(frames, axis=0).reset_index(drop=True)


def _load_and_align_oof(path_a: Path, path_b: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_a = pd.read_csv(path_a)
    df_b = pd.read_csv(path_b)

    req = {"caseid", "fold", "y_true", "y_prob"}
    for p, d in [(path_a, df_a), (path_b, df_b)]:
        missing = req - set(d.columns)
        if missing:
            raise ValueError(f"{p.name} missing columns: {sorted(missing)}")

    # Merge on caseid/fold/y_true to guarantee correct pairing.
    m = df_a.merge(
        df_b,
        on=["caseid", "fold", "y_true"],
        how="inner",
        suffixes=("_a", "_b"),
    )

    if len(m) == 0:
        raise RuntimeError(
            "No matched OOF rows between the two models. "
            "Ensure they were evaluated on the same folds/caseids and have the same y_true." 
        )

    # Return aligned views
    a = m[["caseid", "fold", "y_true", "y_prob_a"]].rename(columns={"y_prob_a": "y_prob"})
    b = m[["caseid", "fold", "y_true", "y_prob_b"]].rename(columns={"y_prob_b": "y_prob"})
    return a, b, m


def main() -> None:
    p = argparse.ArgumentParser(description="Bootstrap CI + paired tests for model comparison")
    p.add_argument("--config", type=Path, default=None, help="Path to YAML config (default: use default config)")
    p.add_argument("--experiment-name", type=str, default="new_optional_exp")
    p.add_argument("--model-a", type=str, required=True, help="Candidate model (e.g., temporal_synergy)")
    p.add_argument("--model-b", type=str, required=True, help="Baseline model (e.g., dilated_rnn)")
    p.add_argument("--metric", choices=["auprc", "auroc"], default="auprc")
    p.add_argument(
        "--tag",
        type=str,
        default="",
        help="OOF tag: '', 'ensemble_mean', or 'seed{N}' (must match filenames)",
    )
    p.add_argument(
        "--pairing",
        choices=["seed_fold", "fold"],
        default="seed_fold",
        help=(
            "How to form paired runs for hypothesis testing. "
            "'seed_fold' loads *_5fold_metrics_seed*.csv and pairs on (seed, fold) (recommended). "
            "'fold' pairs on folds only using the metrics file selected by --tag."
        ),
    )
    p.add_argument("--n-bootstrap", type=int, default=5000)
    p.add_argument("--alpha", type=float, default=0.05, help="alpha=0.05 -> 95%% CI")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-stratified", action="store_true", help="Disable stratified bootstrap")
    p.add_argument(
        "--paired-test",
        choices=["wilcoxon", "ttest", "none"],
        default="wilcoxon",
        help="Paired test on fold metrics",
    )
    p.add_argument(
        "--alternative",
        choices=["greater", "two-sided", "less"],
        default="greater",
        help="H1: model_a > model_b (default)",
    )
    p.add_argument(
        "--out-json",
        type=str,
        default="",
        help="Optional filename to save a JSON summary into results_dir",
    )

    args = p.parse_args()

    model_a = _normalize(args.model_a)
    model_b = _normalize(args.model_b)

    config = load_config(args.config, args.experiment_name)
    paths = get_paths(config)
    results_dir: Path = paths["results_dir"]

    tag = (args.tag or "").strip()

    metric_fn = get_metric_fn(args.metric)
    stratified = not args.no_stratified
    k = "pr_auc" if args.metric == "auprc" else "roc_auc"

    # (B) Paired hypothesis test (primary)
    paired = None
    if args.paired_test != "none":
        if args.pairing == "seed_fold":
            ma = _load_seed_fold_metrics(results_dir, model_a)
            mb = _load_seed_fold_metrics(results_dir, model_b)
            if ma.empty or mb.empty:
                raise FileNotFoundError(
                    "Missing seed-level metrics files for seed-fold pairing. Expected files like: "
                    f"{model_a}_5fold_metrics_seed{{N}}.csv and {model_b}_5fold_metrics_seed{{N}}.csv in {results_dir}"
                )
            reqm = {"seed", "fold", "pr_auc", "roc_auc"}
            if not reqm.issubset(ma.columns) or not reqm.issubset(mb.columns):
                raise ValueError("Seed metrics missing required columns (need seed, fold, pr_auc, roc_auc)")
            mm = ma[["seed", "fold", k]].merge(
                mb[["seed", "fold", k]],
                on=["seed", "fold"],
                how="inner",
                suffixes=("_a", "_b"),
            )
            if len(mm) < 2:
                raise RuntimeError("Not enough matched (seed, fold) pairs to run a paired test")
            paired = paired_test(
                mm[f"{k}_a"].to_numpy(float),
                mm[f"{k}_b"].to_numpy(float),
                test=args.paired_test,
                alternative=args.alternative,
            )
        else:
            mpath_a = _metrics_path(results_dir, model_a, tag)
            mpath_b = _metrics_path(results_dir, model_b, tag)
            if not (mpath_a.exists() and mpath_b.exists()):
                raise FileNotFoundError(
                    f"Missing metrics CSV for fold pairing: {mpath_a} or {mpath_b}. "
                    "Run scripts/evaluate.py (or training) to generate them."
                )
            ma = pd.read_csv(mpath_a)
            mb = pd.read_csv(mpath_b)
            reqm = {"fold", "pr_auc", "roc_auc"}
            if not reqm.issubset(ma.columns) or not reqm.issubset(mb.columns):
                raise ValueError("Metrics CSV missing required columns")
            mm = ma[["fold", k]].merge(mb[["fold", k]], on=["fold"], how="inner", suffixes=("_a", "_b"))
            if len(mm) < 2:
                raise RuntimeError("Not enough matched fold pairs to run a paired test")
            paired = paired_test(
                mm[f"{k}_a"].to_numpy(float),
                mm[f"{k}_b"].to_numpy(float),
                test=args.paired_test,
                alternative=args.alternative,
            )

    # (A) Optional OOF bootstrap CI (supplementary)
    a_point = a_lo = a_hi = b_point = b_lo = b_hi = d_point = d_lo = d_hi = p_boot = float("nan")
    n_oof = 0
    prevalence = float("nan")
    try:
        path_a = _oof_path(results_dir, model_a, tag)
        path_b = _oof_path(results_dir, model_b, tag)
        if path_a.exists() and path_b.exists():
            _, _, merged = _load_and_align_oof(path_a, path_b)
            y = merged["y_true"].to_numpy(dtype=int)
            pa = merged["y_prob_a"].to_numpy(dtype=float)
            pb = merged["y_prob_b"].to_numpy(dtype=float)
            n_oof = int(len(merged))
            prevalence = float(np.mean(y))

            a_point, (a_lo, a_hi), _ = bootstrap_ci(
                metric_fn,
                y,
                pa,
                n_bootstrap=int(args.n_bootstrap),
                alpha=float(args.alpha),
                seed=int(args.seed),
                stratified=stratified,
            )
            b_point, (b_lo, b_hi), _ = bootstrap_ci(
                metric_fn,
                y,
                pb,
                n_bootstrap=int(args.n_bootstrap),
                alpha=float(args.alpha),
                seed=int(args.seed) + 1,
                stratified=stratified,
            )
            d_point, (d_lo, d_hi), d_boot = paired_bootstrap_delta_ci(
                metric_fn,
                y,
                pa,
                pb,
                n_bootstrap=int(args.n_bootstrap),
                alpha=float(args.alpha),
                seed=int(args.seed) + 2,
                stratified=stratified,
            )
            p_greater = float(np.mean(d_boot <= 0.0))
            p_less = float(np.mean(d_boot >= 0.0))
            if args.alternative == "greater":
                p_boot = p_greater
            elif args.alternative == "less":
                p_boot = p_less
            else:
                p_boot = float(2.0 * min(p_greater, p_less))
    except Exception as e:
        print(f"Note: skipping OOF bootstrap CI (could not load/align OOF): {e}")

    ci_pct = int(round((1.0 - float(args.alpha)) * 100))
    print("\n=== Statistical comparison ===")
    print(f"Experiment: {args.experiment_name}")
    print(f"Pairing: {args.pairing}")
    print(f"Tag (for optional OOF CI): {tag or '(none)'}")
    print(f"Metric: {args.metric}")
    if paired is not None:
        label = "(seed, fold) runs" if args.pairing == "seed_fold" else "folds"
        print("\nPaired hypothesis test")
        print(
            f"{paired.test} (alt={paired.alternative}) on n={paired.n} paired {label}: "
            f"mean(Δ)={paired.mean_delta:.6f} median(Δ)={paired.median_delta:.6f} "
            f"stat={paired.statistic:.6g} p={paired.pvalue:.6g}"
        )

    if np.isfinite(a_point) and np.isfinite(b_point):
        print(f"\nN cases (OOF aligned): {n_oof} | prevalence={prevalence:.4f}")
        print("\nBootstrap CI (supplementary, non-parametric)")
        print(f"{model_a}: {a_point:.6f} | {ci_pct}% CI [{a_lo:.6f}, {a_hi:.6f}]")
        print(f"{model_b}: {b_point:.6f} | {ci_pct}% CI [{b_lo:.6f}, {b_hi:.6f}]")
        print("\nPaired bootstrap (supplementary; delta = a - b)")
        print(f"delta: {d_point:.6f} | {ci_pct}% CI [{d_lo:.6f}, {d_hi:.6f}] | p_boot({args.alternative})={p_boot:.6g}")

    if args.out_json:
        import json

        out = {
            "experiment": args.experiment_name,
            "tag": tag,
            "metric": args.metric,
            "model_a": model_a,
            "model_b": model_b,
            "pairing": args.pairing,
            "oof_n": int(n_oof),
            "oof_prevalence": float(prevalence) if np.isfinite(prevalence) else None,
            "bootstrap": {
                "ci_level": float(1.0 - float(args.alpha)),
                "model_a": {"point": a_point, "lo": a_lo, "hi": a_hi},
                "model_b": {"point": b_point, "lo": b_lo, "hi": b_hi},
            },
            "paired_bootstrap": {
                "delta_point": d_point,
                "delta_lo": d_lo,
                "delta_hi": d_hi,
                "p_boot": p_boot,
                "alternative": args.alternative,
            },
        }
        if paired is not None:
            out["paired_test"] = {
                "test": paired.test,
                "alternative": paired.alternative,
                "n": paired.n,
                "mean_delta": paired.mean_delta,
                "median_delta": paired.median_delta,
                "statistic": paired.statistic,
                "pvalue": paired.pvalue,
            }

        out_path = results_dir / args.out_json
        out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nSaved JSON: {out_path}")


if __name__ == "__main__":
    main()
