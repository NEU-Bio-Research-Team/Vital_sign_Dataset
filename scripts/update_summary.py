#!/usr/bin/env python3
"""Rebuild summary CSVs from all available model metrics.

Prefers ensemble metrics ("*_5fold_metrics_ensemble_mean.csv") when present.
Also writes a seed-level diagnostic summary from "*_5fold_metrics_seed*.csv".
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

# Add src to path for direct script execution
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vitaldb_aki.config import load_config
from vitaldb_aki.evaluation.statistics import paired_test
from vitaldb_aki.utils.paths import get_paths


def main():
    """Update summary with all available model results."""
    parser = argparse.ArgumentParser(description="Update model summary CSV")
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="new_optional_exp",
        help="Experiment name (default: new_optional_exp)",
    )
    parser.add_argument(
        "--baseline-model",
        type=str,
        default=None,
        help=(
            "Optional baseline model key to compute paired significance vs each other model "
            "using fold-level metrics. Example: dilated_rnn"
        ),
    )
    parser.add_argument(
        "--paired-test",
        choices=["wilcoxon", "ttest"],
        default="wilcoxon",
        help="Paired test to use when --baseline-model is provided (default: wilcoxon).",
    )
    parser.add_argument(
        "--significance-source",
        choices=["seeds", "main"],
        default="seeds",
        help=(
            "Which metrics to use for significance vs baseline. "
            "'seeds' pairs across (seed, fold) using *_5fold_metrics_seed*.csv (recommended). "
            "'main' uses the same metrics used for the main summary (typically ensemble_mean or single-run)."
        ),
    )
    parser.add_argument(
        "--alternative",
        choices=["greater", "two-sided", "less"],
        default="greater",
        help="Alternative hypothesis for paired test (default: greater => model > baseline).",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(None, args.experiment_name)
    paths = get_paths(config)
    results_dir = paths["results_dir"]

    def _seed_from_filename(path: Path) -> Optional[int]:
        name = path.name
        marker = "_seed"
        if marker not in name:
            return None
        tail = name.split(marker, 1)[1]
        digits = "".join(ch for ch in tail if ch.isdigit())
        return int(digits) if digits else None

    def _load_metric_files(files: list[Path], label: str) -> list[pd.DataFrame]:
        frames: list[pd.DataFrame] = []
        print(f"Found {len(files)} {label} metric files:")
        for f in sorted(files):
            print(f"  - {f.name}")
        for f in sorted(files):
            try:
                df = pd.read_csv(f)
                # If this is a seed-tagged file, make sure we carry an explicit seed column.
                if "seed" not in df.columns:
                    s = _seed_from_filename(f)
                    if s is not None:
                        df["seed"] = int(s)
                df = df.dropna(subset=["roc_auc", "pr_auc"])
                if len(df) > 0:
                    frames.append(df)
                    print(f"  Loaded {len(df)} rows from {f.name}")
            except Exception as e:
                print(f"  Warning: Failed to load {f.name}: {e}")
        return frames

    # 1) Main summary: prefer ensemble_mean metrics if present
    ens_files = list(results_dir.glob("*_5fold_metrics_ensemble_mean.csv"))
    base_files: list[Path]
    if ens_files:
        base_files = ens_files
        base_label = "ensemble_mean"
    else:
        base_files = list(results_dir.glob("*_5fold_metrics.csv"))
        base_label = "base"

    base_frames = _load_metric_files(base_files, base_label)
    if not base_frames:
        print("No valid metrics found for main summary!")
    else:
        df_all = pd.concat(base_frames, axis=0).reset_index(drop=True)
        summary = (
            df_all.groupby("model")[["roc_auc", "pr_auc"]]
            .agg(["mean", "std"])
            .sort_index()
        )
        summary_csv = summary.copy()
        summary_csv.columns = [f"{a}_{b}" for a, b in summary_csv.columns.to_list()]
        summary_path = results_dir / "all_models_5fold_summary.csv"
        summary_csv.to_csv(summary_path, index=True)
        print(f"\n✓ Updated summary saved to: {summary_path}")
        print("\nSummary Statistics:")
        print(summary_csv.to_string())
        print(f"\nTotal models: {len(summary_csv)}")

        # 1b) Optional paired significance vs a baseline model
        if args.baseline_model is not None:
            baseline = str(args.baseline_model).strip()
            sig_rows: list[dict] = []

            if args.significance_source == "seeds":
                # Load per-seed metrics files (written by trainer with run_tag=seed{N}).
                seed_files = list(results_dir.glob("*_5fold_metrics_seed*.csv"))
                seed_frames = _load_metric_files(seed_files, "seed-for-significance")
                if not seed_frames:
                    print("\nWarning: no *_5fold_metrics_seed*.csv found; cannot run seed-fold significance.")
                else:
                    df_sig_src = pd.concat(seed_frames, axis=0).reset_index(drop=True)
                    # Pair on (seed, fold) if seed exists; otherwise fall back to fold.
                    pair_keys = ["seed", "fold"] if "seed" in df_sig_src.columns else ["fold"]

                    if baseline not in set(df_sig_src["model"].astype(str)):
                        print(f"\nWarning: baseline model '{baseline}' not found in seed metrics; skipping significance.")
                    else:
                        base_df = df_sig_src[df_sig_src["model"].astype(str) == baseline].copy()
                        base_df = base_df[pair_keys + ["roc_auc", "pr_auc"]]
                        base_df = base_df.rename(columns={"roc_auc": "roc_auc_base", "pr_auc": "pr_auc_base"})

                        for model, df_m in df_sig_src.groupby(df_sig_src["model"].astype(str)):
                            if model == baseline:
                                continue
                            df_m = df_m[pair_keys + ["roc_auc", "pr_auc"]]
                            merged = df_m.merge(base_df, on=pair_keys, how="inner")
                            if len(merged) < 2:
                                continue

                            roc_res = paired_test(
                                merged["roc_auc"].to_numpy(float),
                                merged["roc_auc_base"].to_numpy(float),
                                test=args.paired_test,
                                alternative=args.alternative,
                            )
                            pr_res = paired_test(
                                merged["pr_auc"].to_numpy(float),
                                merged["pr_auc_base"].to_numpy(float),
                                test=args.paired_test,
                                alternative=args.alternative,
                            )
                            sig_rows.append(
                                {
                                    "model": model,
                                    "baseline": baseline,
                                    "n_pairs": int(pr_res.n),
                                    "roc_auc_mean_delta": roc_res.mean_delta,
                                    "roc_auc_median_delta": roc_res.median_delta,
                                    "roc_auc_pvalue": roc_res.pvalue,
                                    "pr_auc_mean_delta": pr_res.mean_delta,
                                    "pr_auc_median_delta": pr_res.median_delta,
                                    "pr_auc_pvalue": pr_res.pvalue,
                                    "paired_test": pr_res.test,
                                    "alternative": pr_res.alternative,
                                    "pair_keys": "+".join(pair_keys),
                                }
                            )
            else:
                # Use whatever is in the main summary source (fold pairing).
                if baseline not in set(df_all["model"].astype(str)):
                    print(f"\nWarning: baseline model '{baseline}' not found in available metrics; skipping significance.")
                else:
                    pair_keys = ["fold"]
                    if "seed" in df_all.columns:
                        pair_keys = ["seed", "fold"]
                    base_df = df_all[df_all["model"].astype(str) == baseline].copy()
                    base_df = base_df[pair_keys + ["roc_auc", "pr_auc"]]
                    base_df = base_df.rename(columns={"roc_auc": "roc_auc_base", "pr_auc": "pr_auc_base"})
                    for model, df_m in df_all.groupby(df_all["model"].astype(str)):
                        if model == baseline:
                            continue
                        df_m = df_m[pair_keys + ["roc_auc", "pr_auc"]]
                        merged = df_m.merge(base_df, on=pair_keys, how="inner")
                        if len(merged) < 2:
                            continue
                        roc_res = paired_test(
                            merged["roc_auc"].to_numpy(float),
                            merged["roc_auc_base"].to_numpy(float),
                            test=args.paired_test,
                            alternative=args.alternative,
                        )
                        pr_res = paired_test(
                            merged["pr_auc"].to_numpy(float),
                            merged["pr_auc_base"].to_numpy(float),
                            test=args.paired_test,
                            alternative=args.alternative,
                        )
                        sig_rows.append(
                            {
                                "model": model,
                                "baseline": baseline,
                                "n_pairs": int(pr_res.n),
                                "roc_auc_mean_delta": roc_res.mean_delta,
                                "roc_auc_median_delta": roc_res.median_delta,
                                "roc_auc_pvalue": roc_res.pvalue,
                                "pr_auc_mean_delta": pr_res.mean_delta,
                                "pr_auc_median_delta": pr_res.median_delta,
                                "pr_auc_pvalue": pr_res.pvalue,
                                "paired_test": pr_res.test,
                                "alternative": pr_res.alternative,
                                "pair_keys": "+".join(pair_keys),
                            }
                        )

            if sig_rows:
                df_sig = pd.DataFrame(sig_rows).sort_values(["pr_auc_pvalue", "model"]).reset_index(drop=True)
                suffix = "seedfold" if args.significance_source == "seeds" else "main"
                sig_path = results_dir / f"all_models_5fold_significance_{suffix}_vs_{baseline}.csv"
                df_sig.to_csv(sig_path, index=False)
                print(f"\n✓ Saved significance table to: {sig_path}")
            else:
                print("\nNo paired significance results produced (not enough matched pairs).")

    # 2) Seed-level diagnostic summary (optional)
    seed_files = list(results_dir.glob("*_5fold_metrics_seed*.csv"))
    seed_frames = _load_metric_files(seed_files, "seed")
    if seed_frames:
        df_seed = pd.concat(seed_frames, axis=0).reset_index(drop=True)
        seed_summary = (
            df_seed.groupby("model")[["roc_auc", "pr_auc"]]
            .agg(["mean", "std"])
            .sort_index()
        )
        seed_summary_csv = seed_summary.copy()
        seed_summary_csv.columns = [f"{a}_{b}" for a, b in seed_summary_csv.columns.to_list()]
        seed_summary_path = results_dir / "all_models_5fold_summary_seeds.csv"
        seed_summary_csv.to_csv(seed_summary_path, index=True)
        print(f"\n✓ Updated seed-level summary saved to: {seed_summary_path}")
    else:
        print("No valid seed metrics found; skipping seed-level summary.")


if __name__ == "__main__":
    main()

