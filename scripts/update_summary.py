#!/usr/bin/env python3
"""Rebuild summary CSVs from all available model metrics.

Prefers ensemble metrics ("*_5fold_metrics_ensemble_mean.csv") when present.
Also writes a seed-level diagnostic summary from "*_5fold_metrics_seed*.csv".
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Add src to path for direct script execution
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vitaldb_aki.config import load_config
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
    args = parser.parse_args()

    # Load config
    config = load_config(None, args.experiment_name)
    paths = get_paths(config)
    results_dir = paths["results_dir"]

    def _load_metric_files(files: list[Path], label: str) -> list[pd.DataFrame]:
        frames: list[pd.DataFrame] = []
        print(f"Found {len(files)} {label} metric files:")
        for f in sorted(files):
            print(f"  - {f.name}")
        for f in sorted(files):
            try:
                df = pd.read_csv(f)
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

