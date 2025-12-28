#!/usr/bin/env python3
"""Update all_models_5fold_summary.csv with all available model results."""

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
        default="demo_5signals",
        help="Experiment name (default: demo_5signals)",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(None, args.experiment_name)
    paths = get_paths(config)
    results_dir = paths["results_dir"]

    # Find all model metric files
    model_files = list(results_dir.glob("*_5fold_metrics.csv"))
    print(f"Found {len(model_files)} model metric files:")
    for f in sorted(model_files):
        print(f"  - {f.name}")

    if not model_files:
        print("No model results found!")
        return

    # Load all results
    all_results = []
    for f in sorted(model_files):
        try:
            df = pd.read_csv(f)
            # Filter out rows with NaN metrics
            df = df.dropna(subset=["roc_auc", "pr_auc"])
            if len(df) > 0:
                all_results.append(df)
                print(f"  Loaded {len(df)} folds from {f.name}")
        except Exception as e:
            print(f"  Warning: Failed to load {f.name}: {e}")

    if not all_results:
        print("No valid model results found!")
        return

    # Combine and calculate summary
    df_all = pd.concat(all_results, axis=0).reset_index(drop=True)
    summary = df_all.groupby("model")[["roc_auc", "pr_auc"]].agg(["mean", "std"]).sort_index()
    summary_csv = summary.copy()
    summary_csv.columns = [f"{a}_{b}" for a, b in summary_csv.columns.to_list()]

    # Save
    summary_path = results_dir / "all_models_5fold_summary.csv"
    summary_csv.to_csv(summary_path, index=True)
    print(f"\n✓ Updated summary saved to: {summary_path}")
    print("\nSummary Statistics:")
    print(summary_csv.to_string())
    print(f"\nTotal models: {len(summary_csv)}")


if __name__ == "__main__":
    main()

