#!/usr/bin/env python3
"""Re-evaluate models from checkpoints and update metrics."""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# Add src to path for direct script execution
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vitaldb_aki.config import load_config
from vitaldb_aki.training.trainer import train_all_folds
from vitaldb_aki.utils.paths import get_paths


MODEL_DISPLAY_NAMES = {
    "temporal_synergy": "SynerT",
}


def _display_model_name(name: str) -> str:
    key = (name or "").lower().strip()
    return MODEL_DISPLAY_NAMES.get(key, name)


def main():
    """Re-evaluate models and update metrics."""
    parser = argparse.ArgumentParser(description="Re-evaluate models from checkpoints")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to YAML config file (default: use default config)",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="new_optional_exp",
        help="Experiment name (default: new_optional_exp)",
    )
    parser.add_argument(
        "--model",
        type=str,
        nargs="+",
        help="Model(s) to re-evaluate (e.g., tcn temporal_synergy). If not specified, re-evaluates all models.",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config, args.experiment_name)
    print(f"Using experiment: {args.experiment_name}")
    print(f"Artifacts directory: {config.artifacts_dir}")

    # Load folds
    paths = get_paths(config)
    folds_path = paths["artifacts_dir"] / "folds.json"
    if not folds_path.exists():
        raise FileNotFoundError(
            f"Folds not found at {folds_path}. Run preprocessing first."
        )

    with open(folds_path, "r", encoding="utf-8") as f:
        folds = json.load(f)

    # Determine which models to re-evaluate
    if args.model:
        models_to_reevaluate = [m.lower().strip() for m in args.model]
    else:
        # Re-evaluate all models that have checkpoints
        models_dir = paths["models_dir"]
        models_to_reevaluate = [
            d.name for d in models_dir.iterdir() if d.is_dir() and (d / "fold1.pt").exists()
        ]
        models_to_reevaluate.sort()

    print(f"\nRe-evaluating {len(models_to_reevaluate)} model(s): {', '.join(models_to_reevaluate)}")

    # Re-evaluate each model
    all_results = []
    for model_name in models_to_reevaluate:
        print(f"\n=== Re-evaluating {_display_model_name(model_name)} ===")
        # This will load checkpoints and re-evaluate without retraining
        df_results = train_all_folds(model_name, folds, config, force_train=False)
        all_results.append(df_results)
        print(f"✓ Completed {model_name}")

    # Update summary
    if all_results:
        df_all = pd.concat(all_results, axis=0).reset_index(drop=True)
        summary = df_all.groupby("model")[["roc_auc", "pr_auc"]].agg(["mean", "std"]).sort_index()
        summary_path = paths["results_dir"] / "all_models_5fold_summary.csv"
        summary_csv = summary.copy()
        summary_csv.columns = [f"{a}_{b}" for a, b in summary_csv.columns.to_list()]
        # Sort by ROC-AUC descending
        summary_csv = summary_csv.sort_values("roc_auc_mean", ascending=False)
        summary_csv.to_csv(summary_path, index=True)
        print(f"\n✓ Updated summary saved to: {summary_path}")
        print("\nUpdated Summary Statistics:")
        print(summary_csv.to_string())
        print(f"\nTotal models: {len(summary_csv)}")

    print("\n=== Re-evaluation complete ===")


if __name__ == "__main__":
    main()

