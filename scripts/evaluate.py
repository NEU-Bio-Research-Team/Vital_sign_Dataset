#!/usr/bin/env python3
"""Evaluation script for VitalDB AKI prediction."""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# Add src to path for direct script execution
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vitaldb_aki.config import load_config
from vitaldb_aki.evaluation.visualizations import (
    plot_roc_curves,
    plot_pr_curves,
    plot_confusion_matrix,
)
from vitaldb_aki.utils.paths import get_paths


def main():
    """Main evaluation pipeline."""
    parser = argparse.ArgumentParser(description="Evaluate models for AKI prediction")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to YAML config file (default: use default config)",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="demo_5signals",
        help="Experiment name (default: demo_5signals)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model to evaluate (lstm, bilstm, gru, tcn, attention, transformer, dilated_conv, dilated_rnn, wavenet, temporal_synergy, tcn_attention, wavenet_rnn). If not specified, evaluates best model by PR-AUC.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold for confusion matrix (default: 0.5)",
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save plots to files instead of displaying",
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

    # Determine which model to evaluate
    model_name = args.model
    if model_name is None:
        # Auto-pick best by PR mean
        summary_path = paths["results_dir"] / "all_models_5fold_summary.csv"
        if summary_path.exists():
            s = pd.read_csv(summary_path, index_col=0)
            if "pr_auc_mean" in s.columns:
                model_name = str(s["pr_auc_mean"].idxmax())
        if model_name is None:
            model_name = "tcn"
    model_name = model_name.lower().strip()
    print(f"Evaluating model: {model_name}")

    # Generate plots
    if args.save_plots:
        roc_path = paths["results_dir"] / f"{model_name}_roc.png"
        pr_path = paths["results_dir"] / f"{model_name}_pr.png"
        cm_path = paths["results_dir"] / f"{model_name}_cm.png"
    else:
        roc_path = None
        pr_path = None
        cm_path = None

    print("\n=== Plotting ROC curves ===")
    plot_roc_curves(model_name, folds, config, save_path=roc_path)

    print("\n=== Plotting PR curves ===")
    plot_pr_curves(model_name, folds, config, save_path=pr_path)

    print("\n=== Plotting confusion matrix ===")
    metrics = plot_confusion_matrix(
        model_name, folds, config, threshold=args.threshold, save_path=cm_path
    )

    print("\n=== Evaluation complete ===")


if __name__ == "__main__":
    main()

