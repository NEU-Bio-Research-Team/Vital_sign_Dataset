#!/usr/bin/env python3
"""Training script for VitalDB AKI prediction."""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

# Add src to path for direct script execution
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vitaldb_aki.config import load_config
from vitaldb_aki.training.trainer import train_all_folds
from vitaldb_aki.utils.paths import get_paths


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description="Train models for AKI prediction")
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
        default=None,
        help="Model to train (mlp, lstm, bilstm, gru, tcn, attention, transformer, dilated_conv, dilated_rnn, wavenet, temporal_synergy, tcn_attention, wavenet_rnn). If not specified, trains all models.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force retraining even if checkpoints exist",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Optional list of random seeds for multi-seed training. If provided, trains each seed and saves seed-specific outputs, then writes an ensemble OOF by averaging probabilities.",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config, args.experiment_name)
    print(f"Using experiment: {args.experiment_name}")
    print(f"Artifacts directory: {config.artifacts_dir}")
    print(f"Signals: {list(config.signals)}")
    print(f"Required signals: {list(config.required_signals)}")
    print(f"include_optional_signals: {config.include_optional_signals}")

    # Load folds
    paths = get_paths(config)
    folds_path = paths["artifacts_dir"] / "folds.json"
    if not folds_path.exists():
        raise FileNotFoundError(
            f"Folds not found at {folds_path}. Run preprocessing first."
        )

    with open(folds_path, "r", encoding="utf-8") as f:
        folds = json.load(f)

    # Determine which models to train
    models_to_train = [
        "mlp",
        "tcn",
        "gru",
        "lstm",
        "bilstm",
        #"transformer",
        "dilated_conv",
        "dilated_rnn",
        "wavenet",
        #"wavenet_rnn",
        "temporal_synergy",
    ]
    if args.model:
        model_name = args.model.lower().strip()
        if model_name not in models_to_train:
            raise ValueError(f"Unknown model: {model_name}. Use one of: {models_to_train}")
        models_to_train = [model_name]

    # Train models
    all_seed_metrics = []
    all_ensemble_metrics = []
    for model_name in models_to_train:
        print(f"\n=== Training {model_name.upper()} ===")
        if args.seeds:
            seed_results = []
            for seed in args.seeds:
                tag = f"seed{int(seed)}"
                print(f"\n--- Seed {seed} ---")
                df_seed = train_all_folds(
                    model_name,
                    folds,
                    config,
                    force_train=args.force,
                    seed=int(seed),
                    run_tag=tag,
                )
                df_seed = df_seed.copy()
                df_seed["seed"] = int(seed)
                seed_results.append(df_seed)

            df_results = pd.concat(seed_results, axis=0).reset_index(drop=True)
            all_seed_metrics.append(df_results)

            # Build ensemble OOF by averaging y_prob across seeds
            paths = get_paths(config)
            oof_dfs = []
            for seed in args.seeds:
                tag = f"seed{int(seed)}"
                oof_path = paths["results_dir"] / f"{model_name}_oof_predictions_{tag}.csv"
                if not oof_path.exists():
                    raise FileNotFoundError(f"Missing OOF predictions for seed {seed}: {oof_path}")
                d = pd.read_csv(oof_path)
                d["seed"] = int(seed)
                oof_dfs.append(d)

            df_all = pd.concat(oof_dfs, axis=0).reset_index(drop=True)
            # Pivot to average probs per case
            df_piv = (
                df_all.pivot_table(
                    index=["caseid", "fold", "y_true"],
                    columns="seed",
                    values="y_prob",
                    aggfunc="mean",
                )
                .reset_index()
            )
            seed_cols = [c for c in df_piv.columns if isinstance(c, int)]
            if not seed_cols:
                raise RuntimeError("No seed probability columns found while building ensemble.")
            df_piv["y_prob"] = df_piv[seed_cols].mean(axis=1)
            df_ens = df_piv[["caseid", "fold", "y_true", "y_prob"]].sort_values(["fold", "caseid"]).reset_index(drop=True)

            ens_oof_path = paths["results_dir"] / f"{model_name}_oof_predictions_ensemble_mean.csv"
            df_ens.to_csv(ens_oof_path, index=False)
            print(f"Saved ensemble OOF predictions: {ens_oof_path}")

            # Compute per-fold + overall metrics for ensemble
            ens_rows = []
            for fold_idx, d in df_ens.groupby("fold"):
                y = d["y_true"].to_numpy()
                p = d["y_prob"].to_numpy()
                ens_rows.append(
                    {
                        "model": model_name,
                        "fold": int(fold_idx),
                        "roc_auc": float(roc_auc_score(y, p)),
                        "pr_auc": float(average_precision_score(y, p)),
                        "seed": "ensemble_mean",
                    }
                )

            df_ens_metrics = pd.DataFrame(ens_rows).sort_values(["model", "fold"]).reset_index(drop=True)
            ens_metrics_path = paths["results_dir"] / f"{model_name}_5fold_metrics_ensemble_mean.csv"
            df_ens_metrics.to_csv(ens_metrics_path, index=False)
            print(f"Saved ensemble metrics: {ens_metrics_path}")

            all_ensemble_metrics.append(df_ens_metrics)

        else:
            df_results = train_all_folds(model_name, folds, config, force_train=args.force)
            all_seed_metrics.append(df_results)

    # Create summary
    # Note: when using --seeds, the primary score should come from the ensemble OOF
    # (mean across seeds, then metrics across folds). We still optionally save a
    # seed-level summary for debugging.
    paths = get_paths(config)
    if args.seeds:
        if all_ensemble_metrics:
            df_all_ens = pd.concat(all_ensemble_metrics, axis=0).reset_index(drop=True)
            summary_ens = (
                df_all_ens.groupby("model")[["roc_auc", "pr_auc"]]
                .agg(["mean", "std"])
                .sort_index()
            )
            summary_path = paths["results_dir"] / "all_models_5fold_summary.csv"
            summary_csv = summary_ens.copy()
            summary_csv.columns = [f"{a}_{b}" for a, b in summary_csv.columns.to_list()]
            summary_csv.to_csv(summary_path, index=True)
            print(f"\nSaved summary: {summary_path}")
            print(summary_ens)

        if all_seed_metrics:
            df_all_seed = pd.concat(all_seed_metrics, axis=0).reset_index(drop=True)
            summary_seed = (
                df_all_seed.groupby("model")[["roc_auc", "pr_auc"]]
                .agg(["mean", "std"])
                .sort_index()
            )
            summary_seed_path = paths["results_dir"] / "all_models_5fold_summary_seeds.csv"
            summary_seed_csv = summary_seed.copy()
            summary_seed_csv.columns = [f"{a}_{b}" for a, b in summary_seed_csv.columns.to_list()]
            summary_seed_csv.to_csv(summary_seed_path, index=True)
            print(f"Saved seed-level summary: {summary_seed_path}")
    else:
        if all_seed_metrics:
            df_all = pd.concat(all_seed_metrics, axis=0).reset_index(drop=True)
            summary = df_all.groupby("model")[["roc_auc", "pr_auc"]].agg(["mean", "std"]).sort_index()
            summary_path = paths["results_dir"] / "all_models_5fold_summary.csv"
            summary_csv = summary.copy()
            summary_csv.columns = [f"{a}_{b}" for a, b in summary_csv.columns.to_list()]
            summary_csv.to_csv(summary_path, index=True)
            print(f"\nSaved summary: {summary_path}")
            print(summary)

    print("\n=== Training complete ===")


if __name__ == "__main__":
    main()

