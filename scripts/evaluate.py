#!/usr/bin/env python3
"""Evaluation script for VitalDB AKI prediction."""

import argparse
import json
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

# Add src to path for direct script execution
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vitaldb_aki.config import load_config
from vitaldb_aki.utils.paths import get_paths


MODEL_INPUT_ALIASES = {
    "synert": "temporal_synergy",
    "syner_t": "temporal_synergy",
}

MODEL_DISPLAY_NAMES = {
    "temporal_synergy": "SynerT",
}


def _normalize_model_name(name: str) -> str:
    key = (name or "").lower().strip()
    return MODEL_INPUT_ALIASES.get(key, key)


def _display_model_name(name: str) -> str:
    key = (name or "").lower().strip()
    return MODEL_DISPLAY_NAMES.get(key, name)


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
        default="new_optional_exp",
        help="Experiment name (default: new_optional_exp)",
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
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save plots to files (default: enabled). Use --no-save-plots to disable.",
    )
    parser.add_argument(
        "--save-preds",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save out-of-fold predictions CSV to results dir (default: disabled).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Optional list of seeds. If provided, evaluate using ensemble_mean OOF (average probs across seeds).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Evaluate a single seed run by loading seed-specific OOF predictions (expects *_oof_predictions_seed{N}.csv).",
    )
    args = parser.parse_args()

    if args.seeds and args.seed is not None:
        raise ValueError("Use either --seeds (ensemble) OR --seed (single seed), not both.")

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
    model_name = _normalize_model_name(model_name)
    model_display = _display_model_name(model_name)
    print(f"Evaluating model: {model_display}")

    def _load_or_build_oof_predictions() -> Tuple[pd.DataFrame, str]:
        """Return (df_oof, tag) where tag is '' or 'ensemble_mean'."""
        # Single-seed evaluation: load seed-specific OOF.
        if args.seed is not None:
            tag = f"seed{int(args.seed)}"
            seed_path = paths["results_dir"] / f"{model_name}_oof_predictions_{tag}.csv"
            if not seed_path.exists():
                raise FileNotFoundError(
                    f"Missing seed OOF predictions: {seed_path}. "
                    f"Train with --seeds (including {args.seed}) first so the per-seed OOF exists."
                )
            return pd.read_csv(seed_path), tag

        # If multi-seed evaluation requested: use ensemble_mean OOF.
        if args.seeds:
            tag = "ensemble_mean"
            ens_path = paths["results_dir"] / f"{model_name}_oof_predictions_{tag}.csv"
            if ens_path.exists():
                return pd.read_csv(ens_path), tag

            oof_dfs = []
            for seed in args.seeds:
                seed_tag = f"seed{int(seed)}"
                oof_path = paths["results_dir"] / f"{model_name}_oof_predictions_{seed_tag}.csv"
                if not oof_path.exists():
                    raise FileNotFoundError(
                        f"Missing OOF predictions for seed {seed}: {oof_path}. "
                        f"Train with --seeds first (or point to the correct experiment/results dir)."
                    )
                d = pd.read_csv(oof_path)
                d["seed"] = int(seed)
                oof_dfs.append(d)

            df_all = pd.concat(oof_dfs, axis=0).reset_index(drop=True)
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
            df_ens = (
                df_piv[["caseid", "fold", "y_true", "y_prob"]]
                .sort_values(["fold", "caseid"])
                .reset_index(drop=True)
            )
            if args.save_preds:
                df_ens.to_csv(ens_path, index=False)
                print(f"Saved ensemble OOF predictions: {ens_path}")
            return df_ens, tag

        # Single-run evaluation: prefer using an existing OOF file; otherwise compute from checkpoints.
        tag = ""
        oof_path = paths["results_dir"] / f"{model_name}_oof_predictions.csv"
        if oof_path.exists():
            return pd.read_csv(oof_path), tag

        from vitaldb_aki.evaluation.metrics import evaluate_model
        from vitaldb_aki.training.trainer import load_checkpoint
        import torch

        device = torch.device(
            config.device if torch.cuda.is_available() and config.device == "cuda" else "cpu"
        )

        rows = []
        for f in folds:
            fold_idx = int(f["fold"])
            val_ids = [int(x) for x in f["val_caseids"]]
            model = load_checkpoint(model_name, fold_idx, config)
            y_true, y_prob = evaluate_model(model_name, fold_idx, val_ids, config, model, device)

            if len(val_ids) != int(len(y_true)):
                raise RuntimeError(
                    f"OOF size mismatch: fold {fold_idx} has {len(val_ids)} caseids but {len(y_true)} preds"
                )

            for caseid, yt, yp in zip(val_ids, y_true.tolist(), y_prob.tolist()):
                rows.append(
                    {
                        "caseid": int(caseid),
                        "fold": int(fold_idx),
                        "y_true": int(yt),
                        "y_prob": float(yp),
                    }
                )

        df_oof = pd.DataFrame(rows).sort_values(["fold", "caseid"]).reset_index(drop=True)
        if args.save_preds:
            df_oof.to_csv(oof_path, index=False)
            print(f"Saved OOF predictions: {oof_path}")
        return df_oof, tag

    df_oof, tag = _load_or_build_oof_predictions()
    required_cols = {"caseid", "fold", "y_true", "y_prob"}
    missing = required_cols - set(df_oof.columns)
    if missing:
        raise ValueError(f"OOF predictions missing required columns: {sorted(missing)}")

    # Authoritative scores: compute per-fold metrics from OOF predictions.
    metric_rows = []
    for fold_idx, d in df_oof.groupby("fold"):
        y = d["y_true"].to_numpy(dtype=int)
        p = d["y_prob"].to_numpy(dtype=float)
        metric_rows.append(
            {
                "model": model_name,
                "fold": int(fold_idx),
                "roc_auc": float(roc_auc_score(y, p)),
                "pr_auc": float(average_precision_score(y, p)),
                "seed": (tag or "single"),
            }
        )
    df_metrics = pd.DataFrame(metric_rows).sort_values(["model", "fold"]).reset_index(drop=True)
    suffix = f"_{tag}" if tag else ""
    # Keep CSV artifact names stable (use internal model key).
    metrics_path = paths["results_dir"] / f"{model_name}_5fold_metrics{suffix}.csv"
    df_metrics.to_csv(metrics_path, index=False)
    summary = df_metrics[["roc_auc", "pr_auc"]].agg(["mean", "std"]).to_dict()
    print(f"Saved metrics: {metrics_path}")
    print("OOF summary:", {k: {kk: float(vv) for kk, vv in v.items()} for k, v in summary.items()})

    # Auto-save plots based on OOF predictions.
    if args.save_plots:
        import matplotlib.pyplot as plt

        # Use display name for plot filenames so SynerT shows up in outputs.
        out_stem = model_display
        roc_path = paths["results_dir"] / f"{out_stem}_roc{suffix}.png"
        pr_path = paths["results_dir"] / f"{out_stem}_pr{suffix}.png"
        cm_path = paths["results_dir"] / f"{out_stem}_cm{suffix}.png"

        # ROC curves
        plt.figure(figsize=(6, 5))
        for fold_idx, d in df_oof.groupby("fold"):
            y = d["y_true"].to_numpy(dtype=int)
            p = d["y_prob"].to_numpy(dtype=float)
            fpr, tpr, _ = roc_curve(y, p)
            ra = float(auc(fpr, tpr))
            plt.plot(fpr, tpr, linewidth=2, label=f"Fold {int(fold_idx)} (AUC={ra:.3f})")
        plt.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.6)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC curves | model={model_display}{(' | ' + tag) if tag else ''}")
        plt.grid(True, alpha=0.3)
        plt.legend(loc="lower right")
        plt.tight_layout()
        roc_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(roc_path)
        plt.close()
        print(f"Saved ROC plot: {roc_path}")

        # PR curves
        plt.figure(figsize=(6, 5))
        for fold_idx, d in df_oof.groupby("fold"):
            y = d["y_true"].to_numpy(dtype=int)
            p = d["y_prob"].to_numpy(dtype=float)
            prec, rec, _ = precision_recall_curve(y, p)
            ap = float(average_precision_score(y, p))
            prev = float(y.mean())
            plt.plot(rec, prec, linewidth=2, label=f"Fold {int(fold_idx)} (AP={ap:.3f})")
            plt.hlines(prev, 0, 1, colors="gray", linestyles="--", linewidth=0.8, alpha=0.35)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(
            f"PR curves | model={model_display}{(' | ' + tag) if tag else ''}  (dashed=prevalence)"
        )
        plt.grid(True, alpha=0.3)
        plt.legend(loc="lower left")
        plt.tight_layout()
        pr_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(pr_path)
        plt.close()
        print(f"Saved PR plot: {pr_path}")

        # Confusion matrix
        cm_path.parent.mkdir(parents=True, exist_ok=True)

        # 1) Overall OOF confusion matrix (all folds combined)
        y_true_all = df_oof["y_true"].to_numpy(dtype=int)
        y_prob_all = df_oof["y_prob"].to_numpy(dtype=float)
        y_pred_all = (y_prob_all >= float(args.threshold)).astype(int)
        cm_all = confusion_matrix(y_true_all, y_pred_all, labels=[0, 1])
        disp_all = ConfusionMatrixDisplay(confusion_matrix=cm_all, display_labels=[0, 1])
        fig, ax = plt.subplots(figsize=(5, 5))
        disp_all.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
        ax.set_title(
            f"Confusion matrix (OOF, all folds) | model={model_display}{(' | ' + tag) if tag else ''} | thr={args.threshold}"
        )
        plt.tight_layout()
        plt.savefig(cm_path)
        plt.close(fig)
        print(f"Saved confusion matrix plot: {cm_path}")

        # 2) Per-fold confusion matrices (5 folds -> 5 plots)
        for fold_idx, d in df_oof.groupby("fold"):
            y = d["y_true"].to_numpy(dtype=int)
            p = d["y_prob"].to_numpy(dtype=float)
            y_pred = (p >= float(args.threshold)).astype(int)
            cm = confusion_matrix(y, y_pred, labels=[0, 1])
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
            fig, ax = plt.subplots(figsize=(5, 5))
            disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
            ax.set_title(
                f"Confusion matrix (OOF) | model={model_display}{(' | ' + tag) if tag else ''} | fold={int(fold_idx)} | thr={args.threshold}"
            )
            plt.tight_layout()
            fold_cm_path = paths["results_dir"] / f"{out_stem}_cm_fold{int(fold_idx)}{suffix}.png"
            plt.savefig(fold_cm_path)
            plt.close(fig)
            print(f"Saved fold confusion matrix plot: {fold_cm_path}")

    print("\n=== Evaluation complete ===")


if __name__ == "__main__":
    main()

