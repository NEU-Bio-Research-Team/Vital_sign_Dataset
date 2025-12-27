"""Visualization functions for evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    roc_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from ..config import Config
from ..evaluation.metrics import evaluate_model
from ..utils.paths import get_paths
import torch


def plot_roc_curves(
    model_name: str,
    folds: List[dict],
    config: Config,
    save_path: Optional[Path] = None,
) -> None:
    """Plot ROC curves for all folds.

    Args:
        model_name: Model name.
        folds: List of fold dictionaries.
        config: Configuration object.
        save_path: Path to save plot. If None, displays plot.
    """
    from ..training.trainer import load_checkpoint
    
    device = torch.device(
        config.device if torch.cuda.is_available() and config.device == "cuda" else "cpu"
    )

    roc_lines = []
    for f in folds:
        fold_idx = int(f["fold"])
        val_ids = [int(x) for x in f["val_caseids"]]

        model = load_checkpoint(model_name, fold_idx, config)
        y_true, y_prob = evaluate_model(
            model_name, fold_idx, val_ids, config, model, device
        )

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = float(auc(fpr, tpr))
        roc_lines.append((fold_idx, fpr, tpr, roc_auc))

    plt.figure(figsize=(6, 5))
    for fold_idx, fpr, tpr, roc_auc in sorted(roc_lines, key=lambda x: x[0]):
        plt.plot(fpr, tpr, linewidth=2, label=f"Fold {fold_idx} (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.6)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC curves | model={model_name}")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_pr_curves(
    model_name: str,
    folds: List[dict],
    config: Config,
    save_path: Optional[Path] = None,
) -> None:
    """Plot PR curves for all folds.

    Args:
        model_name: Model name.
        folds: List of fold dictionaries.
        config: Configuration object.
        save_path: Path to save plot. If None, displays plot.
    """
    from ..training.trainer import load_checkpoint
    
    device = torch.device(
        config.device if torch.cuda.is_available() and config.device == "cuda" else "cpu"
    )

    pr_lines = []
    for f in folds:
        fold_idx = int(f["fold"])
        val_ids = [int(x) for x in f["val_caseids"]]

        model = load_checkpoint(model_name, fold_idx, config)
        y_true, y_prob = evaluate_model(
            model_name, fold_idx, val_ids, config, model, device
        )

        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        ap = float(average_precision_score(y_true, y_prob))
        prev = float(y_true.mean())
        pr_lines.append((fold_idx, rec, prec, ap, prev))

    plt.figure(figsize=(6, 5))
    for fold_idx, rec, prec, ap, prev in sorted(pr_lines, key=lambda x: x[0]):
        plt.plot(rec, prec, linewidth=2, label=f"Fold {fold_idx} (AP={ap:.3f})")
        plt.hlines(prev, 0, 1, colors="gray", linestyles="--", linewidth=0.8, alpha=0.35)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR curves | model={model_name}  (dashed=prevalence)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower left")
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_confusion_matrix(
    model_name: str,
    folds: List[dict],
    config: Config,
    threshold: float = 0.5,
    save_path: Optional[Path] = None,
) -> dict:
    """Plot confusion matrix for out-of-fold predictions.

    Args:
        model_name: Model name.
        folds: List of fold dictionaries.
        config: Configuration object.
        threshold: Classification threshold.
        save_path: Path to save plot. If None, displays plot.

    Returns:
        Dictionary with metrics.
    """
    from ..training.trainer import load_checkpoint
    
    device = torch.device(
        config.device if torch.cuda.is_available() and config.device == "cuda" else "cpu"
    )

    y_true_all, y_prob_all = [], []
    for f in folds:
        fold_idx = int(f["fold"])
        val_ids = [int(x) for x in f["val_caseids"]]

        model = load_checkpoint(model_name, fold_idx, config)
        y_true, y_prob = evaluate_model(
            model_name, fold_idx, val_ids, config, model, device
        )
        y_true_all.extend(y_true.tolist())
        y_prob_all.extend(y_prob.tolist())

    y_true_all = np.asarray(y_true_all, dtype=int)
    y_prob_all = np.asarray(y_prob_all, dtype=float)
    y_pred = (y_prob_all >= float(threshold)).astype(int)

    cm = confusion_matrix(y_true_all, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title(f"Confusion matrix (OOF) | model={model_name} | thr={threshold}")
    plt.tight_layout()

    metrics = {
        "acc": float(accuracy_score(y_true_all, y_pred)),
        "precision": float(precision_score(y_true_all, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true_all, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true_all, y_pred, zero_division=0)),
    }

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

    print("OOF metrics @ threshold:", metrics)
    print(
        "Counts:",
        {
            "n": int(len(y_true_all)),
            "pos": int(y_true_all.sum()),
            "neg": int((1 - y_true_all).sum()),
        },
    )

    return metrics

