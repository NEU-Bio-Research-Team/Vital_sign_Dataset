"""Evaluation metrics for AKI prediction."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Compute classification metrics.

    Args:
        y_true: True labels.
        y_prob: Predicted probabilities.

    Returns:
        Dictionary with metrics.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    out: Dict[str, float] = {}
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        out["roc_auc"] = float("nan")
    try:
        out["pr_auc"] = float(average_precision_score(y_true, y_prob))
    except Exception:
        out["pr_auc"] = float("nan")
    return out


def evaluate_model(
    model_name: str,
    fold_idx: int,
    val_caseids: list,
    config,
    model,
    device,
) -> tuple:
    """Evaluate a model on validation set.

    Args:
        model_name: Model name.
        fold_idx: Fold index.
        val_caseids: Validation case IDs.
        config: Configuration object.
        model: Trained model.
        device: Device to run on.

    Returns:
        Tuple of (y_true, y_prob).
    """
    from torch.utils.data import DataLoader
    from ..training.dataset import DemoFoldDataset, pad_collate_time_major

    val_ds = DemoFoldDataset(val_caseids, fold_idx, config)
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=pad_collate_time_major,
    )

    y_true_all, y_prob_all = [], []
    model.eval()
    with torch.no_grad():
        for Xb, yb, lengths in val_loader:
            logits = model(Xb.to(device), lengths.to(device))
            prob = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
            y_true = yb.detach().cpu().numpy().reshape(-1)
            y_true_all.extend(y_true.tolist())
            y_prob_all.extend(prob.tolist())

    return np.asarray(y_true_all, dtype=int), np.asarray(y_prob_all, dtype=float)

