"""Training utilities for AKI prediction models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

from ..config import Config
from ..evaluation.metrics import compute_metrics
from ..models.architectures import build_model
from ..training.dataset import DemoFoldDataset, pad_collate_time_major
from ..utils.helpers import set_torch_seed
from ..utils.paths import get_paths


@dataclass
class FoldResult:
    """Result from training a single fold."""

    model: str
    fold: int
    roc_auc: float
    pr_auc: float
    ckpt_path: str
    trained: bool
    best_epoch: int


def train_or_load_one_fold(
    *,
    model_name: str,
    fold_idx: int,
    train_caseids: List[int],
    val_caseids: List[int],
    config: Config,
    build_model_fn: Optional[Callable[[], nn.Module]] = None,
    force_train: bool = False,
) -> FoldResult:
    """Train or load a model for a single fold.

    Args:
        model_name: Model name.
        fold_idx: Fold index (1-based).
        train_caseids: Training case IDs.
        val_caseids: Validation case IDs.
        config: Configuration object.
        build_model_fn: Function to build model. If None, uses build_model.
        force_train: If True, retrain even if checkpoint exists.

    Returns:
        FoldResult object.
    """
    paths = get_paths(config)
    ckpt_dir = paths["models_dir"] / model_name.lower()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"fold{fold_idx}.pt"

    device = torch.device(config.device if torch.cuda.is_available() and config.device == "cuda" else "cpu")

    if build_model_fn is None:
        signal_names = list(config.signals)
        input_dim = len(signal_names) * 2  # signals + masks
        
        # For TCN, use default levels=6 for backward compatibility with old checkpoints
        model_kwargs = {
            "hidden_dim": config.model_hidden_dim,
            "num_layers": config.model_num_layers,
            "dropout": config.model_dropout,
        }
        if model_name.lower() == "tcn":
            # TCN default is levels=6, don't override unless needed
            model_kwargs.pop("num_layers", None)
        
        build_model_fn = lambda: build_model(model_name, input_dim, **model_kwargs)

    model = build_model_fn().to(device)

    def _eval(model: nn.Module, val_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model on validation set."""
        model.eval()
        y_true_all: List[float] = []
        y_prob_all: List[float] = []
        with torch.no_grad():
            for Xb, yb, lengths in val_loader:
                logits = model(Xb.to(device), lengths.to(device))
                prob = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
                y_true = yb.detach().cpu().numpy().reshape(-1)
                y_true_all.extend(y_true.tolist())
                y_prob_all.extend(prob.tolist())
        return compute_metrics(np.asarray(y_true_all, dtype=int), np.asarray(y_prob_all, dtype=float))

    # Check if checkpoint exists
    if ckpt_path.exists() and not force_train:
        ckpt = torch.load(ckpt_path, map_location=device)
        # Try to load checkpoint, handling architecture mismatches
        try:
            model.load_state_dict(ckpt["model_state"], strict=True)
            checkpoint_loaded = True
        except RuntimeError as e:
            error_msg = str(e).lower()
            # Check if it's a size mismatch (can't be fixed with strict=False)
            if "size mismatch" in error_msg or "shape" in error_msg:
                print(
                    f"  Warning: Checkpoint architecture mismatch for {model_name} fold {fold_idx}. "
                    f"Deleting incompatible checkpoint and retraining..."
                )
                ckpt_path.unlink()  # Delete incompatible checkpoint
                checkpoint_loaded = False
            else:
                # Try non-strict loading for missing keys only
                try:
                    model.load_state_dict(ckpt["model_state"], strict=False)
                    checkpoint_loaded = True
                except RuntimeError:
                    print(
                        f"  Warning: Failed to load checkpoint for {model_name} fold {fold_idx}. "
                        f"Retraining..."
                    )
                    checkpoint_loaded = False
        
        if checkpoint_loaded:
            train_ds = DemoFoldDataset(train_caseids, fold_idx, config)
            val_ds = DemoFoldDataset(val_caseids, fold_idx, config)
            val_loader = DataLoader(
                val_ds,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=pad_collate_time_major,
            )
            m = _eval(model, val_loader)
            roc_auc = float(m["roc_auc"])
            pr_auc = float(m["pr_auc"])
            # Warn if metrics are NaN (might indicate invalid checkpoint or evaluation issue)
            if np.isnan(roc_auc) or np.isnan(pr_auc):
                print(
                    f"  Warning: NaN metrics detected for {model_name} fold {fold_idx}. "
                    f"This may indicate an invalid checkpoint. Consider retraining with --force."
                )
            return FoldResult(
                model=model_name,
                fold=int(fold_idx),
                roc_auc=roc_auc,
                pr_auc=pr_auc,
                ckpt_path=str(ckpt_path),
                trained=False,
                best_epoch=int(ckpt.get("best_epoch", -1)),
            )
        # If checkpoint loading failed, fall through to training

    # Training
    # Validate that we have data
    if not train_caseids or len(train_caseids) == 0:
        raise ValueError(
            f"Empty training set for {model_name} fold {fold_idx}. "
            f"Please regenerate folds by running: "
            f"python scripts/preprocess.py --experiment-name {config.artifacts_dir.split('/')[-1]} --force"
        )
    if not val_caseids or len(val_caseids) == 0:
        raise ValueError(
            f"Empty validation set for {model_name} fold {fold_idx}. "
            f"Please regenerate folds by running: "
            f"python scripts/preprocess.py --experiment-name {config.artifacts_dir.split('/')[-1]} --force"
        )
    
    train_ds = DemoFoldDataset(train_caseids, fold_idx, config)
    val_ds = DemoFoldDataset(val_caseids, fold_idx, config)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=pad_collate_time_major,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=pad_collate_time_major,
    )

    # Compute pos_weight
    y_train_all: List[float] = []
    for _, yb, _ in train_loader:
        y_train_all.extend(yb.detach().cpu().numpy().reshape(-1).tolist())
    y_train = np.asarray(y_train_all, dtype=float)
    n_pos = float(np.sum(y_train > 0.5))
    n_neg = float(np.sum(y_train <= 0.5))
    pos_weight = n_neg / max(n_pos, 1.0)

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    best = -float("inf")
    best_epoch = -1
    bad = 0
    for ep in range(1, int(config.epochs) + 1):
        model.train()
        for Xb, yb, lengths in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(Xb.to(device), lengths.to(device))
            loss = criterion(logits, yb.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        m = _eval(model, val_loader)
        score = float(m.get(config.monitor, float("nan")))
        if np.isnan(score):
            score = -float("inf")

        if score > best:
            best = score
            best_epoch = ep
            bad = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "best_epoch": best_epoch,
                    "monitor": config.monitor,
                    "best_score": best,
                    "pos_weight": pos_weight,
                },
                ckpt_path,
            )
        else:
            bad += 1
            if bad >= int(config.patience):
                break

    # Load best checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    m = _eval(model, val_loader)
    return FoldResult(
        model=model_name,
        fold=int(fold_idx),
        roc_auc=float(m["roc_auc"]),
        pr_auc=float(m["pr_auc"]),
        ckpt_path=str(ckpt_path),
        trained=True,
        best_epoch=int(ckpt.get("best_epoch", -1)),
    )


def train_all_folds(
    model_name: str, folds: List[Dict], config: Config, *, force_train: bool = False
) -> pd.DataFrame:
    """Train model on all folds.

    Args:
        model_name: Model name.
        folds: List of fold dictionaries.
        config: Configuration object.
        force_train: If True, retrain even if checkpoints exist.

    Returns:
        DataFrame with metrics per fold.
    """
    set_torch_seed(config.random_state)

    rows = []
    for f in folds:
        fold_idx = int(f["fold"])
        train_ids = [int(x) for x in f["train_caseids"]]
        val_ids = [int(x) for x in f["val_caseids"]]

        res = train_or_load_one_fold(
            model_name=model_name,
            fold_idx=fold_idx,
            train_caseids=train_ids,
            val_caseids=val_ids,
            config=config,
            force_train=force_train,
        )
        rows.append(
            {
                "model": res.model,
                "fold": res.fold,
                "roc_auc": res.roc_auc,
                "pr_auc": res.pr_auc,
                "trained": res.trained,
                "best_epoch": res.best_epoch,
                "ckpt_path": res.ckpt_path,
            }
        )
        print(
            f"[{model_name}] Fold {fold_idx}: ROC={res.roc_auc:.4f} PR={res.pr_auc:.4f} | "
            f"trained={res.trained} | best_epoch={res.best_epoch}"
        )

    df = pd.DataFrame(rows).sort_values(["model", "fold"]).reset_index(drop=True)
    paths = get_paths(config)
    out_csv = paths["results_dir"] / f"{model_name}_5fold_metrics.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")
    return df


def load_checkpoint(model_name: str, fold_idx: int, config: Config) -> nn.Module:
    """Load a trained model checkpoint.

    Args:
        model_name: Model name.
        fold_idx: Fold index (1-based).
        config: Configuration object.

    Returns:
        Loaded model.
    """
    paths = get_paths(config)
    ckpt_path = paths["models_dir"] / model_name.lower() / f"fold{fold_idx}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = torch.device(config.device if torch.cuda.is_available() and config.device == "cuda" else "cpu")
    signal_names = list(config.signals)
    input_dim = len(signal_names) * 2
    
    # For TCN, use default levels=6 for backward compatibility with old checkpoints
    model_kwargs = {
        "hidden_dim": config.model_hidden_dim,
        "num_layers": config.model_num_layers,
        "dropout": config.model_dropout,
    }
    if model_name.lower() == "tcn":
        # TCN default is levels=6, don't override for old checkpoints
        model_kwargs.pop("num_layers", None)
    
    model = build_model(model_name, input_dim, **model_kwargs).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    # Try to load checkpoint, handling architecture mismatches
    try:
        model.load_state_dict(ckpt["model_state"], strict=True)
    except RuntimeError as e:
        error_msg = str(e).lower()
        # Check if it's a size mismatch (can't be fixed with strict=False)
        if "size mismatch" in error_msg or "shape" in error_msg:
            raise RuntimeError(
                f"Checkpoint architecture mismatch for {model_name} fold {fold_idx}. "
                f"The checkpoint was saved with a different model architecture than the current config. "
                f"Please retrain the model by running: "
                f"python scripts/train.py --experiment-name {config.artifacts_dir.split('/')[-1]} --model {model_name} --force"
            ) from e
        else:
            # Try non-strict loading for missing keys only
            try:
                model.load_state_dict(ckpt["model_state"], strict=False)
            except RuntimeError as e2:
                raise RuntimeError(
                    f"Failed to load checkpoint for {model_name} fold {fold_idx}. "
                    f"Error: {e2}"
                ) from e2
    return model

