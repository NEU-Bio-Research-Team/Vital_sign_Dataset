"""Training utilities for AKI prediction models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

from ..config import Config
from ..evaluation.metrics import compute_metrics
from ..models.architectures import build_model
from ..training.dataset import (
    DemoFoldDataset,
    pad_collate_time_major,
    pad_collate_time_major_with_caseids,
)
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
    seed: Optional[int] = None,
    run_tag: Optional[str] = None,
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
    tag = (run_tag or "").strip()
    suffix = f"_{tag}" if tag else ""

    ckpt_dir = paths["models_dir"] / model_name.lower()
    if tag:
        ckpt_dir = ckpt_dir / tag
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
        elif model_name.lower() == "attention":
            # Add attention-specific parameters
            model_kwargs["num_heads"] = getattr(config, "model_num_heads", 4)
            model_kwargs["use_multiscale"] = getattr(config, "model_use_multiscale", True)
        elif model_name.lower() == "transformer":
            # Add transformer-specific parameters
            model_kwargs["num_heads"] = getattr(config, "model_num_heads", 8)
            model_kwargs["max_len"] = getattr(config, "model_max_len", 3600)
        elif model_name.lower() == "dilated_conv":
            # Add dilated_conv-specific parameters
            model_kwargs["use_multiscale"] = getattr(config, "model_use_multiscale", True)
            # Use more layers for dilated_conv (default 10 vs TCN's 6)
            if "num_layers" not in model_kwargs or model_kwargs["num_layers"] == config.model_num_layers:
                model_kwargs["num_layers"] = 10  # Deeper than TCN
        elif model_name.lower() == "dilated_rnn":
            # Add dilated_rnn-specific parameters
            model_kwargs["cell_type"] = getattr(config, "model_cell_type", "lstm")
            model_kwargs["use_attention"] = getattr(config, "model_use_attention", True)
            # Default to 4 layers for dilated_rnn
            if "num_layers" not in model_kwargs:
                model_kwargs["num_layers"] = 4
        elif model_name.lower() == "wavenet":
            # Add wavenet-specific parameters
            model_kwargs["use_attention"] = getattr(config, "model_use_attention", True)
            model_kwargs["num_stacks"] = getattr(config, "model_num_stacks", 3)
            model_kwargs["num_layers_per_stack"] = getattr(config, "model_num_layers_per_stack", 10)
            # Map num_layers to num_layers_per_stack if using default
            if "num_layers" in model_kwargs and "num_layers_per_stack" not in model_kwargs:
                model_kwargs["num_layers_per_stack"] = model_kwargs.pop("num_layers")
        elif model_name.lower() == "temporal_synergy":
            # Add temporal_synergy-specific parameters
            model_kwargs["tcn_levels"] = getattr(config, "model_tcn_levels", 4)
            model_kwargs["rnn_layers"] = getattr(config, "model_rnn_layers", 4)
            model_kwargs["cell_type"] = getattr(config, "model_cell_type", "lstm")
            model_kwargs["use_attention"] = getattr(config, "model_use_attention", True)
            model_kwargs["kernel_size"] = getattr(config, "model_kernel_size", 3)
        elif model_name.lower() == "tcn_attention":
            # Add tcn_attention-specific parameters
            model_kwargs["num_heads"] = getattr(config, "model_num_heads", 4)
            # Use same num_layers as TCN (default 6)
            if "num_layers" not in model_kwargs:
                model_kwargs["num_layers"] = config.model_num_layers
        elif model_name.lower() == "wavenet_rnn":
            # Add wavenet_rnn-specific parameters
            model_kwargs["use_attention"] = getattr(config, "model_use_attention", True)
            model_kwargs["num_stacks"] = getattr(config, "model_num_stacks", 3)
            model_kwargs["num_layers_per_stack"] = getattr(config, "model_num_layers_per_stack", 10)
            model_kwargs["rnn_layers"] = getattr(config, "model_rnn_layers", 2)
            model_kwargs["cell_type"] = getattr(config, "model_cell_type", "lstm")
            # Map num_layers to num_layers_per_stack if using default
            if "num_layers" in model_kwargs and "num_layers_per_stack" not in model_kwargs:
                model_kwargs["num_layers_per_stack"] = model_kwargs.pop("num_layers")
        
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

    def _predict_with_caseids(
        model: nn.Module, val_loader: DataLoader
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict on validation set and return (caseids, y_true, y_prob)."""
        model.eval()
        caseids_all: List[int] = []
        y_true_all: List[float] = []
        y_prob_all: List[float] = []
        with torch.no_grad():
            for Xb, yb, lengths, caseids in val_loader:
                logits = model(Xb.to(device), lengths.to(device))
                prob = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
                y_true = yb.detach().cpu().numpy().reshape(-1)
                caseids_np = caseids.detach().cpu().numpy().reshape(-1)
                caseids_all.extend([int(x) for x in caseids_np.tolist()])
                y_true_all.extend(y_true.tolist())
                y_prob_all.extend(prob.tolist())
        return (
            np.asarray(caseids_all, dtype=int),
            np.asarray(y_true_all, dtype=int),
            np.asarray(y_prob_all, dtype=float),
        )

    def _save_epoch_history_minimal(*, m: Dict[str, float], best_epoch: int, trained: bool) -> None:
        """Save a minimal epoch history row (useful when we loaded from checkpoint)."""
        paths_local = get_paths(config)
        hist_path = paths_local["results_dir"] / f"{model_name}_fold{fold_idx}_epoch_history{suffix}.csv"
        row = {
            "epoch": int(best_epoch) if best_epoch is not None else -1,
            "train_loss_mean": float("nan"),
            "val_roc_auc": float(m.get("roc_auc", float("nan"))),
            "val_pr_auc": float(m.get("pr_auc", float("nan"))),
            "monitor": float(m.get(config.monitor, float("nan"))),
            "best_so_far": float(m.get(config.monitor, float("nan"))),
            "trained": bool(trained),
            "note": "loaded_from_checkpoint" if not trained else "trained_this_run",
        }
        pd.DataFrame([row]).to_csv(hist_path, index=False)
        print(f"  Saved epoch history: {hist_path}")

    def _save_val_predictions(model: nn.Module) -> None:
        """Save per-fold validation predictions (caseid, fold, y_true, y_prob)."""
        paths_local = get_paths(config)
        val_ds_with_ids = DemoFoldDataset(
            val_caseids,
            fold_idx,
            config,
            return_caseid=True,
        )
        # Use same batch sizing rule as above
        eff_bs = config.batch_size
        if model_name.lower() in ("attention", "transformer", "tcn_attention"):
            eff_bs = max(8, config.batch_size // 2)

        val_loader_with_ids = DataLoader(
            val_ds_with_ids,
            batch_size=eff_bs,
            shuffle=False,
            num_workers=0,
            collate_fn=pad_collate_time_major_with_caseids,
        )
        caseids_np, y_true_np, y_prob_np = _predict_with_caseids(model, val_loader_with_ids)
        df_fold = pd.DataFrame(
            {
                "caseid": caseids_np.astype(int),
                "fold": int(fold_idx),
                "y_true": y_true_np.astype(int),
                "y_prob": y_prob_np.astype(float),
            }
        ).sort_values(["fold", "caseid"]).reset_index(drop=True)
        fold_pred_path = paths_local["results_dir"] / f"{model_name}_fold{fold_idx}_val_predictions{suffix}.csv"
        df_fold.to_csv(fold_pred_path, index=False)
        print(f"  Saved fold val predictions: {fold_pred_path}")

    # Check if checkpoint exists
    if ckpt_path.exists() and not force_train:
        # NOTE: keep your existing torch.load logic
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)

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

            effective_batch_size = config.batch_size
            if model_name.lower() in ("attention", "transformer", "tcn_attention"):
                effective_batch_size = max(8, config.batch_size // 2)

            val_loader = DataLoader(
                val_ds,
                batch_size=effective_batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=pad_collate_time_major,
            )
            m = _eval(model, val_loader)
            roc_auc = float(m["roc_auc"])
            pr_auc = float(m["pr_auc"])

            # NEW: save preds + minimal epoch history even when loaded
            try:
                _save_val_predictions(model)
            except Exception as e:
                print(f"  Warning: failed to save val predictions for {model_name} fold {fold_idx}: {e}")

            try:
                _save_epoch_history_minimal(m=m, best_epoch=int(ckpt.get('best_epoch', -1)), trained=False)
            except Exception as e:
                print(f"  Warning: failed to save epoch history for {model_name} fold {fold_idx}: {e}")

            return FoldResult(
                model=model_name,
                fold=int(fold_idx),
                roc_auc=roc_auc,
                pr_auc=pr_auc,
                ckpt_path=str(ckpt_path),
                trained=False,
                best_epoch=int(ckpt.get("best_epoch", -1)),
            )

    # Training
    # Validate that we have data
    if not train_caseids or len(train_caseids) == 0:
        exp_name = Path(config.artifacts_dir).name
        raise ValueError(
            f"Empty training set for {model_name} fold {fold_idx}. "
            f"Please regenerate folds by running: "
            f"python scripts/preprocess.py --experiment-name {exp_name} --force"
        )
    if not val_caseids or len(val_caseids) == 0:
        exp_name = Path(config.artifacts_dir).name
        raise ValueError(
            f"Empty validation set for {model_name} fold {fold_idx}. "
            f"Please regenerate folds by running: "
            f"python scripts/preprocess.py --experiment-name {exp_name} --force"
        )
    
    train_ds = DemoFoldDataset(train_caseids, fold_idx, config)
    val_ds = DemoFoldDataset(val_caseids, fold_idx, config)

    # Use smaller batch size for memory-intensive models (attention/transformer/tcn_attention)
    effective_batch_size = config.batch_size
    if model_name.lower() in ("attention", "transformer", "tcn_attention"):
        effective_batch_size = max(8, config.batch_size // 2)  # Half batch size, minimum 8
        print(f"  Using reduced batch size {effective_batch_size} for {model_name} (memory optimization)")

    train_loader = DataLoader(
        train_ds,
        batch_size=effective_batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=pad_collate_time_major,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=effective_batch_size,
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

    # Per-epoch history logging
    epoch_rows: List[Dict[str, float]] = []

    best = -float("inf")
    best_epoch = -1
    bad = 0
    for ep in range(1, int(config.epochs) + 1):
        model.train()
        train_losses: List[float] = []
        for Xb, yb, lengths in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(Xb.to(device), lengths.to(device))
            loss = criterion(logits, yb.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu().item()))

        m = _eval(model, val_loader)
        score = float(m.get(config.monitor, float("nan")))
        if np.isnan(score):
            score = -float("inf")

        epoch_rows.append(
            {
                "epoch": int(ep),
                "train_loss_mean": float(np.mean(train_losses)) if train_losses else float("nan"),
                "val_roc_auc": float(m.get("roc_auc", float("nan"))),
                "val_pr_auc": float(m.get("pr_auc", float("nan"))),
                "monitor": float(m.get(config.monitor, float("nan"))),
                "best_so_far": float(best if best != -float("inf") else float("nan")),
            }
        )

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

    # Save per-epoch history
    try:
        paths = get_paths(config)
        hist_path = paths["results_dir"] / f"{model_name}_fold{fold_idx}_epoch_history{suffix}.csv"
        pd.DataFrame(epoch_rows).to_csv(hist_path, index=False)
        print(f"  Saved epoch history: {hist_path}")
    except Exception as e:
        print(f"  Warning: failed to save epoch history for {model_name} fold {fold_idx}: {e}")

    # Load best checkpoint
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"], strict=True)
    m = _eval(model, val_loader)

    # Save per-fold validation predictions (for OOF aggregation / CI)
    try:
        paths = get_paths(config)
        val_ds_with_ids = DemoFoldDataset(
            val_caseids,
            fold_idx,
            config,
            return_caseid=True,
        )
        val_loader_with_ids = DataLoader(
            val_ds_with_ids,
            batch_size=effective_batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=pad_collate_time_major_with_caseids,
        )
        caseids_np, y_true_np, y_prob_np = _predict_with_caseids(model, val_loader_with_ids)
        df_fold = pd.DataFrame(
            {
                "caseid": caseids_np.astype(int),
                "fold": int(fold_idx),
                "y_true": y_true_np.astype(int),
                "y_prob": y_prob_np.astype(float),
            }
        ).sort_values(["fold", "caseid"]).reset_index(drop=True)
        fold_pred_path = paths["results_dir"] / f"{model_name}_fold{fold_idx}_val_predictions{suffix}.csv"
        df_fold.to_csv(fold_pred_path, index=False)
        print(f"  Saved fold val predictions: {fold_pred_path}")
    except Exception as e:
        print(f"  Warning: failed to save val predictions for {model_name} fold {fold_idx}: {e}")

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
    model_name: str,
    folds: List[Dict],
    config: Config,
    *,
    force_train: bool = False,
    seed: Optional[int] = None,
    run_tag: Optional[str] = None,
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
    set_torch_seed(int(config.random_state if seed is None else seed))

    tag = (run_tag or "").strip()
    suffix = f"_{tag}" if tag else ""

    rows = []
    oof_rows: List[Dict[str, float]] = []
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
            seed=seed,
            run_tag=run_tag,
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

        # If fold prediction file exists (written by train_or_load_one_fold), collect it for OOF.
        try:
            paths = get_paths(config)
            fold_pred_path = paths["results_dir"] / f"{model_name}_fold{fold_idx}_val_predictions{suffix}.csv"
            if fold_pred_path.exists():
                df_fold = pd.read_csv(fold_pred_path)
                oof_rows.extend(df_fold.to_dict(orient="records"))
        except Exception:
            # Non-fatal; training metrics are still valid.
            pass
        print(
            f"[{model_name}] Fold {fold_idx}: ROC={res.roc_auc:.4f} PR={res.pr_auc:.4f} | "
            f"trained={res.trained} | best_epoch={res.best_epoch}"
        )

    df = pd.DataFrame(rows).sort_values(["model", "fold"]).reset_index(drop=True)
    if seed is not None:
        # Keep seed explicit so downstream statistical tests can pair on (seed, fold).
        df["seed"] = int(seed)
    paths = get_paths(config)
    out_csv = paths["results_dir"] / f"{model_name}_5fold_metrics{suffix}.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    # Save aggregated OOF predictions if available
    if oof_rows:
        df_oof = pd.DataFrame(oof_rows)
        # tolerate missing columns if user edited files
        keep = [c for c in ["caseid", "fold", "y_true", "y_prob"] if c in df_oof.columns]
        df_oof = df_oof[keep].copy() if keep else df_oof.copy()
        if "fold" in df_oof.columns and "caseid" in df_oof.columns:
            df_oof = df_oof.sort_values(["fold", "caseid"]).reset_index(drop=True)
        oof_path = paths["results_dir"] / f"{model_name}_oof_predictions{suffix}.csv"
        df_oof.to_csv(oof_path, index=False)
        print(f"Saved OOF predictions: {oof_path}")

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
    elif model_name.lower() == "attention":
        # Add attention-specific parameters
        model_kwargs["num_heads"] = getattr(config, "model_num_heads", 4)
        model_kwargs["use_multiscale"] = getattr(config, "model_use_multiscale", True)
    elif model_name.lower() == "transformer":
        # Add transformer-specific parameters
        model_kwargs["num_heads"] = getattr(config, "model_num_heads", 8)
        model_kwargs["max_len"] = getattr(config, "model_max_len", 3600)
    elif model_name.lower() == "dilated_conv":
        # Add dilated_conv-specific parameters
        model_kwargs["use_multiscale"] = getattr(config, "model_use_multiscale", True)
        # Use more layers for dilated_conv (default 10 vs TCN's 6)
        if "num_layers" not in model_kwargs or model_kwargs["num_layers"] == config.model_num_layers:
            model_kwargs["num_layers"] = 10  # Deeper than TCN
    elif model_name.lower() == "dilated_rnn":
        # Add dilated_rnn-specific parameters
        model_kwargs["cell_type"] = getattr(config, "model_cell_type", "lstm")
        model_kwargs["use_attention"] = getattr(config, "model_use_attention", True)
        # Default to 4 layers for dilated_rnn
        if "num_layers" not in model_kwargs:
            model_kwargs["num_layers"] = 4
    elif model_name.lower() == "wavenet":
        # Add wavenet-specific parameters
        model_kwargs["use_attention"] = getattr(config, "model_use_attention", True)
        model_kwargs["num_stacks"] = getattr(config, "model_num_stacks", 3)
        model_kwargs["num_layers_per_stack"] = getattr(config, "model_num_layers_per_stack", 10)
        # Map num_layers to num_layers_per_stack if using default
        if "num_layers" in model_kwargs and "num_layers_per_stack" not in model_kwargs:
            model_kwargs["num_layers_per_stack"] = model_kwargs.pop("num_layers")
    elif model_name.lower() == "temporal_synergy":
        # Add temporal_synergy-specific parameters
        model_kwargs["tcn_levels"] = getattr(config, "model_tcn_levels", 4)
        model_kwargs["rnn_layers"] = getattr(config, "model_rnn_layers", 4)
        model_kwargs["cell_type"] = getattr(config, "model_cell_type", "lstm")
        model_kwargs["use_attention"] = getattr(config, "model_use_attention", True)
        model_kwargs["kernel_size"] = getattr(config, "model_kernel_size", 3)
    elif model_name.lower() == "tcn_attention":
        # Add tcn_attention-specific parameters
        model_kwargs["num_heads"] = getattr(config, "model_num_heads", 4)
        # Use same num_layers as TCN (default 6)
        if "num_layers" not in model_kwargs:
            model_kwargs["num_layers"] = config.model_num_layers
    elif model_name.lower() == "wavenet_rnn":
        # Add wavenet_rnn-specific parameters
        model_kwargs["use_attention"] = getattr(config, "model_use_attention", True)
        model_kwargs["num_stacks"] = getattr(config, "model_num_stacks", 3)
        model_kwargs["num_layers_per_stack"] = getattr(config, "model_num_layers_per_stack", 10)
        model_kwargs["rnn_layers"] = getattr(config, "model_rnn_layers", 2)
        model_kwargs["cell_type"] = getattr(config, "model_cell_type", "lstm")
        # Map num_layers to num_layers_per_stack if using default
        if "num_layers" in model_kwargs and "num_layers_per_stack" not in model_kwargs:
            model_kwargs["num_layers_per_stack"] = model_kwargs.pop("num_layers")
    
    model = build_model(model_name, input_dim, **model_kwargs).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
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
                f"python scripts/train.py --experiment-name {Path(config.artifacts_dir).name} --model {model_name} --force"
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

