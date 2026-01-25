"""PyTorch Dataset classes for training."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

from ..config import Config
from ..data.loaders import (
    load_case_npz_raw,
    load_fold_scalers,
    apply_scalers_to_x,
)
from ..utils.paths import get_paths


class DemoFoldDataset(Dataset):
    """Dataset for a specific fold with normalization."""

    def __init__(
        self,
        caseids: List[int],
        fold_idx: int,
        config: Config,
        *,
        return_caseid: bool = False,
    ):
        """Initialize dataset.

        Args:
            caseids: List of case IDs.
            fold_idx: Fold index (1-based).
            config: Configuration object.
        """
        self.caseids = [int(c) for c in caseids]
        self.fold_idx = int(fold_idx)
        self.config = config
        self.return_caseid = bool(return_caseid)
        self.scalers = load_fold_scalers(self.fold_idx, config)

        paths = get_paths(self.config)
        master_path = paths["artifacts_dir"] / "cohort_master.csv"
        import pandas as pd

        df_master = pd.read_csv(master_path, usecols=["caseid", "aki"])
        df_master["caseid"] = pd.to_numeric(df_master["caseid"], errors="coerce").astype("Int64")
        df_master = df_master.dropna(subset=["caseid"]).copy()
        df_master["caseid"] = df_master["caseid"].astype(int)
        self._label_by_caseid = dict(zip(df_master["caseid"].tolist(), df_master["aki"].astype(int).tolist()))

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.caseids)

    def __getitem__(self, idx: int):
        """Get a single sample.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (x tensor, y label, valid_len).
            
        Raises:
            ValueError: If NPZ file is corrupted.
        """
        caseid = int(self.caseids[idx])
        try:
            x, valid_len = load_case_npz_raw(caseid, self.config)
        except (ValueError, FileNotFoundError) as e:
            # Re-raise with caseid for better error message
            raise ValueError(f"Error loading case {caseid}: {e}")
        
        x = apply_scalers_to_x(x, valid_len, self.scalers, self.config)
        x = x[:, :valid_len].T.astype(np.float32)
        label = int(self._label_by_caseid[caseid])
        y = torch.tensor([label], dtype=torch.float32)
        if self.return_caseid:
            return torch.from_numpy(x), y, int(valid_len), int(caseid)
        return torch.from_numpy(x), y, int(valid_len)


def pad_collate_time_major(batch):
    """Collate function for variable-length sequences.

    Args:
        batch: List of (x, y, length) tuples.

    Returns:
        Tuple of (X padded, y, lengths).
    """
    xs, ys, lens = zip(*batch)
    lengths = torch.tensor(lens, dtype=torch.long)
    max_len = int(lengths.max().item()) if len(lengths) else 0
    feat_dim = int(xs[0].shape[1])
    X = torch.zeros((len(xs), max_len, feat_dim), dtype=torch.float32)
    y = torch.stack([yy.reshape(1) for yy in ys], dim=0).float()
    for i, x in enumerate(xs):
        L = int(x.shape[0])
        X[i, :L] = x
    return X, y, lengths


def pad_collate_time_major_with_caseids(batch):
    """Collate function for variable-length sequences including caseids.

    Args:
        batch: List of (x, y, length, caseid) tuples.

    Returns:
        Tuple of (X padded, y, lengths, caseids).
    """
    xs, ys, lens, caseids = zip(*batch)
    lengths = torch.tensor(lens, dtype=torch.long)
    max_len = int(lengths.max().item()) if len(lengths) else 0
    feat_dim = int(xs[0].shape[1])
    X = torch.zeros((len(xs), max_len, feat_dim), dtype=torch.float32)
    y = torch.stack([yy.reshape(1) for yy in ys], dim=0).float()
    for i, x in enumerate(xs):
        L = int(x.shape[0])
        X[i, :L] = x
    return X, y, lengths, torch.tensor(caseids, dtype=torch.long)

