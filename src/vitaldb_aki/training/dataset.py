"""PyTorch Dataset classes for training."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

from ..config import Config
from ..data.loaders import load_case_npz_raw, load_fold_scalers, apply_scalers_to_x


class DemoFoldDataset(Dataset):
    """Dataset for a specific fold with normalization."""

    def __init__(self, caseids: List[int], fold_idx: int, config: Config):
        """Initialize dataset.

        Args:
            caseids: List of case IDs.
            fold_idx: Fold index (1-based).
            config: Configuration object.
        """
        self.caseids = [int(c) for c in caseids]
        self.fold_idx = int(fold_idx)
        self.config = config
        self.scalers = load_fold_scalers(self.fold_idx, config)

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
        # Load label from master file
        from pathlib import Path
        import pandas as pd

        artifacts_dir = Path(self.config.artifacts_dir)
        master_path = artifacts_dir / "cohort_master.csv"
        df_master = pd.read_csv(master_path)
        label = int(df_master[df_master["caseid"] == caseid]["aki"].iloc[0])
        y = torch.tensor([label], dtype=torch.float32)
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

