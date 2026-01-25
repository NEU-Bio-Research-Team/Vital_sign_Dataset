"""Helper utility functions."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import torch


def save_json(path: Path, obj) -> None:
    """Save object to JSON file.

    Args:
        path: Path to save JSON file.
        obj: Object to serialize (must be JSON serializable).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def safe_lower_series(s: pd.Series) -> pd.Series:
    """Convert series to lowercase strings safely.

    Args:
        s: Pandas Series.

    Returns:
        Series with lowercase string values.
    """
    return s.astype(str).str.lower()


def ensure_cols(df: pd.DataFrame, cols: Iterable[str], df_name: str) -> None:
    """Ensure DataFrame has required columns.

    Args:
        df: DataFrame to check.
        cols: Required column names.
        df_name: Name of DataFrame for error messages.

    Raises:
        KeyError: If any required columns are missing.
    """
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"{df_name} missing columns: {missing}. "
            f"Available: {list(df.columns)[:50]}"
        )


def infer_time_value_cols(df: pd.DataFrame) -> Tuple[str, str]:
    """Infer time/value columns from a track dataframe.

    Args:
        df: DataFrame with track data.

    Returns:
        Tuple of (time_column, value_column).

    Raises:
        ValueError: If columns cannot be inferred.
    """
    cols = [c.lower() for c in df.columns]
    time_candidates = ["t", "time", "dt", "sec", "seconds"]
    value_candidates = ["v", "value", "val", "y"]
    t_col = next(
        (df.columns[i] for i, c in enumerate(cols) if c in time_candidates), None
    )
    v_col = next(
        (df.columns[i] for i, c in enumerate(cols) if c in value_candidates), None
    )
    if t_col is None or v_col is None:
        # fallback: 1st two columns
        if df.shape[1] >= 2:
            return df.columns[0], df.columns[1]
        raise ValueError(
            f"Cannot infer time/value cols from columns: {list(df.columns)}"
        )
    return t_col, v_col


def transform_signal(name: str, x: np.ndarray) -> np.ndarray:
    """Transform observed signal values (clip/log).

    Args:
        name: Signal name.
        x: 1D float32 array of signal values.

    Returns:
        Transformed signal array.
    """
    if name == "ART_MBP":
        return np.clip(x, 0.0, 200.0)
    if name == "PLETH_HR":
        return np.clip(x, 0.0, 250.0)
    if name == "ART_SBP":
        return np.clip(x, 0.0, 300.0)
    if name == "ART_DBP":
        return np.clip(x, 0.0, 200.0)
    if name == "HR":
        return np.clip(x, 0.0, 250.0)
    if name == "PLETH_SPO2":
        return np.clip(x, 0.0, 100.0)
    if name == "ETCO2":
        return np.clip(x, 0.0, 100.0)
    return x


def set_torch_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def effective_grid_start_end(cutoff_mode: str, max_len_sec: int, t_cut_sec: int, preop_window_sec: int) -> Tuple[float, float]:
    """Calculate effective grid start and end times.

    Args:
        cutoff_mode: "early_intraop" or "preop".
        max_len_sec: Maximum length in seconds.
        t_cut_sec: Cutoff time for early_intraop.
        preop_window_sec: Preop window for preop mode.

    Returns:
        Tuple of (start_sec, end_sec).
    """
    if cutoff_mode == "preop":
        start = -float(preop_window_sec)
        end = 0.0
        return start, end
    # early_intraop: only need up to t_cut_sec
    start = 0.0
    end = float(min(max_len_sec, t_cut_sec))
    return start, end


def grid_seconds(fs_hz: float, cutoff_mode: str, max_len_sec: int, t_cut_sec: int, preop_window_sec: int) -> np.ndarray:
    """Generate time grid for resampling.

    Args:
        fs_hz: Sampling frequency in Hz.
        cutoff_mode: "early_intraop" or "preop".
        max_len_sec: Maximum length in seconds.
        t_cut_sec: Cutoff time for early_intraop.
        preop_window_sec: Preop window for preop mode.

    Returns:
        Time grid array.
    """
    dt = 1.0 / float(fs_hz)
    start, end = effective_grid_start_end(cutoff_mode, max_len_sec, t_cut_sec, preop_window_sec)
    # end is exclusive in arange; include endpoint-ish by adding dt
    return np.arange(start, end + 1e-6, dt, dtype=np.float32)

