"""Data loading utilities for VitalDB API and cached data."""

from __future__ import annotations

import io
import json
import os
import time
import threading
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ..config import Config
from ..utils.helpers import infer_time_value_cols, transform_signal
from ..utils.paths import get_paths


# Thread-local storage for HTTP sessions
_thread_local = threading.local()


def _get_session() -> requests.Session:
    """Get or create thread-local HTTP session with retry logic."""
    s = getattr(_thread_local, "session", None)
    if s is None:
        s = requests.Session()
        retry = Retry(
            total=5,
            connect=5,
            read=5,
            backoff_factor=0.8,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET",),
            raise_on_status=False,
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=16, pool_maxsize=16)
        s.mount("http://", adapter)
        s.mount("https://", adapter)
        _thread_local.session = s
    return s


def fetch_track_df(tid: str, api_base: str) -> pd.DataFrame:
    """Fetch a track by tid from VitalDB API with timeouts + retries.

    Args:
        tid: Track ID.
        api_base: Base URL for VitalDB API.

    Returns:
        DataFrame with track data.

    Raises:
        RuntimeError: If HTTP request fails.
    """
    url = f"{api_base}/{str(tid)}"
    sess = _get_session()
    r = sess.get(url, timeout=(5, 30))
    # If still rate-limited after retries, sleep briefly and try once more
    if r.status_code == 429:
        retry_after = r.headers.get("Retry-After")
        try:
            wait = float(retry_after) if retry_after is not None else 5.0
        except Exception:
            wait = 5.0
        time.sleep(min(max(wait, 1.0), 30.0))
        r = sess.get(url, timeout=(5, 30))
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code} fetching tid={tid}")
    return pd.read_csv(io.StringIO(r.text))


def load_track_df_cached(tid: str, config: Config, *, force: bool = False) -> pd.DataFrame:
    """Fetch track CSV and cache it locally.

    Args:
        tid: Track ID.
        config: Configuration object.
        force: If True, refetch even if cached.

    Returns:
        DataFrame with track data.
    """
    paths = get_paths(config)
    track_dir = paths["tables_dir"] / "tracks"
    track_dir.mkdir(parents=True, exist_ok=True)
    safe_tid = str(tid)
    path = track_dir / f"{safe_tid}.csv"
    if path.exists() and not force:
        try:
            # Guard against truncated/empty cache files
            if path.stat().st_size < 64:
                raise ValueError("cache file too small")
            return pd.read_csv(path)
        except Exception:
            # fall through to refetch
            pass
    df = fetch_track_df(safe_tid, config.api_base)
    tmp_path = track_dir / f"{safe_tid}.csv.tmp"
    df.to_csv(tmp_path, index=False)
    os.replace(str(tmp_path), str(path))
    return df


def read_or_fetch_csv(name: str, url: str, config: Config, *, force: bool = False) -> pd.DataFrame:
    """Read cached CSV or fetch from URL.

    Args:
        name: Filename for cache.
        url: URL to fetch from.
        config: Configuration object.
        force: If True, refetch even if cached.

    Returns:
        DataFrame.
    """
    paths = get_paths(config)
    path = paths["tables_dir"] / name
    if path.exists() and not force:
        return pd.read_csv(path)
    df = pd.read_csv(url)
    df.to_csv(path, index=False)
    return df


def load_case_npz_raw(caseid: int, config: Config) -> Tuple[np.ndarray, int]:
    """Load raw case tensor from cached NPZ file.

    Args:
        caseid: Case ID.
        config: Configuration object.

    Returns:
        Tuple of (x tensor, valid_len).

    Raises:
        FileNotFoundError: If file doesn't exist.
        ValueError: If file is corrupted or invalid.
    """
    paths = get_paths(config)
    p = paths["cache_dir"] / f"case_{int(caseid)}.npz"
    
    if not p.exists():
        raise FileNotFoundError(f"NPZ file not found: {p}")
    
    # Check file size (corrupted files are often very small)
    file_size = p.stat().st_size
    if file_size < 100:  # NPZ files should be at least a few hundred bytes
        raise ValueError(
            f"NPZ file appears corrupted (too small: {file_size} bytes): {p}. "
            f"Please remove this file and regenerate it by running preprocessing with --force flag."
        )
    
    try:
        data = np.load(p, allow_pickle=True)
    except (OSError, ValueError, EOFError, Exception) as e:
        raise ValueError(
            f"Failed to load NPZ file (may be corrupted): {p}. "
            f"Error: {e}. "
            f"Please remove this file and regenerate it by running preprocessing with --force flag."
        )
    
    if "x" not in data:
        raise ValueError(f"NPZ file missing 'x' key: {p}")
    if "valid_len" not in data:
        raise ValueError(f"NPZ file missing 'valid_len' key: {p}")
    
    x = data["x"].astype(np.float32)
    valid_len = int(data["valid_len"])
    return x, valid_len


def load_fold_scalers(fold_idx: int, config: Config) -> Dict[str, Dict]:
    """Load normalization scalers for a fold.

    Args:
        fold_idx: Fold index (1-based).
        config: Configuration object.

    Returns:
        Dictionary of scalers per signal.
    """
    paths = get_paths(config)
    p = paths["scalers_dir"] / f"scalers_fold{int(fold_idx)}.json"
    return json.loads(p.read_text(encoding="utf-8"))


def apply_scalers_to_x(x: np.ndarray, valid_len: int, scalers: Dict[str, Dict], config: Config) -> np.ndarray:
    """Apply normalization scalers to tensor.

    Args:
        x: Raw tensor (signals + masks).
        valid_len: Valid length.
        scalers: Dictionary of scalers per signal.
        config: Configuration object.

    Returns:
        Normalized tensor.
    """
    x = x.astype(np.float32, copy=True)
    signal_names = list(config.signals)
    n_sig = len(signal_names)
    sig = x[:n_sig, :valid_len]
    mask = x[n_sig : 2 * n_sig, :valid_len]

    for i, name in enumerate(signal_names):
        sc = scalers.get(name, {})
        idx = mask[i] > 0.5
        if not np.any(idx):
            continue

        if sc.get("type") == "robust":
            med = float(sc.get("median", 0.0))
            iqr = float(sc.get("iqr", 1.0))
            if iqr == 0.0:
                iqr = 1.0
            sig[i, idx] = (sig[i, idx] - med) / iqr
        elif sc.get("type") == "z":
            mean = float(sc.get("mean", 0.0))
            std = float(sc.get("std", 1.0))
            if std == 0.0:
                std = 1.0
            sig[i, idx] = (sig[i, idx] - mean) / std

    x[:n_sig, :valid_len] = sig
    return x

