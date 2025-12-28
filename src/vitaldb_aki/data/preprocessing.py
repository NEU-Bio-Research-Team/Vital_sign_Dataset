"""Preprocessing pipeline for VitalDB AKI prediction."""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from ..config import Config
from ..data.loaders import load_track_df_cached, read_or_fetch_csv, load_case_npz_raw
from ..utils.helpers import (
    save_json,
    safe_lower_series,
    ensure_cols,
    infer_time_value_cols,
    transform_signal,
    effective_grid_start_end,
    grid_seconds,
)
from ..utils.paths import get_paths, setup_directories

# Ordered regex fallback patterns per signal
PATTERNS: Dict[str, List[str]] = {
    "ART_MBP": [r"(^|/)ART_MBP$", r"ART_MBP", r"\bMBP\b"],
    "CVP": [r"(^|/)CVP$", r"\bCVP\b"],
    "NEPI_RATE": [r"(^|/)NEPI_RATE$", r"NEPI_RATE", r"NOREPI", r"NOREP"],
    "PLETH_HR": [r"(^|/)PLETH_HR$", r"PLETH_HR", r"\bHR\b"],
    "PLETH_SPO2": [r"(^|/)PLETH_SPO2$", r"PLETH_SPO2", r"\bSPO2\b"],
}


def build_aki_labels(
    config: Config,
    *,
    force: bool = False,
    name_col: str = "name",
    dt_col: str = "dt",
    result_col: str = "result",
    caseid_col: str = "caseid",
    cr_names: Tuple[str, ...] = ("cr", "creatinine"),
) -> pd.DataFrame:
    """Build AKI labels from labs data.

    Args:
        config: Configuration object.
        force: If True, rebuild even if labels exist.
        name_col: Column name for lab name.
        dt_col: Column name for time delta.
        result_col: Column name for lab result.
        caseid_col: Column name for case ID.
        cr_names: Names for creatinine labs.

    Returns:
        DataFrame with AKI labels and provenance.
    """
    paths = get_paths(config)
    labels_path = paths["artifacts_dir"] / "labels.csv"

    if labels_path.exists() and not force:
        print(f"Loading existing labels from {labels_path}")
        return pd.read_csv(labels_path)

    print("Building AKI labels from labs...")
    labs_url = f"{config.api_base}/labs"
    df_labs = read_or_fetch_csv("labs.csv", labs_url, config, force=force)

    labs = df_labs.copy()
    ensure_cols(labs, [caseid_col, name_col, dt_col, result_col], "labs")
    labs[name_col] = safe_lower_series(labs[name_col])
    labs = labs[labs[name_col].isin(cr_names)].copy()
    labs[dt_col] = pd.to_numeric(labs[dt_col], errors="coerce")
    labs[result_col] = pd.to_numeric(labs[result_col], errors="coerce")
    labs = labs.dropna(subset=[caseid_col, dt_col, result_col])
    labs[caseid_col] = pd.to_numeric(labs[caseid_col], errors="coerce").astype("Int64")
    labs = labs.dropna(subset=[caseid_col])
    labs[caseid_col] = labs[caseid_col].astype(int)

    # Baseline window: [-baseline_window_sec, 0]
    pre = labs[
        (labs[dt_col] <= 0) & (labs[dt_col] >= -config.baseline_window_sec)
    ].copy()
    pre = pre.sort_values([caseid_col, dt_col])
    idx_base = pre.groupby(caseid_col)[dt_col].idxmax()
    baseline = pre.loc[idx_base, [caseid_col, dt_col, result_col]].rename(
        columns={dt_col: "baseline_dt", result_col: "baseline_cr"}
    )

    # Postop window: [0, postop_window_sec]
    post = labs[
        (labs[dt_col] >= 0) & (labs[dt_col] <= config.postop_window_sec)
    ].copy()
    post = post.sort_values([caseid_col, result_col, dt_col])
    idx_post = post.groupby(caseid_col)[result_col].idxmax()
    postop = post.loc[idx_post, [caseid_col, dt_col, result_col]].rename(
        columns={dt_col: "postop_dt_of_max", result_col: "postop_max_cr"}
    )

    # Counts for provenance
    n_pre = pre.groupby(caseid_col).size().rename("n_preop_labs")
    n_post = post.groupby(caseid_col).size().rename("n_postop_labs")

    out = baseline.merge(postop, on=caseid_col, how="inner")
    out = out.merge(n_pre, on=caseid_col, how="left").merge(
        n_post, on=caseid_col, how="left"
    )
    out["aki"] = (out["postop_max_cr"] >= 1.5 * out["baseline_cr"]).astype(int)
    out = out.sort_values(caseid_col).reset_index(drop=True)

    out.to_csv(labels_path, index=False)
    print(f"Saved labels: {labels_path}")
    print(f"Labelled cases: {len(out)}, AKI positive: {int(out['aki'].sum())}")

    return out


def pick_tid_for_signal(group: pd.DataFrame, signal: str) -> Optional[str]:
    """Pick a single tid for a signal using ordered regex fallback.

    Args:
        group: DataFrame with tracks for a case.
        signal: Signal name.

    Returns:
        Track ID or None if not found.
    """
    pats = PATTERNS.get(signal, [signal])
    tname = group["tname"].astype(str)
    for pat in pats:
        m = tname.str.contains(pat, regex=True, na=False)
        if m.any():
            return str(group.loc[m, "tid"].iloc[0])
    return None


def build_manifest(
    config: Config, df_labels: pd.DataFrame, *, force: bool = False
) -> pd.DataFrame:
    """Build manifest mapping caseid to track IDs.

    Args:
        config: Configuration object.
        df_labels: DataFrame with labels.
        force: If True, rebuild even if manifest exists.

    Returns:
        DataFrame with manifest (caseid -> tid mappings).
    """
    paths = get_paths(config)
    manifest_path = paths["artifacts_dir"] / "manifest_relaxed.csv"

    if manifest_path.exists() and not force:
        print(f"Loading existing manifest from {manifest_path}")
        return pd.read_csv(manifest_path)

    print("Building manifest...")
    trks_url = f"{config.api_base}/trks"
    df_trks = read_or_fetch_csv("trks.csv", trks_url, config, force=force)

    ensure_cols(df_trks, ["caseid", "tid", "tname"], "trks")
    df_trks = df_trks.copy()
    df_trks["caseid"] = pd.to_numeric(df_trks["caseid"], errors="coerce").astype("Int64")
    df_trks = df_trks.dropna(subset=["caseid"]).copy()
    df_trks["caseid"] = df_trks["caseid"].astype(int)
    df_trks["tid"] = df_trks["tid"].astype(str)
    df_trks = df_trks[df_trks["tid"].str.len() > 0].copy()
    df_trks["tname"] = df_trks["tname"].astype(str)

    if len(df_labels) == 0:
        raise ValueError(
            "df_labels is empty. Cannot build manifest without labelled cases."
        )

    label_caseids = set(df_labels["caseid"].astype(int).tolist())
    trks_lab = df_trks[df_trks["caseid"].isin(label_caseids)].copy()

    if len(trks_lab) == 0:
        raise ValueError("No tracks available after restricting to labelled cases.")

    # Build manifest row-by-row
    from tqdm.auto import tqdm

    manifest_rows = []
    for caseid, g in tqdm(trks_lab.groupby("caseid"), desc="Building manifest", unit="case"):
        row = {"caseid": int(caseid)}
        for sig in config.signals:
            row[f"tid_{sig}"] = pick_tid_for_signal(g, sig)
        manifest_rows.append(row)

    tid_cols = [f"tid_{s}" for s in config.signals]
    expected_cols = ["caseid", *tid_cols]
    df_manifest = pd.DataFrame(manifest_rows, columns=expected_cols)

    # Relaxed cohort: keep cases with all required signals
    required_tid_cols = [f"tid_{s}" for s in config.required_signals]
    df_manifest_relaxed = df_manifest.dropna(subset=required_tid_cols).copy()
    df_manifest_relaxed = df_manifest_relaxed.sort_values("caseid").reset_index(drop=True)

    df_manifest_relaxed.to_csv(manifest_path, index=False)
    print(f"Saved manifest: {manifest_path}")
    print(
        f"Cases with required signals: {len(df_manifest_relaxed)} / labelled: {len(df_labels)}"
    )

    return df_manifest_relaxed


def resample_to_grid(
    times: np.ndarray,
    values: np.ndarray,
    grid_t: np.ndarray,
    *,
    cutoff_sec: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Resample irregular time series onto uniform grid.

    Args:
        times: Irregular time points.
        values: Values at time points.
        grid_t: Uniform time grid.
        cutoff_sec: Cutoff time in seconds.

    Returns:
        Tuple of (signal, mask) arrays.
    """
    times = np.asarray(times, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    ok = np.isfinite(times) & np.isfinite(values)
    times = times[ok]
    values = values[ok]
    if times.size == 0:
        sig = np.zeros_like(grid_t, dtype=np.float32)
        mask = np.zeros_like(grid_t, dtype=np.float32)
        return sig, mask

    order = np.argsort(times)
    times = times[order]
    values = values[order]

    # Remove duplicate times (keep last)
    times_rev = times[::-1]
    values_rev = values[::-1]
    _, uniq_idx_rev = np.unique(times_rev, return_index=True)
    keep_rev = np.sort(uniq_idx_rev)
    times = times_rev[keep_rev][::-1]
    values = values_rev[keep_rev][::-1]

    t_min = float(np.min(times))
    t_max = float(np.max(times))

    inside = (grid_t >= t_min) & (grid_t <= t_max) & (grid_t <= cutoff_sec)
    sig = np.full_like(grid_t, np.nan, dtype=np.float32)
    mask = np.zeros_like(grid_t, dtype=np.float32)
    if inside.any():
        sig[inside] = np.interp(
            grid_t[inside].astype(np.float64), times, values
        ).astype(np.float32)
        mask[inside] = 1.0
    sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return sig, mask


def get_cutoff_sec(config: Config) -> float:
    """Get cutoff time in seconds.

    Args:
        config: Configuration object.

    Returns:
        Cutoff time in seconds.
    """
    if config.cutoff_mode == "preop":
        return 0.0
    return float(min(config.t_cut_sec, config.max_len_sec))


def _tid_or_none(v) -> Optional[str]:
    """Convert value to tid string or None."""
    if v is None:
        return None
    if isinstance(v, float) and np.isnan(v):
        return None
    s = str(v).strip()
    if s == "" or s.lower() in ("nan", "none", "null"):
        return None
    return s


def _process_one_signal(
    sig_name: str, tid: str, config: Config, grid_t: np.ndarray, cutoff_sec: float
) -> Tuple[str, np.ndarray, np.ndarray, int]:
    """Process a single signal track.

    Args:
        sig_name: Signal name.
        tid: Track ID.
        config: Configuration object.
        grid_t: Time grid.
        cutoff_sec: Cutoff time.

    Returns:
        Tuple of (signal_name, signal_array, mask_array, n_observations).
    """
    df_track = load_track_df_cached(tid, config)
    t_col, v_col = infer_time_value_cols(df_track)
    t = pd.to_numeric(df_track[t_col], errors="coerce").to_numpy(dtype=np.float64)
    v = pd.to_numeric(df_track[v_col], errors="coerce").to_numpy(dtype=np.float64)
    sig, mask = resample_to_grid(t, v, grid_t, cutoff_sec=cutoff_sec)
    sig = transform_signal(sig_name, sig.astype(np.float32))
    sig = sig * mask
    return sig_name, sig.astype(np.float32), mask.astype(np.float32), int(mask.sum())


def build_case_tensor_from_manifest_row(
    row: pd.Series,
    config: Config,
    grid_t: np.ndarray,
    cutoff_sec: float,
    *,
    executor: ThreadPoolExecutor,
) -> Tuple[np.ndarray, int, Dict[str, int]]:
    """Build case tensor from manifest row.

    Args:
        row: Manifest row.
        config: Configuration object.
        grid_t: Time grid.
        cutoff_sec: Cutoff time.
        executor: Thread pool executor.

    Returns:
        Tuple of (x tensor, valid_len, obs_counts).
    """
    T = len(grid_t)
    sig_names = list(config.signals)
    n_sig = len(sig_names)
    sig_mat = np.zeros((n_sig, T), dtype=np.float32)
    mask_mat = np.zeros((n_sig, T), dtype=np.float32)
    obs_counts: Dict[str, int] = {name: 0 for name in sig_names}

    futures = {}
    for name in sig_names:
        is_required = name in config.required_signals
        is_optional = not is_required
        if is_optional and not config.include_optional_signals:
            continue
        tid = _tid_or_none(row.get(f"tid_{name}"))
        if tid is None:
            continue
        futures[
            executor.submit(_process_one_signal, name, tid, config, grid_t, cutoff_sec)
        ] = name

    for fut in as_completed(futures):
        name = futures[fut]
        try:
            sig_name, sig, mask, n_obs = fut.result()
        except Exception:
            obs_counts[name] = 0
            continue
        i = sig_names.index(sig_name)
        sig_mat[i] = sig
        mask_mat[i] = mask
        obs_counts[sig_name] = int(n_obs)

    any_mask = mask_mat.sum(axis=0) > 0
    valid_len = int(np.max(np.where(any_mask)[0]) + 1) if any_mask.any() else 0
    x = np.concatenate([sig_mat, mask_mat], axis=0).astype(np.float32)
    return x, valid_len, obs_counts


def quality_gates(
    obs_counts: Dict[str, int], valid_len: int, config: Config
) -> Tuple[bool, str]:
    """Check quality gates for a case.

    Args:
        obs_counts: Observation counts per signal.
        valid_len: Valid length.
        config: Configuration object.

    Returns:
        Tuple of (passed, reason).
    """
    for sig_name in config.required_signals:
        if obs_counts.get(sig_name, 0) < config.min_obs_points_per_channel:
            return False, f"min_obs_fail_required:{sig_name}"
    if valid_len < int(config.min_len_sec * config.fs_hz):
        return False, "min_len_fail"
    return True, "ok"


def ingest_tracks(
    config: Config, df_manifest: pd.DataFrame, *, force: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Ingest tracks and create cached tensors.

    Args:
        config: Configuration object.
        df_manifest: Manifest DataFrame.
        force: If True, reprocess even if cached.

    Returns:
        Tuple of (usable_df, failed_df).
    """
    paths = get_paths(config)
    usable_path = paths["artifacts_dir"] / "df_usable.csv"
    failed_path = paths["artifacts_dir"] / "df_failed.csv"

    if usable_path.exists() and failed_path.exists() and not force:
        print(f"Loading existing ingestion results from {usable_path}")
        return pd.read_csv(usable_path), pd.read_csv(failed_path)

    print("Ingesting tracks and creating cached tensors...")

    # Generate time grid
    start, end = effective_grid_start_end(
        config.cutoff_mode,
        config.max_len_sec,
        config.t_cut_sec,
        config.preop_window_sec,
    )
    grid_t = grid_seconds(
        config.fs_hz,
        config.cutoff_mode,
        config.max_len_sec,
        config.t_cut_sec,
        config.preop_window_sec,
    )
    cutoff_sec = get_cutoff_sec(config)
    print(f"Cutoff mode: {config.cutoff_mode}, cutoff_sec: {cutoff_sec}")

    df_manifest_run = df_manifest
    if config.max_cases_to_cache is not None:
        df_manifest_run = df_manifest_run.head(int(config.max_cases_to_cache)).copy()
    print(f"Caching cases: {len(df_manifest_run)} (of {len(df_manifest)})")

    usable_rows = []
    failed_rows = []

    from tqdm.auto import tqdm

    with ThreadPoolExecutor(max_workers=int(config.n_threads)) as ex:
        for _, row in tqdm(
            df_manifest_run.iterrows(),
            total=len(df_manifest_run),
            desc="Caching cases",
            unit="case",
        ):
            caseid = int(row["caseid"])
            out_path = paths["cache_dir"] / f"case_{caseid}.npz"
            if out_path.exists() and not force:
                usable_rows.append(
                    {"caseid": caseid, "cache_path": str(out_path), "status": "cached"}
                )
                continue
            try:
                x, valid_len, obs_counts = build_case_tensor_from_manifest_row(
                    row, config, grid_t, cutoff_sec, executor=ex
                )
                ok, reason = quality_gates(obs_counts, valid_len, config)
                if not ok:
                    failed_rows.append(
                        {
                            "caseid": caseid,
                            "reason": reason,
                            **{f"obs_{k}": v for k, v in obs_counts.items()},
                            "valid_len": valid_len,
                        }
                    )
                    continue
                np.savez_compressed(
                    out_path,
                    x=x,
                    valid_len=np.int32(valid_len),
                    grid_t=grid_t,
                    cutoff_mode=np.array([config.cutoff_mode]),
                    cutoff_sec=np.float32(cutoff_sec),
                    obs_counts=np.array([obs_counts], dtype=object),
                )
                usable_rows.append(
                    {
                        "caseid": caseid,
                        "cache_path": str(out_path),
                        "status": "new",
                        **{f"obs_{k}": v for k, v in obs_counts.items()},
                        "valid_len": valid_len,
                    }
                )
            except Exception as e:
                failed_rows.append(
                    {
                        "caseid": caseid,
                        "reason": f"exception:{type(e).__name__}",
                        "detail": str(e)[:200],
                    }
                )

    df_usable = pd.DataFrame(usable_rows)
    df_failed = pd.DataFrame(failed_rows)
    df_usable.to_csv(usable_path, index=False)
    df_failed.to_csv(failed_path, index=False)
    print(f"Saved usable: {usable_path} ({len(df_usable)} cases)")
    print(f"Saved failed: {failed_path} ({len(df_failed)} cases)")

    return df_usable, df_failed


def create_folds(
    config: Config,
    df_labels: pd.DataFrame,
    df_manifest: pd.DataFrame,
    df_usable: pd.DataFrame,
    *,
    force: bool = False,
) -> Tuple[List[Dict], pd.DataFrame]:
    """Create cross-validation folds.

    Args:
        config: Configuration object.
        df_labels: Labels DataFrame.
        df_manifest: Manifest DataFrame.
        df_usable: Usable cases DataFrame.
        force: If True, recreate even if folds exist.

    Returns:
        Tuple of (folds list, master DataFrame).
    """
    paths = get_paths(config)
    folds_path = paths["artifacts_dir"] / "folds.json"
    master_path = paths["artifacts_dir"] / "cohort_master.csv"

    if folds_path.exists() and master_path.exists() and not force:
        print(f"Loading existing folds from {folds_path}")
        with open(folds_path, "r", encoding="utf-8") as f:
            import json

            folds = json.load(f)
        df_master = pd.read_csv(master_path)
        return folds, df_master

    print("Creating cross-validation folds...")

    usable_caseids = set(
        pd.to_numeric(df_usable["caseid"], errors="coerce").dropna().astype(int).tolist()
    )
    df_labels_usable = df_labels[df_labels["caseid"].isin(usable_caseids)].copy()

    df_master = df_labels_usable.merge(df_manifest, on="caseid", how="inner")
    df_master = df_master.merge(
        df_usable[["caseid", "cache_path"]], on="caseid", how="inner"
    )
    df_master = df_master.sort_values("caseid").reset_index(drop=True)

    df_master.to_csv(master_path, index=False)
    print(f"Saved cohort_master: {master_path} ({len(df_master)} cases)")

    caseids = df_master["caseid"].astype(int).to_numpy()
    y = df_master["aki"].astype(int).to_numpy()

    skf = StratifiedKFold(
        n_splits=config.n_splits, shuffle=True, random_state=config.random_state
    )
    folds = []
    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(caseids, y), start=1):
        folds.append(
            {
                "fold": fold_idx,
                "train_caseids": caseids[tr_idx].tolist(),
                "val_caseids": caseids[va_idx].tolist(),
                "n_train": int(len(tr_idx)),
                "n_val": int(len(va_idx)),
                "train_pos": int(y[tr_idx].sum()),
                "val_pos": int(y[va_idx].sum()),
            }
        )

    save_json(folds_path, folds)
    print(f"Saved folds: {folds_path}")
    print(pd.DataFrame(folds)[["fold", "n_train", "n_val", "train_pos", "val_pos"]])

    return folds, df_master


def fit_channel_stats(values: np.ndarray, *, robust: bool) -> Dict[str, float]:
    """Fit normalization statistics for a channel.

    Args:
        values: Signal values.
        robust: If True, use robust scaling (median/IQR).

    Returns:
        Dictionary with scaling parameters.
    """
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"type": "empty"}
    if robust:
        med = float(np.median(values))
        q25 = float(np.percentile(values, 25))
        q75 = float(np.percentile(values, 75))
        iqr = float(q75 - q25)
        if iqr <= 1e-6:
            iqr = 1.0
        return {"type": "robust", "median": med, "iqr": iqr, "q25": q25, "q75": q75}
    mean = float(values.mean())
    std = float(values.std())
    if std <= 1e-6:
        std = 1.0
    return {"type": "z", "mean": mean, "std": std}


def fit_scalers(
    config: Config, folds: List[Dict], *, force: bool = False
) -> List[Dict]:
    """Fit per-fold normalization scalers.

    Args:
        config: Configuration object.
        folds: List of fold dictionaries.
        force: If True, refit even if scalers exist.

    Returns:
        List of scaler metadata dictionaries.
    """
    paths = get_paths(config)
    fold_scalers_path = paths["artifacts_dir"] / "fold_scalers_index.json"

    if fold_scalers_path.exists() and not force:
        print(f"Loading existing scalers from {fold_scalers_path}")
        import json

        return json.loads(fold_scalers_path.read_text(encoding="utf-8"))

    print("Fitting per-fold scalers...")

    def load_case_x(caseid: int):
        """Load case tensor."""
        x, valid_len = load_case_npz_raw(caseid, config)
        n_sig = len(config.signals)
        sig = x[:n_sig, :valid_len]
        mask = x[n_sig:, :valid_len]
        return sig, mask, valid_len

    def fit_fold_scaler(train_caseids: List[int]) -> Dict[str, Dict[str, float]]:
        """Fit scalers for a fold."""
        sig_names = list(config.signals)
        n_sig = len(sig_names)
        collected: List[List[np.ndarray]] = [[] for _ in range(n_sig)]

        from tqdm.auto import tqdm

        for cid in tqdm(train_caseids, desc="Collect train vals", unit="case"):
            sig, mask, _ = load_case_x(cid)
            for i in range(n_sig):
                vals = sig[i][mask[i] > 0.5]
                if vals.size:
                    collected[i].append(vals.astype(np.float32))

        scalers: Dict[str, Dict[str, float]] = {}
        for i, name in enumerate(sig_names):
            vals = (
                np.concatenate(collected[i], axis=0)
                if collected[i]
                else np.array([], dtype=np.float32)
            )
            robust = name in ("NEPI_RATE",)
            scalers[name] = fit_channel_stats(vals, robust=robust)
            if vals.size:
                scalers[name].update(
                    {
                        "q001": float(np.quantile(vals, 0.001)),
                        "q999": float(np.quantile(vals, 0.999)),
                        "n": int(vals.size),
                    }
                )
            else:
                scalers[name].update({"q001": None, "q999": None, "n": 0})
        return scalers

    fold_scalers = []
    for f in folds:
        fold_idx = int(f["fold"])
        train_caseids = [int(x) for x in f["train_caseids"]]
        scalers = fit_fold_scaler(train_caseids)
        out_path = paths["scalers_dir"] / f"scalers_fold{fold_idx}.json"
        save_json(out_path, scalers)
        fold_scalers.append({"fold": fold_idx, "path": str(out_path)})
        print(f"Saved scalers: {out_path}")

    save_json(fold_scalers_path, fold_scalers)
    print(f"Saved fold scalers index: {fold_scalers_path}")

    return fold_scalers

