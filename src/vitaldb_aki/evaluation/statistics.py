"""Statistical utilities for model comparison.

Focus:
- Bootstrap confidence intervals (non-parametric)
- Paired tests on matched folds (Wilcoxon / paired t-test)

Designed to support imbalanced clinical tasks where AUPRC variance is high.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional, Tuple

import numpy as np


MetricName = Literal["auprc", "auroc"]


def _as_1d_numpy_int(x) -> np.ndarray:
    a = np.asarray(x)
    if a.ndim != 1:
        a = a.reshape(-1)
    return a.astype(int)


def _as_1d_numpy_float(x) -> np.ndarray:
    a = np.asarray(x)
    if a.ndim != 1:
        a = a.reshape(-1)
    return a.astype(float)


def _rng(seed: Optional[int]) -> np.random.Generator:
    return np.random.default_rng(seed)


def _stratified_resample_indices(y_true: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Bootstrap resampling indices that preserves class counts.

    This avoids degenerate resamples with zero positives (common in rare-event AKI).
    """

    y_true = _as_1d_numpy_int(y_true)
    pos_idx = np.flatnonzero(y_true == 1)
    neg_idx = np.flatnonzero(y_true == 0)

    if len(pos_idx) == 0 or len(neg_idx) == 0:
        # Fallback to vanilla bootstrap; metric may be undefined but caller handles it.
        return rng.integers(0, len(y_true), size=len(y_true), endpoint=False)

    pos_s = rng.choice(pos_idx, size=len(pos_idx), replace=True)
    neg_s = rng.choice(neg_idx, size=len(neg_idx), replace=True)
    idx = np.concatenate([pos_s, neg_s])
    rng.shuffle(idx)
    return idx


def bootstrap_ci(
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    y_true,
    y_score,
    *,
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    seed: Optional[int] = 0,
    stratified: bool = True,
) -> Tuple[float, Tuple[float, float], np.ndarray]:
    """Compute a bootstrap CI for a metric.

    Returns (point_estimate, (lo, hi), bootstrap_samples).
    """

    y_true_arr = _as_1d_numpy_int(y_true)
    y_score_arr = _as_1d_numpy_float(y_score)

    if len(y_true_arr) != len(y_score_arr):
        raise ValueError("y_true and y_score must have the same length")

    point = float(metric_fn(y_true_arr, y_score_arr))

    rng = _rng(seed)
    samples: list[float] = []
    max_tries = max(n_bootstrap * 10, 10_000)
    tries = 0

    while len(samples) < n_bootstrap and tries < max_tries:
        tries += 1
        if stratified:
            idx = _stratified_resample_indices(y_true_arr, rng)
        else:
            idx = rng.integers(0, len(y_true_arr), size=len(y_true_arr), endpoint=False)

        yt = y_true_arr[idx]
        ys = y_score_arr[idx]

        try:
            v = float(metric_fn(yt, ys))
        except Exception:
            v = float("nan")

        if np.isfinite(v):
            samples.append(v)

    if len(samples) < max(50, int(0.5 * n_bootstrap)):
        raise RuntimeError(
            f"Too many invalid bootstrap resamples: got {len(samples)}/{n_bootstrap}. "
            "Check labels, prevalence, or turn off stratified bootstrapping."
        )

    boot = np.asarray(samples, dtype=float)
    lo = float(np.quantile(boot, alpha / 2.0))
    hi = float(np.quantile(boot, 1.0 - alpha / 2.0))
    return point, (lo, hi), boot


def paired_bootstrap_delta_ci(
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    y_true,
    y_score_a,
    y_score_b,
    *,
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    seed: Optional[int] = 0,
    stratified: bool = True,
) -> Tuple[float, Tuple[float, float], np.ndarray]:
    """Paired bootstrap CI for delta = metric(a) - metric(b) using the same resamples."""

    y_true_arr = _as_1d_numpy_int(y_true)
    a = _as_1d_numpy_float(y_score_a)
    b = _as_1d_numpy_float(y_score_b)

    if len(y_true_arr) != len(a) or len(y_true_arr) != len(b):
        raise ValueError("y_true, y_score_a, y_score_b must have the same length")

    point = float(metric_fn(y_true_arr, a) - metric_fn(y_true_arr, b))

    rng = _rng(seed)
    samples: list[float] = []
    max_tries = max(n_bootstrap * 10, 10_000)
    tries = 0

    while len(samples) < n_bootstrap and tries < max_tries:
        tries += 1
        if stratified:
            idx = _stratified_resample_indices(y_true_arr, rng)
        else:
            idx = rng.integers(0, len(y_true_arr), size=len(y_true_arr), endpoint=False)

        yt = y_true_arr[idx]
        aa = a[idx]
        bb = b[idx]

        try:
            v = float(metric_fn(yt, aa) - metric_fn(yt, bb))
        except Exception:
            v = float("nan")

        if np.isfinite(v):
            samples.append(v)

    if len(samples) < max(50, int(0.5 * n_bootstrap)):
        raise RuntimeError(
            f"Too many invalid paired bootstrap resamples: got {len(samples)}/{n_bootstrap}."
        )

    boot = np.asarray(samples, dtype=float)
    lo = float(np.quantile(boot, alpha / 2.0))
    hi = float(np.quantile(boot, 1.0 - alpha / 2.0))
    return point, (lo, hi), boot


@dataclass(frozen=True)
class PairedTestResult:
    n: int
    mean_delta: float
    median_delta: float
    test: str
    statistic: float
    pvalue: float
    alternative: str


def paired_test(
    a,
    b,
    *,
    test: Literal["wilcoxon", "ttest"] = "wilcoxon",
    alternative: Literal["two-sided", "greater", "less"] = "greater",
) -> PairedTestResult:
    """Paired significance test on matched samples.

    Parameters
    - a, b: arrays of matched metric values
    - alternative: default 'greater' to test H1: a > b
    """

    aa = _as_1d_numpy_float(a)
    bb = _as_1d_numpy_float(b)
    if len(aa) != len(bb):
        raise ValueError("a and b must have the same length")

    delta = aa - bb

    # Drop NaNs in a paired way
    mask = np.isfinite(delta)
    aa = aa[mask]
    bb = bb[mask]
    delta = delta[mask]

    if len(delta) < 2:
        raise ValueError("Not enough paired samples after removing NaNs")

    if test == "wilcoxon":
        from scipy.stats import wilcoxon

        stat, p = wilcoxon(aa, bb, alternative=alternative, zero_method="wilcox")
        stat = float(stat)
        p = float(p)
    elif test == "ttest":
        from scipy.stats import ttest_rel

        res = ttest_rel(aa, bb, alternative=alternative)
        stat = float(res.statistic)
        p = float(res.pvalue)
    else:
        raise ValueError(f"Unknown test: {test}")

    return PairedTestResult(
        n=int(len(delta)),
        mean_delta=float(np.mean(delta)),
        median_delta=float(np.median(delta)),
        test=str(test),
        statistic=stat,
        pvalue=p,
        alternative=str(alternative),
    )


def get_metric_fn(name: MetricName) -> Callable[[np.ndarray, np.ndarray], float]:
    if name == "auprc":
        from sklearn.metrics import average_precision_score

        return lambda y, p: float(average_precision_score(y, p))
    if name == "auroc":
        from sklearn.metrics import roc_auc_score

        return lambda y, p: float(roc_auc_score(y, p))
    raise ValueError(f"Unknown metric: {name}")
