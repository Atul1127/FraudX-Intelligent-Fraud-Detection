from __future__ import annotations

from typing import Iterable

import numpy as np


def _clean(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    return array[np.isfinite(array)]


def psi(reference: Iterable[float], current: Iterable[float], bins: int = 10) -> float:
    """Population Stability Index for numeric distributions.

    The bin edges are learned from the reference distribution. A small epsilon
    prevents zero-frequency bins from causing infinite values.
    """
    ref = _clean(reference)
    cur = _clean(current)
    if ref.size == 0 or cur.size == 0:
        raise ValueError("Reference and current samples must both contain numeric values")
    if bins < 2:
        raise ValueError("bins must be at least 2")

    edges = np.quantile(ref, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    if edges.size < 3:
        return 0.0
    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)
    eps = 1e-6
    ref_pct = np.clip(ref_counts / ref.size, eps, None)
    cur_pct = np.clip(cur_counts / cur.size, eps, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def drift_level(score: float) -> str:
    """Interpret PSI using common operational bands."""
    if score < 0.10:
        return "stable"
    if score < 0.25:
        return "warning"
    return "drift"


def prediction_shift(reference: Iterable[float], current: Iterable[float]) -> dict[str, float | str]:
    """Summarize score-distribution drift between two prediction windows."""
    score = psi(reference, current)
    return {"psi": score, "level": drift_level(score)}
