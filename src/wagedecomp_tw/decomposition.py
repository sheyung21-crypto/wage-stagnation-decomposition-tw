from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _endpoints(panel: pd.DataFrame, metric: str, start: int, end: int) -> tuple[np.ndarray, ...]:
    left = panel.loc[panel.year.eq(start), ["industry", "employees", metric]].set_index("industry")
    right = panel.loc[panel.year.eq(end), ["industry", "employees", metric]].set_index("industry")
    common = left.index.intersection(right.index)
    if len(common) == 0:
        raise ValueError(f"No common endpoint industries for {start}-{end}")
    left = left.loc[common]
    right = right.loc[common]
    s0 = left.employees.to_numpy(float) / left.employees.sum()
    s1 = right.employees.to_numpy(float) / right.employees.sum()
    w0 = left[metric].to_numpy(float)
    w1 = right[metric].to_numpy(float)
    if np.any(s0 <= 0) or np.any(s1 <= 0) or np.any(w0 <= 0) or np.any(w1 <= 0):
        raise ValueError("Törnqvist-compatible endpoint data must be strictly positive")
    return s0, s1, w0, w1


def laspeyres_arrays(s0: np.ndarray, s1: np.ndarray, w0: np.ndarray, w1: np.ndarray) -> dict[str, float]:
    ds = s1 - s0
    dw = w1 - w0
    within = float(np.sum(s0 * dw))
    shift = float(np.sum(w0 * ds))
    interaction = float(np.sum(ds * dw))
    total = float(np.sum(s1 * w1) - np.sum(s0 * w0))
    return {"within": within, "shift": shift, "interaction": interaction, "residual": 0.0, "total": total}


def paasche_arrays(s0: np.ndarray, s1: np.ndarray, w0: np.ndarray, w1: np.ndarray) -> dict[str, float]:
    ds = s1 - s0
    dw = w1 - w0
    within = float(np.sum(s1 * dw))
    shift = float(np.sum(w1 * ds))
    interaction = float(-np.sum(ds * dw))
    total = float(np.sum(s1 * w1) - np.sum(s0 * w0))
    return {"within": within, "shift": shift, "interaction": interaction, "residual": 0.0, "total": total}


def tornqvist_arrays(s0: np.ndarray, s1: np.ndarray, w0: np.ndarray, w1: np.ndarray) -> dict[str, float]:
    W0 = float(np.sum(s0 * w0))
    W1 = float(np.sum(s1 * w1))
    theta0 = s0 * w0 / W0
    theta1 = s1 * w1 / W1
    theta = (theta0 + theta1) / 2.0
    within = float(np.sum(theta * np.log(w1 / w0)))
    shift = float(np.sum(theta * np.log(s1 / s0)))
    total = math.log(W1 / W0)
    residual = total - within - shift
    return {"within": within, "shift": shift, "interaction": 0.0, "residual": residual, "total": total}


METHODS = {
    "laspeyres": laspeyres_arrays,
    "paasche": paasche_arrays,
    "tornqvist": tornqvist_arrays,
}


def decompose(panel: pd.DataFrame, metric: str, start: int, end: int, method: str) -> dict[str, float | str | int]:
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")
    values = METHODS[method](*_endpoints(panel, metric, start, end))
    check = values["within"] + values["shift"] + values["interaction"] + values["residual"]
    if not math.isclose(check, values["total"], rel_tol=1e-10, abs_tol=1e-10):
        raise AssertionError(f"Decomposition identity failed: {check} != {values['total']}")
    scale = "log_change" if method == "tornqvist" else "2024_twd"
    n_industries = len(set(panel.loc[panel.year.eq(start), "industry"]) & set(panel.loc[panel.year.eq(end), "industry"]))
    return {"start_year": start, "end_year": end, "metric": metric, "method": method, "scale": scale, "n_industries": n_industries, **values}


def nominal_then_deflate_bridge(panel: pd.DataFrame, metric: str, start: int, end: int) -> dict[str, float]:
    nominal = decompose(panel, metric, start, end, "laspeyres")
    base = panel.loc[panel.year.eq(start)]
    W0 = float(np.average(base[metric], weights=base.employees))
    d0 = float(100.0 / base.cpi_2024_100.iloc[0])
    end_rows = panel.loc[panel.year.eq(end)]
    d1 = float(100.0 / end_rows.cpi_2024_100.iloc[0])
    components = {key: float(nominal[key]) * d1 for key in ("within", "shift", "interaction")}
    components["price"] = W0 * (d1 - d0)
    components["total"] = sum(components.values())
    return components
