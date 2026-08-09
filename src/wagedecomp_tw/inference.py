from __future__ import annotations

import numpy as np
import pandas as pd


def circular_block_indices(
    n_differences: int,
    replications: int = 10000,
    block_length: int = 3,
    seed: int = 20260809,
) -> np.ndarray:
    if n_differences < 1 or block_length < 1:
        raise ValueError("Positive difference count and block length are required")
    rng = np.random.default_rng(seed)
    blocks = int(np.ceil(n_differences / block_length))
    starts = rng.integers(0, n_differences, size=(replications, blocks))
    offsets = np.arange(block_length)
    return ((starts[..., None] + offsets) % n_differences).reshape(replications, -1)[:, :n_differences]


def block_bootstrap_decomposition(
    panel: pd.DataFrame,
    metric: str,
    start: int,
    end: int,
    replications: int = 10000,
    block_length: int = 3,
    seed: int = 20260809,
) -> pd.DataFrame:
    subset = panel.loc[panel.year.between(start, end), ["year", "industry", "employees", metric]].copy()
    years = sorted(subset.year.unique())
    if years != list(range(start, end + 1)):
        raise ValueError("Bootstrap requires a complete annual sequence")
    complete = subset.groupby("industry").year.nunique()
    complete = complete.index[complete.eq(len(years))]
    subset = subset.loc[subset.industry.isin(complete)]
    employment = subset.pivot(index="year", columns="industry", values="employees").loc[years].to_numpy(float)
    wage = subset.pivot(index="year", columns="industry", values=metric).loc[years].to_numpy(float)
    indices = circular_block_indices(len(years) - 1, replications, block_length, seed)
    end_employment = np.exp(np.log(employment[0]) + np.diff(np.log(employment), axis=0)[indices].sum(axis=1))
    end_wage = np.exp(np.log(wage[0]) + np.diff(np.log(wage), axis=0)[indices].sum(axis=1))
    valid = (end_employment > 0).all(axis=1) & (end_wage > 0).all(axis=1)
    if valid.mean() < 0.999:
        raise ValueError("Invalid log-difference bootstrap reconstructions")
    end_employment = end_employment[valid]
    end_wage = end_wage[valid]
    s0 = employment[0] / employment[0].sum()
    s1 = end_employment / end_employment.sum(axis=1, keepdims=True)
    w0 = wage[0]
    ds = s1 - s0
    dw = end_wage - w0
    values = {
        "within": np.sum(s0 * dw, axis=1),
        "shift": np.sum(w0 * ds, axis=1),
        "interaction": np.sum(ds * dw, axis=1),
    }
    values["total"] = values["within"] + values["shift"] + values["interaction"]
    rows = []
    for component, sample in values.items():
        p_value = min(1.0, 2.0 * min(float(np.mean(sample <= 0)), float(np.mean(sample >= 0))))
        rows.append(
            {
                "metric": metric,
                "start_year": start,
                "end_year": end,
                "method": "laspeyres",
                "component": component,
                "estimate_mean": float(np.mean(sample)),
                "ci_lower": float(np.quantile(sample, 0.025)),
                "ci_upper": float(np.quantile(sample, 0.975)),
                "p_value_two_sided": p_value,
                "valid_replications": int(len(sample)),
                "seed": seed,
                "block_length": block_length,
            }
        )
    return pd.DataFrame(rows)
