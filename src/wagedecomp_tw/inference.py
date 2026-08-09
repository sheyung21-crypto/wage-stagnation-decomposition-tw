from __future__ import annotations

import numpy as np
import pandas as pd

from wagedecomp_tw.decomposition import complete_path_arrays


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


def block_bootstrap_chained_paths(
    panel: pd.DataFrame,
    metric: str,
    start: int,
    end: int,
    method: str = "laspeyres",
    replications: int = 10000,
    block_length: int = 3,
    seed: int = 20260809,
) -> pd.DataFrame:
    """Circular moving-block bootstrap intervals for cumulative chained paths."""
    if method not in {"laspeyres", "paasche"}:
        raise ValueError("Chained bootstrap supports laspeyres or paasche")
    years, industries, employment, wage = complete_path_arrays(panel, metric, start, end)
    indices = circular_block_indices(len(years) - 1, replications, block_length, seed)

    employment_increments = np.diff(np.log(employment), axis=0)[indices]
    wage_increments = np.diff(np.log(wage), axis=0)[indices]
    initial_employment = np.broadcast_to(
        np.log(employment[0]), (replications, 1, len(industries))
    )
    initial_wage = np.broadcast_to(
        np.log(wage[0]), (replications, 1, len(industries))
    )
    employment_path = np.exp(
        np.concatenate(
            [initial_employment, initial_employment + np.cumsum(employment_increments, axis=1)],
            axis=1,
        )
    )
    wage_path = np.exp(
        np.concatenate(
            [initial_wage, initial_wage + np.cumsum(wage_increments, axis=1)], axis=1
        )
    )
    valid = (
        np.isfinite(employment_path).all(axis=(1, 2))
        & np.isfinite(wage_path).all(axis=(1, 2))
        & (employment_path > 0).all(axis=(1, 2))
        & (wage_path > 0).all(axis=(1, 2))
    )
    if valid.mean() < 0.999:
        raise ValueError("Invalid chained bootstrap reconstructions")
    employment_path = employment_path[valid]
    wage_path = wage_path[valid]
    shares = employment_path / employment_path.sum(axis=2, keepdims=True)
    ds = np.diff(shares, axis=1)
    dw = np.diff(wage_path, axis=1)
    cross = np.sum(ds * dw, axis=2)
    if method == "laspeyres":
        annual_within = np.sum(shares[:, :-1] * dw, axis=2)
        annual_shift = np.sum(wage_path[:, :-1] * ds, axis=2)
        annual_interaction = cross
    else:
        annual_within = np.sum(shares[:, 1:] * dw, axis=2)
        annual_shift = np.sum(wage_path[:, 1:] * ds, axis=2)
        annual_interaction = -cross
    aggregate = np.sum(shares * wage_path, axis=2)
    annual_total = np.diff(aggregate, axis=1)
    annual_components = annual_within + annual_shift + annual_interaction
    if not np.allclose(annual_components, annual_total, rtol=1e-10, atol=1e-10):
        raise AssertionError("Bootstrap annual chained identity failed")

    zeros = np.zeros((len(annual_total), 1), dtype=float)
    cumulative = {
        "within": np.concatenate([zeros, np.cumsum(annual_within, axis=1)], axis=1),
        "shift": np.concatenate([zeros, np.cumsum(annual_shift, axis=1)], axis=1),
        "interaction": np.concatenate(
            [zeros, np.cumsum(annual_interaction, axis=1)], axis=1
        ),
        "total": np.concatenate([zeros, np.cumsum(annual_total, axis=1)], axis=1),
    }
    if not np.allclose(
        cumulative["within"] + cumulative["shift"] + cumulative["interaction"],
        cumulative["total"],
        rtol=1e-10,
        atol=1e-10,
    ):
        raise AssertionError("Bootstrap cumulative chained identity failed")

    rows: list[dict[str, float | int | str]] = []
    for component, sample in cumulative.items():
        lower = np.quantile(sample, 0.025, axis=0)
        upper = np.quantile(sample, 0.975, axis=0)
        mean = np.mean(sample, axis=0)
        p_value = np.minimum(
            1.0,
            2.0
            * np.minimum(np.mean(sample <= 0, axis=0), np.mean(sample >= 0, axis=0)),
        )
        for index, year in enumerate(years):
            rows.append(
                {
                    "metric": metric,
                    "method": method,
                    "year": int(year),
                    "component": component,
                    "estimate_mean": float(mean[index]),
                    "ci_lower": float(lower[index]),
                    "ci_upper": float(upper[index]),
                    "p_value_two_sided": float(p_value[index]),
                    "valid_replications": int(len(sample)),
                    "seed": seed,
                    "block_length": block_length,
                    "n_industries": len(industries),
                }
            )
    return pd.DataFrame(rows)
