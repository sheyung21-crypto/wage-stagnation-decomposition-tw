from __future__ import annotations

import numpy as np
import pandas as pd

from .deflate import rebase_cpi, to_reference_prices


MONTHLY_METRICS = ["regular_monthly", "total_monthly"]


def enrich_panel(panel: pd.DataFrame, annual_cpi: pd.DataFrame, reference_year: int = 2024) -> pd.DataFrame:
    result = panel.copy()
    cpi = annual_cpi.set_index("year").cpi
    cpi_rebased = rebase_cpi(cpi, reference_year)
    result["employment_share"] = result.employees / result.groupby("year").employees.transform("sum")
    result["regular_hourly"] = result.regular_monthly / result.regular_hours
    result["total_hourly"] = result.total_monthly / result.total_hours
    for metric in ["regular_monthly", "total_monthly", "regular_hourly", "total_hourly"]:
        result[f"real_{metric}"] = to_reference_prices(result[metric], result.year, cpi_rebased)
    result["cpi_2024_100"] = result.year.map(cpi_rebased)
    if not np.allclose(result.groupby("year").employment_share.sum().values, 1.0, atol=1e-12):
        raise AssertionError("Employment shares do not add to one")
    if (result.employees < 0).any() or (result[["regular_monthly", "total_monthly"]] < 0).any().any():
        raise AssertionError("Employment and earnings must be nonnegative")
    if (result[["regular_hours", "total_hours"]] <= 0).any().any():
        raise AssertionError("Working hours must be positive")
    return result


def aggregate_wage(panel: pd.DataFrame, metric: str) -> pd.Series:
    numerator = (panel[metric] * panel.employees).groupby(panel.year).sum()
    denominator = panel.employees.groupby(panel.year).sum()
    return numerator / denominator


def common_industry_path(panel: pd.DataFrame, metric: str, start: int, end: int = 2024) -> pd.DataFrame:
    subset = panel.loc[panel.year.between(start, end)]
    counts = subset.groupby("industry").year.nunique()
    common = counts.index[counts.eq(end - start + 1)]
    subset = subset.loc[subset.industry.isin(common)]
    aggregate = aggregate_wage(subset, metric)
    return pd.DataFrame({"year": aggregate.index.astype(int), "value": aggregate.values, "metric": metric})
