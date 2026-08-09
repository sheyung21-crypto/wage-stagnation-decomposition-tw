from __future__ import annotations

import numpy as np
import pandas as pd


def frozen_share_path(panel: pd.DataFrame, metric: str, base_year: int) -> pd.DataFrame:
    base = panel.loc[panel.year.eq(base_year), ["industry", "employees"]].copy()
    if base.empty:
        raise ValueError(f"No data for base year {base_year}")
    base["base_share"] = base.employees / base.employees.sum()
    merged = panel.loc[panel.year.ge(base_year)].merge(base[["industry", "base_share"]], on="industry", validate="many_to_one")
    rows = []
    for year, group in merged.groupby("year"):
        actual = float(np.average(group[metric], weights=group.employees))
        counterfactual = float(np.sum(group.base_share * group[metric]))
        rows.append(
            {
                "base_year": base_year,
                "year": int(year),
                "metric": metric,
                "actual": actual,
                "counterfactual": counterfactual,
                "difference": counterfactual - actual,
                "difference_percent_actual": (counterfactual / actual - 1.0) * 100.0,
            }
        )
    return pd.DataFrame(rows)

