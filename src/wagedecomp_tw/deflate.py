from __future__ import annotations

import pandas as pd


def rebase_cpi(cpi: pd.Series, reference_year: int = 2024) -> pd.Series:
    if reference_year not in cpi.index:
        raise ValueError(f"CPI reference year {reference_year} is unavailable")
    if (cpi <= 0).any():
        raise ValueError("CPI must be strictly positive")
    return cpi / float(cpi.loc[reference_year]) * 100.0


def to_reference_prices(values: pd.Series, years: pd.Series, cpi: pd.Series) -> pd.Series:
    mapped = years.map(cpi)
    if mapped.isna().any():
        missing = sorted(years[mapped.isna()].unique())
        raise ValueError(f"Missing CPI years: {missing}")
    return values * 100.0 / mapped

