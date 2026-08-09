from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd


MIGRANT_TO_PANEL = {
    "manufacturing": "manufacturing",
    "construction": "construction",
    "waste": "water_waste",
}


def _number(value: object) -> float:
    text = str(value).strip().replace(",", "").replace("(", "").replace(")", "")
    if not text or text in {"-", "--", "..."}:
        return float("nan")
    return float(text)


def _year(value: object) -> int | None:
    match = re.search(r"(19|20)\d{2}", str(value))
    return int(match.group(0)) if match else None


def read_migrant_counts(path: Path, start_year: int = 2000, end_year: int = 2024) -> pd.DataFrame:
    """Read MOL year-end foreign-worker counts by work responsibility."""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.reader(stream))
    records: list[dict[str, float | int | str]] = []
    for row in rows:
        if len(row) < 17:
            continue
        year = _year(row[1])
        if year is None or not start_year <= year <= end_year:
            continue
        records.append(
            {
                "year": year,
                "grand_total": _number(row[2]),
                "valid_permits": _number(row[3]),
                "productive_total": _number(row[4]),
                "agriculture": _number(row[7]),
                "manufacturing": _number(row[8]),
                "construction": _number(row[9]),
                "waste": _number(row[10]),
                "social_welfare_total": _number(row[11]),
                "nursing": _number(row[14]),
                "home_maids": _number(row[15]),
                "expired_permits": _number(row[16]),
                "migrant_count_basis": "MOL year-end stock; valid permits by work responsibility from 2015 onward",
            }
        )
    result = pd.DataFrame(records).sort_values("year").reset_index(drop=True)
    expected = set(range(start_year, end_year + 1))
    if set(result.year) != expected:
        raise ValueError(f"Migrant series does not cover every requested year: {sorted(expected - set(result.year))}")
    return result


def read_minimum_monthly_wage(
    path: Path, start_year: int = 2000, end_year: int = 2024
) -> pd.DataFrame:
    """Read the official end-of-year monthly minimum wage from MOL table 1-1."""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.reader(stream))
    records = []
    for row in rows:
        if len(row) < 5:
            continue
        year = _year(row[1])
        if year is None or not start_year <= year <= end_year:
            continue
        monthly = _number(row[3])
        # Table 1-1 is continued across several blocks and repeats the year
        # column.  In the preceding block column 4 is a growth-rate change,
        # not the minimum wage.  Restrict the parser to plausible monthly
        # currency levels so that the first repeated year cannot win the
        # later drop_duplicates call.
        if np.isfinite(monthly) and 10_000 <= monthly <= 100_000:
            records.append(
                {
                    "year": year,
                    "minimum_monthly_wage_twd": monthly,
                    "timing_basis": "official monthly minimum wage in force at year end",
                }
            )
    result = pd.DataFrame(records).drop_duplicates("year").sort_values("year").reset_index(drop=True)
    expected = set(range(start_year, end_year + 1))
    if set(result.year) != expected:
        raise ValueError(f"Minimum-wage series does not cover every requested year: {sorted(expected - set(result.year))}")
    return result


def migrant_share_paths(migrants: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Map MOL migrant stocks to DGBAS annual-average payroll employment.

    The numerator is a year-end stock while the denominator is an annual average.
    The function keeps this timing mismatch explicit and stops if any implied share
    exceeds 50 percent, as required by the preregistered external-consistency gate.
    """
    rows: list[dict[str, float | int | str]] = []
    panel_employment = panel.groupby(["year", "industry"], as_index=False).employees.sum()
    for migrant_name, panel_name in MIGRANT_TO_PANEL.items():
        denominator = panel_employment.loc[panel_employment.industry.eq(panel_name), ["year", "employees"]]
        merged = migrants[["year", migrant_name]].merge(denominator, on="year", how="left", validate="one_to_one")
        if merged.employees.isna().any():
            raise ValueError(f"Missing DGBAS employment denominator for {panel_name}")
        merged["migrant_share"] = merged[migrant_name] / merged.employees
        for item in merged.itertuples(index=False):
            rows.append(
                {
                    "year": int(item.year),
                    "industry": panel_name,
                    "migrant_count": float(getattr(item, migrant_name)),
                    "employees_annual_average": float(item.employees),
                    "migrant_share": float(item.migrant_share),
                    "mapping_status": "direct major-industry match",
                    "numerator_timing": "MOL year-end stock",
                    "denominator_timing": "DGBAS annual-average employees",
                }
            )

    matched_numerator = migrants[["year", *MIGRANT_TO_PANEL]].copy()
    matched_numerator["migrant_count"] = matched_numerator[list(MIGRANT_TO_PANEL)].sum(axis=1)
    matched_codes = set(MIGRANT_TO_PANEL.values())
    matched_denominator = (
        panel_employment.loc[panel_employment.industry.isin(matched_codes)]
        .groupby("year", as_index=False)
        .employees.sum()
    )
    matched = matched_numerator[["year", "migrant_count"]].merge(
        matched_denominator, on="year", how="left", validate="one_to_one"
    )
    matched["migrant_share"] = matched.migrant_count / matched.employees
    for item in matched.itertuples(index=False):
        rows.append(
            {
                "year": int(item.year),
                "industry": "matched_productive_industries",
                "migrant_count": float(item.migrant_count),
                "employees_annual_average": float(item.employees),
                "migrant_share": float(item.migrant_share),
                "mapping_status": "manufacturing + construction + water/waste",
                "numerator_timing": "MOL year-end stock",
                "denominator_timing": "DGBAS annual-average employees",
            }
        )

    result = pd.DataFrame(rows)
    invalid = result.loc[
        result.migrant_share.gt(0.5)
        | result.migrant_share.lt(0)
        | result.migrant_count.gt(result.employees_annual_average)
    ]
    if not invalid.empty:
        raise ValueError(
            "Migrant/employment universes are inconsistent: implied share exceeds 50% "
            f"or numerator exceeds denominator: {invalid.to_dict('records')}"
        )
    return result.sort_values(["industry", "year"]).reset_index(drop=True)


def migrant_mapping_validation(migrants: pd.DataFrame, shares: pd.DataFrame) -> pd.DataFrame:
    matched = shares.loc[shares.industry.eq("matched_productive_industries"), ["year", "migrant_count"]]
    result = migrants[["year", "productive_total", "agriculture", "valid_permits", "grand_total"]].merge(
        matched, on="year", how="left", validate="one_to_one"
    )
    result["matched_fraction_of_productive_migrants"] = result.migrant_count / result.productive_total
    result["unmatched_count"] = result.productive_total - result.migrant_count
    result["unmatched_fraction"] = 1.0 - result.matched_fraction_of_productive_migrants
    result["unmatched_category"] = "agriculture plus classification residual"
    result["count_basis"] = "year-end stock; 2015 onward industry split is valid-permit workers"
    if (result.migrant_count - result.productive_total > 1e-8).any():
        raise ValueError("Matched migrant count exceeds the MOL productive-industry total")
    return result


def manufacturing_native_wage_bounds(
    shares: pd.DataFrame,
    panel: pd.DataFrame,
    minimum_wage: pd.DataFrame,
    fixed_migrant_to_native_ratio: float = 0.8,
) -> pd.DataFrame:
    """Return an interval implied by two explicit migrant-wage scenarios.

    No midpoint or preferred point estimate is produced. The interval endpoints
    correspond to (i) migrant regular pay equal to the year-end monthly minimum
    wage and (ii) migrant regular pay fixed at 0.8 of native regular pay.
    """
    if not 0 < fixed_migrant_to_native_ratio < 1:
        raise ValueError("The fixed migrant/native wage ratio must lie strictly between zero and one")
    manufacturing = panel.loc[
        panel.industry.eq("manufacturing"),
        ["year", "regular_monthly", "cpi_2024_100"],
    ]
    share = shares.loc[
        shares.industry.eq("manufacturing"), ["year", "migrant_share"]
    ]
    result = (
        manufacturing.merge(share, on="year", validate="one_to_one")
        .merge(minimum_wage, on="year", validate="one_to_one")
        .sort_values("year")
        .reset_index(drop=True)
    )
    if result.migrant_share.ge(0.5).any():
        raise ValueError("Manufacturing migrant share reaches the preregistered 50% stop threshold")
    s = result.migrant_share
    wbar = result.regular_monthly
    minimum_scenario = (wbar - s * result.minimum_monthly_wage_twd) / (1.0 - s)
    ratio_scenario = wbar / (1.0 - s * (1.0 - fixed_migrant_to_native_ratio))
    result["native_regular_monthly_lower_twd"] = np.minimum(minimum_scenario, ratio_scenario)
    result["native_regular_monthly_upper_twd"] = np.maximum(minimum_scenario, ratio_scenario)
    result["native_real_regular_monthly_lower_2024_twd"] = (
        result.native_regular_monthly_lower_twd * 100.0 / result.cpi_2024_100
    )
    result["native_real_regular_monthly_upper_2024_twd"] = (
        result.native_regular_monthly_upper_twd * 100.0 / result.cpi_2024_100
    )
    result["observed_real_regular_monthly_2024_twd"] = wbar * 100.0 / result.cpi_2024_100
    result["lower_endpoint_assumption"] = "migrant regular wage = 0.8 × native regular wage"
    result["upper_endpoint_assumption"] = "migrant regular wage = year-end monthly minimum wage"
    result["identified_status"] = "hypothetical bounds; native wage is not identified"
    return result


def manufacturing_contribution_bounds(
    bounds: pd.DataFrame,
    panel: pd.DataFrame,
    start_year: int = 2000,
    end_year: int = 2024,
) -> pd.DataFrame:
    common = set(panel.loc[panel.year.eq(start_year), "industry"]) & set(
        panel.loc[panel.year.eq(end_year), "industry"]
    )
    base = panel.loc[panel.year.eq(start_year) & panel.industry.isin(common)]
    base_share = float(
        base.loc[base.industry.eq("manufacturing"), "employees"].iloc[0] / base.employees.sum()
    )
    start = bounds.loc[bounds.year.eq(start_year)].iloc[0]
    end = bounds.loc[bounds.year.eq(end_year)].iloc[0]
    scenario_changes = [
        end.native_real_regular_monthly_lower_2024_twd
        - start.native_real_regular_monthly_lower_2024_twd,
        end.native_real_regular_monthly_upper_2024_twd
        - start.native_real_regular_monthly_upper_2024_twd,
    ]
    observed_change = (
        end.observed_real_regular_monthly_2024_twd
        - start.observed_real_regular_monthly_2024_twd
    )
    contribution_values = base_share * np.asarray(scenario_changes)
    result = pd.DataFrame(
        [
            {
                "start_year": start_year,
                "end_year": end_year,
                "manufacturing_base_employment_share": base_share,
                "observed_manufacturing_within_contribution_2024_twd": base_share * observed_change,
                "composition_removed_within_contribution_lower_2024_twd": float(contribution_values.min()),
                "composition_removed_within_contribution_upper_2024_twd": float(contribution_values.max()),
                "composition_adjustment_lower_2024_twd": float(contribution_values.min() - base_share * observed_change),
                "composition_adjustment_upper_2024_twd": float(contribution_values.max() - base_share * observed_change),
                "assumptions": "interval spans migrant wage = minimum wage and migrant wage = 0.8 × native wage",
                "identified_status": "hypothetical bounds; no point estimate",
            }
        ]
    )
    return result
