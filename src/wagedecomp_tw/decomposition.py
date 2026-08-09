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


def complete_path_arrays(
    panel: pd.DataFrame,
    metric: str,
    start: int,
    end: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return a balanced annual path in deterministic year/industry order."""
    subset = panel.loc[
        panel.year.between(start, end), ["year", "industry", "employees", metric]
    ].copy()
    years = np.arange(start, end + 1)
    observed_years = np.sort(subset.year.unique())
    if not np.array_equal(observed_years, years):
        raise ValueError("Chained decomposition requires a complete annual sequence")
    complete = subset.groupby("industry", sort=True).year.nunique()
    industries = complete.index[complete.eq(len(years))].to_numpy()
    if len(industries) == 0:
        raise ValueError("No industries have a complete annual path")
    subset = subset.loc[subset.industry.isin(industries)]
    employment = (
        subset.pivot(index="year", columns="industry", values="employees")
        .reindex(index=years, columns=industries)
        .to_numpy(float)
    )
    wage = (
        subset.pivot(index="year", columns="industry", values=metric)
        .reindex(index=years, columns=industries)
        .to_numpy(float)
    )
    if not np.isfinite(employment).all() or not np.isfinite(wage).all():
        raise ValueError("Balanced path contains missing or non-finite values")
    if np.any(employment <= 0) or np.any(wage <= 0):
        raise ValueError("Chained decomposition requires strictly positive inputs")
    return years, industries, employment, wage


def chained_decomposition(
    panel: pd.DataFrame,
    metric: str,
    start: int,
    end: int,
    method: str = "laspeyres",
) -> pd.DataFrame:
    """Decompose every adjacent-year change and accumulate the annual links."""
    if method not in {"laspeyres", "paasche"}:
        raise ValueError("Chained decomposition supports laspeyres or paasche")
    years, industries, employment, wage = complete_path_arrays(panel, metric, start, end)
    shares = employment / employment.sum(axis=1, keepdims=True)
    ds = np.diff(shares, axis=0)
    dw = np.diff(wage, axis=0)
    cross = np.sum(ds * dw, axis=1)
    if method == "laspeyres":
        annual_within = np.sum(shares[:-1] * dw, axis=1)
        annual_shift = np.sum(wage[:-1] * ds, axis=1)
        annual_interaction = cross
    else:
        annual_within = np.sum(shares[1:] * dw, axis=1)
        annual_shift = np.sum(wage[1:] * ds, axis=1)
        annual_interaction = -cross
    aggregate = np.sum(shares * wage, axis=1)
    annual_total = np.diff(aggregate)
    annual_components = annual_within + annual_shift + annual_interaction
    if not np.allclose(annual_components, annual_total, rtol=1e-10, atol=1e-10):
        raise AssertionError("Annual chained decomposition identity failed")

    zeros = np.zeros(1, dtype=float)
    data = {
        "year": years,
        "annual_within": np.r_[zeros, annual_within],
        "annual_shift": np.r_[zeros, annual_shift],
        "annual_interaction": np.r_[zeros, annual_interaction],
        "annual_total": np.r_[zeros, annual_total],
        "cumulative_within": np.r_[zeros, np.cumsum(annual_within)],
        "cumulative_shift": np.r_[zeros, np.cumsum(annual_shift)],
        "cumulative_interaction": np.r_[zeros, np.cumsum(annual_interaction)],
        "cumulative_total": np.r_[zeros, np.cumsum(annual_total)],
    }
    result = pd.DataFrame(data)
    cumulative_components = (
        result.cumulative_within
        + result.cumulative_shift
        + result.cumulative_interaction
    )
    if not np.allclose(
        cumulative_components, result.cumulative_total, rtol=1e-10, atol=1e-10
    ):
        raise AssertionError("Cumulative chained decomposition identity failed")
    result.insert(0, "n_industries", len(industries))
    result.insert(0, "scale", "2024_twd")
    result.insert(0, "method", method)
    result.insert(0, "metric", metric)
    return result


def institutional_period_decompositions(
    panel: pd.DataFrame,
    metric: str,
    periods: list[tuple[int, int, str, str]],
) -> pd.DataFrame:
    """Compare endpoint and annual-link decompositions for equally weighted periods."""
    rows: list[dict[str, float | int | str | bool]] = []
    for start, end, period, event in periods:
        duration = end - start
        for method in ("laspeyres", "paasche"):
            endpoint = decompose(panel, metric, start, end, method)
            chained = chained_decomposition(panel, metric, start, end, method).iloc[-1]
            row: dict[str, float | int | str | bool] = {
                "period": period,
                "event_marker": event,
                "start_year": start,
                "end_year": end,
                "years": duration,
                "metric": metric,
                "method": method,
                "scale": "2024_twd",
                "n_industries": int(chained.n_industries),
                "equal_weight_period": True,
            }
            for component in ("within", "shift", "interaction", "total"):
                endpoint_value = float(endpoint[component])
                chained_value = float(chained[f"cumulative_{component}"])
                row[f"endpoint_{component}"] = endpoint_value
                row[f"chained_{component}"] = chained_value
                row[f"chained_minus_endpoint_{component}"] = chained_value - endpoint_value
                row[f"endpoint_annualized_{component}"] = endpoint_value / duration
                row[f"chained_annualized_{component}"] = chained_value / duration
            rows.append(row)
    return pd.DataFrame(rows)


def covid_prediction_checks(panel: pd.DataFrame) -> pd.DataFrame:
    """Evaluate the three conditional predictions recorded before Phase 2 calculation."""
    first = decompose(panel, "real_regular_monthly", 2019, 2021, "laspeyres")
    second = decompose(panel, "real_regular_monthly", 2022, 2024, "laspeyres")
    common = set(panel.loc[panel.year.eq(2000), "industry"]) & set(
        panel.loc[panel.year.eq(2024), "industry"]
    )
    subset = panel.loc[panel.industry.isin(common)]

    def aggregate(metric: str, year: int) -> float:
        rows = subset.loc[subset.year.eq(year)]
        return float(np.average(rows[metric], weights=rows.employees))

    gaps: dict[int, float] = {}
    for year in (2019, 2020, 2021):
        monthly = aggregate("real_regular_monthly", year)
        hourly = aggregate("real_regular_hourly", year)
        start_monthly = aggregate("real_regular_monthly", 2000)
        start_hourly = aggregate("real_regular_hourly", 2000)
        gaps[year] = 100.0 * math.log(hourly / start_hourly) - 100.0 * math.log(
            monthly / start_monthly
        )
    records = [
        {
            "prediction_id": "P1",
            "window": "2019-2021",
            "measure": "endpoint_laspeyres_shift_2024_twd",
            "predicted_direction": "positive",
            "estimate": float(first["shift"]),
            "supported": bool(first["shift"] > 0),
        },
        {
            "prediction_id": "P2",
            "window": "2022-2024",
            "measure": "endpoint_laspeyres_shift_2024_twd",
            "predicted_direction": "negative_reversal",
            "estimate": float(second["shift"]),
            "supported": bool(second["shift"] < 0),
        },
        {
            "prediction_id": "P3",
            "window": "2019-2021",
            "measure": "monthly_hourly_growth_gap_change_percentage_points",
            "predicted_direction": "wider_during_2020_2021",
            "estimate": gaps[2021] - gaps[2019],
            "supported": bool(gaps[2020] > gaps[2019] and gaps[2021] > gaps[2020]),
            "gap_2019": gaps[2019],
            "gap_2020": gaps[2020],
            "gap_2021": gaps[2021],
        },
    ]
    result = pd.DataFrame(records)
    result["prediction_status"] = (
        "conditional_ex_ante_researcher_had_seen_full_period_endpoint_results"
    )
    return result


def industry_contributions(
    panel: pd.DataFrame,
    metric: str,
    start: int,
    end: int,
) -> pd.DataFrame:
    """Return industry-level Laspeyres contributions on a common endpoint sample."""
    left = panel.loc[
        panel.year.eq(start), ["industry_code", "industry", "employees", metric]
    ].set_index("industry")
    right = panel.loc[
        panel.year.eq(end), ["industry_code", "industry", "employees", metric]
    ].set_index("industry")
    common = left.index.intersection(right.index)
    left = left.loc[common]
    right = right.loc[common]
    s0 = left.employees / left.employees.sum()
    s1 = right.employees / right.employees.sum()
    w0 = left[metric]
    w1 = right[metric]
    ds = s1 - s0
    dw = w1 - w0
    total = float(np.sum(s1 * w1) - np.sum(s0 * w0))
    result = pd.DataFrame(
        {
            "industry_code": left.industry_code,
            "industry": common,
            "start_share": s0,
            "end_share": s1,
            "start_wage_2024_twd": w0,
            "end_wage_2024_twd": w1,
            "within_2024_twd": s0 * dw,
            "shift_2024_twd": w0 * ds,
            "interaction_2024_twd": ds * dw,
            "industry_total_change_2024_twd": s1 * w1 - s0 * w0,
        }
    ).reset_index(drop=True)
    within_total = float(result.within_2024_twd.sum())
    result["within_share_of_within_percent"] = result.within_2024_twd / within_total * 100.0
    for component in ("within", "shift", "interaction", "industry_total_change"):
        result[f"{component}_share_of_total_percent"] = (
            result[f"{component}_2024_twd"] / total * 100.0
        )
    result = result.sort_values("within_2024_twd", ascending=False).reset_index(drop=True)
    result.insert(0, "within_rank", np.arange(1, len(result) + 1))
    net_top3_share = float(result.head(3).within_2024_twd.sum() / within_total * 100.0)
    concentration = float(
        result.head(3).within_2024_twd.abs().sum()
        / result.within_2024_twd.abs().sum()
        * 100.0
    )
    result["top3_within_concentration_percent"] = concentration
    result["top3_within_share_of_net_within_percent"] = net_top3_share
    result["top3_concentration_exceeds_50_percent"] = concentration > 50.0
    result["metric"] = metric
    result["start_year"] = start
    result["end_year"] = end
    result["method"] = "laspeyres"
    result["scale"] = "2024_twd"
    result["aggregate_total_change_2024_twd"] = total
    endpoint = decompose(panel, metric, start, end, "laspeyres")
    for component in ("within", "shift", "interaction"):
        if not math.isclose(
            float(result[f"{component}_2024_twd"].sum()),
            float(endpoint[component]),
            rel_tol=1e-10,
            abs_tol=1e-10,
        ):
            raise AssertionError(f"Industry {component} contributions do not add up")
    if not math.isclose(
        float(result.industry_total_change_2024_twd.sum()),
        total,
        rel_tol=1e-10,
        abs_tol=1e-10,
    ):
        raise AssertionError("Industry total contributions do not add up")
    return result


def total_regular_industry_difference(
    panel: pd.DataFrame,
    start: int,
    end: int,
) -> pd.DataFrame:
    """Decompose which industries generate the total-minus-regular wage result gap."""
    regular = industry_contributions(panel, "real_regular_monthly", start, end)
    total = industry_contributions(panel, "real_total_monthly", start, end)
    keys = ["industry_code", "industry"]
    columns = [
        "within_2024_twd",
        "shift_2024_twd",
        "interaction_2024_twd",
        "industry_total_change_2024_twd",
    ]
    merged = total[keys + columns].merge(
        regular[keys + columns], on=keys, suffixes=("_total_wage", "_regular_wage")
    )
    for column in columns:
        merged[f"{column}_difference"] = (
            merged[f"{column}_total_wage"] - merged[f"{column}_regular_wage"]
        )
    merged["row_type"] = "industry_total_minus_regular"
    merged["start_year"] = start
    merged["end_year"] = end
    merged["method"] = "laspeyres"
    merged["scale"] = "2024_twd"
    return merged.sort_values("within_2024_twd_difference", ascending=False).reset_index(drop=True)


def hours_mechanism_table(panel: pd.DataFrame, start: int = 2000, end: int = 2024) -> pd.DataFrame:
    """Build annual normal/overtime paths and the 2017 reform-window comparison."""
    subset = panel.loc[panel.year.between(start, end)].copy()
    counts = subset.groupby("industry").year.nunique()
    common = counts.index[counts.eq(end - start + 1)]
    subset = subset.loc[subset.industry.isin(common)].copy()
    subset["normal_hours"] = subset.regular_hours
    subset["overtime_hours"] = subset.total_hours - subset.regular_hours
    subset["employment_share_common"] = subset.employees / subset.groupby("year").employees.transform("sum")
    annual = subset[
        [
            "year",
            "industry_code",
            "industry",
            "employees",
            "employment_share_common",
            "normal_hours",
            "overtime_hours",
        ]
    ].copy()
    annual.insert(0, "row_type", "annual_industry_path")

    aggregate = (
        subset.assign(
            weighted_normal=subset.normal_hours * subset.employees,
            weighted_overtime=subset.overtime_hours * subset.employees,
        )
        .groupby("year", as_index=False)
        .agg(
            employees=("employees", "sum"),
            weighted_normal=("weighted_normal", "sum"),
            weighted_overtime=("weighted_overtime", "sum"),
        )
    )
    aggregate["normal_hours"] = aggregate.weighted_normal / aggregate.employees
    aggregate["overtime_hours"] = aggregate.weighted_overtime / aggregate.employees
    aggregate["industry_code"] = "ALL"
    aggregate["industry"] = "common_16_weighted_aggregate"
    aggregate["employment_share_common"] = 1.0
    aggregate.insert(0, "row_type", "annual_aggregate_path")
    aggregate = aggregate[annual.columns]

    wide_normal = subset.pivot(index="industry", columns="year", values="normal_hours")
    wide_overtime = subset.pivot(index="industry", columns="year", values="overtime_hours")
    codes = subset.drop_duplicates("industry").set_index("industry").industry_code
    comparison = pd.DataFrame(
        {
            "industry": common,
            "industry_code": codes.loc[common].values,
            "normal_pretrend_annual_change_2000_2016": (
                wide_normal.loc[common, 2016] - wide_normal.loc[common, 2000]
            ).to_numpy()
            / 16.0,
            "normal_reform_window_annual_change_2016_2019": (
                wide_normal.loc[common, 2019] - wide_normal.loc[common, 2016]
            ).to_numpy()
            / 3.0,
            "overtime_pretrend_annual_change_2000_2016": (
                wide_overtime.loc[common, 2016] - wide_overtime.loc[common, 2000]
            ).to_numpy()
            / 16.0,
            "overtime_reform_window_annual_change_2016_2019": (
                wide_overtime.loc[common, 2019] - wide_overtime.loc[common, 2016]
            ).to_numpy()
            / 3.0,
        }
    )
    comparison["normal_change_vs_pretrend"] = (
        comparison.normal_reform_window_annual_change_2016_2019
        - comparison.normal_pretrend_annual_change_2000_2016
    )
    comparison["overtime_change_vs_pretrend"] = (
        comparison.overtime_reform_window_annual_change_2016_2019
        - comparison.overtime_pretrend_annual_change_2000_2016
    )
    comparison.insert(0, "row_type", "reform_window_comparison")
    aggregate_indexed = aggregate.set_index("year")
    aggregate_comparison = pd.DataFrame(
        [
            {
                "row_type": "reform_window_comparison",
                "industry_code": "ALL",
                "industry": "common_16_weighted_aggregate",
                "normal_pretrend_annual_change_2000_2016": (
                    aggregate_indexed.loc[2016, "normal_hours"]
                    - aggregate_indexed.loc[2000, "normal_hours"]
                )
                / 16.0,
                "normal_reform_window_annual_change_2016_2019": (
                    aggregate_indexed.loc[2019, "normal_hours"]
                    - aggregate_indexed.loc[2016, "normal_hours"]
                )
                / 3.0,
                "overtime_pretrend_annual_change_2000_2016": (
                    aggregate_indexed.loc[2016, "overtime_hours"]
                    - aggregate_indexed.loc[2000, "overtime_hours"]
                )
                / 16.0,
                "overtime_reform_window_annual_change_2016_2019": (
                    aggregate_indexed.loc[2019, "overtime_hours"]
                    - aggregate_indexed.loc[2016, "overtime_hours"]
                )
                / 3.0,
            }
        ]
    )
    aggregate_comparison["normal_change_vs_pretrend"] = (
        aggregate_comparison.normal_reform_window_annual_change_2016_2019
        - aggregate_comparison.normal_pretrend_annual_change_2000_2016
    )
    aggregate_comparison["overtime_change_vs_pretrend"] = (
        aggregate_comparison.overtime_reform_window_annual_change_2016_2019
        - aggregate_comparison.overtime_pretrend_annual_change_2000_2016
    )

    availability = pd.DataFrame(
        [
            {
                "row_type": "part_time_availability_audit",
                "industry": "industry_and_services_total",
                "part_time_series_requested_period": "2000-2024",
                "part_time_series_consistent_coverage": False,
                "part_time_series_used": False,
                "part_time_series_reason": "official_query_returned_no_consistent_annual_values_for_2000_2024",
            }
        ]
    )
    return pd.concat(
        [annual, aggregate, comparison, aggregate_comparison, availability],
        ignore_index=True,
        sort=False,
    )


def official_comparison_table(
    panel: pd.DataFrame,
    official_nominal: pd.DataFrame,
    official_published_real: pd.DataFrame,
    start: int = 2000,
    end: int = 2024,
) -> pd.DataFrame:
    """Compare official published real series with the balanced analysis sample."""
    common = set(panel.loc[panel.year.eq(start), "industry"]) & set(
        panel.loc[panel.year.eq(end), "industry"]
    )
    rows: list[dict[str, float | int | str | bool]] = []
    summaries: list[dict[str, float | int | str | bool]] = []
    published = official_published_real.set_index("year")
    official = official_nominal.set_index("year")
    for composition in ("regular", "total"):
        nominal_metric = f"{composition}_monthly"
        real_metric = f"real_{nominal_metric}"
        published_metric = f"official_real_{composition}_monthly"
        common_path: dict[int, float] = {}
        full_path: dict[int, float] = {}
        official_deflated_path: dict[int, float] = {}
        published_path: dict[int, float] = {}
        scale_factor = float(official.loc[end, f"official_{nominal_metric}"]) / float(
            published.loc[end, published_metric]
        )
        for year in range(start, end + 1):
            year_rows = panel.loc[panel.year.eq(year)]
            common_rows = year_rows.loc[year_rows.industry.isin(common)]
            common_value = float(np.average(common_rows[real_metric], weights=common_rows.employees))
            full_value = float(np.average(year_rows[real_metric], weights=year_rows.employees))
            cpi_factor = float(100.0 / year_rows.cpi_2024_100.iloc[0])
            official_deflated = float(official.loc[year, f"official_{nominal_metric}"]) * cpi_factor
            published_value = float(published.loc[year, published_metric]) * scale_factor
            common_path[year] = common_value
            full_path[year] = full_value
            official_deflated_path[year] = official_deflated
            published_path[year] = published_value
            rows.append(
                {
                    "row_type": "annual_sequence",
                    "composition": composition,
                    "year": year,
                    "common_sample_real_2024_twd": common_value,
                    "full_panel_real_2024_twd": full_value,
                    "official_nominal_deflated_2024_twd": official_deflated,
                    "official_published_real_original_base": float(published.loc[year, published_metric]),
                    "official_published_real_rebased_2024_twd": published_value,
                    "official_minus_common_2024_twd": published_value - common_value,
                }
            )

        def growth(path: dict[int, float]) -> float:
            return 100.0 * math.log(path[end] / path[start])

        common_growth = growth(common_path)
        full_growth = growth(full_path)
        official_deflated_growth = growth(official_deflated_path)
        published_growth = growth(published_path)
        observed_gap = published_growth - common_growth
        effects = {
            "common_sample_effect": full_growth - common_growth,
            "coverage_aggregation_effect": official_deflated_growth - full_growth,
            "cpi_vintage_effect": published_growth - official_deflated_growth,
        }
        residual = observed_gap - sum(effects.values())
        denominator = abs(observed_gap) if not math.isclose(observed_gap, 0.0) else np.nan
        for source, value in {**effects, "residual": residual}.items():
            share = abs(value) / denominator * 100.0 if np.isfinite(denominator) else np.nan
            summaries.append(
                {
                    "row_type": "growth_gap_decomposition",
                    "composition": composition,
                    "start_year": start,
                    "end_year": end,
                    "source": source,
                    "effect_percentage_points": value,
                    "share_of_absolute_observed_gap_percent": share,
                    "observed_official_minus_common_growth_gap_percentage_points": observed_gap,
                    "official_published_log_growth_percent": published_growth,
                    "common_sample_log_growth_percent": common_growth,
                    "residual_exceeds_30_percent": bool(source == "residual" and share > 30.0),
                }
            )
    return pd.concat([pd.DataFrame(rows), pd.DataFrame(summaries)], ignore_index=True, sort=False)


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
