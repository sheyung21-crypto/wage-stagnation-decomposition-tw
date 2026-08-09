from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


PANEL_TO_NATIONAL_ACCOUNTS = {
    "mining": "6",
    "manufacturing": "7",
    "electricity_gas": "33",
    "water_waste": "36",
    "construction": "39",
    "wholesale_retail": "40",
    "transportation_storage": "43",
    "accommodation_food": "49",
    "information_communication": "52",
    "finance_insurance": "56",
    "real_estate": "60",
    "professional_scientific": "63",
    "support_services": "64",
    "education": "68",
    "health_social": "69",
    "arts_recreation": "72",
    "other_services": "73",
}


def read_dgbas_sdmx_json(path: Path) -> pd.DataFrame:
    """Decode the DGBAS compact SDMX-JSON layout into a tidy annual table.

    DGBAS stores the selected field in the series key and flattens the remaining
    industry-by-time grid into the observation index, with industry changing
    fastest. The structure metadata is used to recover every coordinate.
    """
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    data = payload["data"]
    structure = data["structure"]
    series_dimensions = structure["dimensions"]["series"]
    observation_dimensions = structure["dimensions"]["observation"]
    if len(series_dimensions) != 2 or len(observation_dimensions) != 1:
        raise ValueError("Unexpected DGBAS SDMX dimensional structure")
    fields = series_dimensions[0]["values"]
    industries = series_dimensions[1]["values"]
    periods = observation_dimensions[0]["values"]
    n_industries = len(industries)
    expected_observations = n_industries * len(periods)
    records: list[dict[str, object]] = []
    for key, series in data["dataSets"][0]["series"].items():
        field_index = int(key.split(":")[-1])
        if field_index >= len(fields):
            raise ValueError(f"SDMX field index {field_index} is outside the metadata range")
        observations = series["observations"]
        if len(observations) != expected_observations:
            raise ValueError(
                f"Expected {expected_observations} flattened observations, got {len(observations)}"
            )
        for flat_index_text, item in observations.items():
            flat_index = int(flat_index_text)
            year_index, industry_index = divmod(flat_index, n_industries)
            records.append(
                {
                    "year": int(periods[year_index]["id"]),
                    "field_id": fields[field_index]["id"],
                    "field": fields[field_index]["name"],
                    "industry_id": industries[industry_index]["id"],
                    "industry": industries[industry_index]["name"],
                    "value": float(item[0]),
                    "dataset": structure["name"],
                }
            )
    result = pd.DataFrame(records).sort_values(["field_id", "year", "industry_id"]).reset_index(drop=True)
    expected_rows = len(fields) * len(industries) * len(periods)
    if len(result) != expected_rows:
        raise ValueError(f"Decoded row count {len(result)} does not equal expected {expected_rows}")
    return result


def _pivot_field(frame: pd.DataFrame, field_id: str, name: str) -> pd.DataFrame:
    subset = frame.loc[frame.field_id.eq(field_id), ["year", "industry_id", "value"]].copy()
    if subset.empty:
        raise ValueError(f"DGBAS field {field_id} is missing")
    return subset.rename(columns={"value": name})


def _scope_accounting_inputs(
    output: pd.DataFrame,
    compensation: pd.DataFrame,
    panel: pd.DataFrame,
    scope: str,
) -> pd.DataFrame:
    nominal = _pivot_field(output, "3", "nominal_output_million_twd")
    real = _pivot_field(output, "4", "real_output_chain_million_twd")
    comp = _pivot_field(compensation, "6", "employee_compensation_million_twd")
    accounts = nominal.merge(real, on=["year", "industry_id"], validate="one_to_one").merge(
        comp, on=["year", "industry_id"], validate="one_to_one"
    )
    if scope == "manufacturing":
        industry_names = {"manufacturing"}
    elif scope == "industry_services_common_16":
        endpoint_common = set(panel.loc[panel.year.eq(panel.year.min()), "industry"]) & set(
            panel.loc[panel.year.eq(panel.year.max()), "industry"]
        )
        industry_names = endpoint_common
    else:
        raise ValueError(f"Unknown productivity scope: {scope}")
    account_ids = {PANEL_TO_NATIONAL_ACCOUNTS[name] for name in industry_names}
    accounts = accounts.loc[accounts.industry_id.isin(account_ids)]
    account_annual = accounts.groupby("year", as_index=False)[
        ["nominal_output_million_twd", "real_output_chain_million_twd", "employee_compensation_million_twd"]
    ].sum()
    hours = panel.loc[panel.industry.isin(industry_names)].copy()
    hours["employee_hours"] = hours.employees * hours.total_hours * 12.0
    hours_annual = hours.groupby("year", as_index=False).employee_hours.sum()
    result = account_annual.merge(hours_annual, on="year", validate="one_to_one")
    result.insert(0, "scope", scope)
    return result


def productivity_wage_decomposition(
    output: pd.DataFrame,
    compensation: pd.DataFrame,
    panel: pd.DataFrame,
    cpi: pd.DataFrame,
) -> pd.DataFrame:
    """Build the exact employee-compensation accounting decomposition.

    W is national-accounts employee compensation per paid-employee hour. It is
    deliberately not relabelled as the survey's regular wage. Employee
    compensation excludes self-employed mixed income by construction.
    """
    frames = []
    for scope in ("industry_services_common_16", "manufacturing"):
        data = _scope_accounting_inputs(output, compensation, panel, scope)
        data = data.merge(cpi[["year", "cpi_2024_100"]], on="year", validate="one_to_one")
        data["output_price_raw"] = (
            data.nominal_output_million_twd / data.real_output_chain_million_twd
        )
        data["output_price_2024_100"] = (
            data.output_price_raw / data.loc[data.year.eq(2024), "output_price_raw"].iloc[0] * 100.0
        )
        data["employee_compensation_per_hour_twd"] = (
            data.employee_compensation_million_twd * 1_000_000.0 / data.employee_hours
        )
        data["real_consumer_compensation_per_hour_2024_twd"] = (
            data.employee_compensation_per_hour_twd * 100.0 / data.cpi_2024_100
        )
        data["real_output_per_employee_hour_twd"] = (
            data.real_output_chain_million_twd * 1_000_000.0 / data.employee_hours
        )
        data["employee_compensation_share"] = (
            data.employee_compensation_million_twd / data.nominal_output_million_twd
        )
        data["consumer_output_price_ratio"] = data.cpi_2024_100 / data.output_price_2024_100
        for column, index_name in (
            ("real_consumer_compensation_per_hour_2024_twd", "real_consumer_compensation_index_2000_100"),
            ("real_output_per_employee_hour_twd", "labor_productivity_index_2000_100"),
        ):
            data[index_name] = data[column] / data.loc[data.year.eq(2000), column].iloc[0] * 100.0
        data["real_consumer_compensation_log_change"] = np.log(
            data.real_consumer_compensation_per_hour_2024_twd
        ).diff()
        data["labor_productivity_log_change"] = np.log(data.real_output_per_employee_hour_twd).diff()
        data["employee_compensation_share_log_change"] = np.log(data.employee_compensation_share).diff()
        data["price_wedge_log_change"] = np.log(data.consumer_output_price_ratio).diff()
        data["identity_residual"] = data.real_consumer_compensation_log_change - (
            data.labor_productivity_log_change
            + data.employee_compensation_share_log_change
            - data.price_wedge_log_change
        )
        residual = data.identity_residual.dropna().abs().max()
        if residual >= 1e-10:
            raise AssertionError(f"Productivity-wage identity residual {residual} exceeds 1e-10 for {scope}")
        data["labor_share_definition"] = "employee compensation / nominal value added; excludes self-employed mixed income"
        data["real_output_aggregation_note"] = (
            "manufacturing direct series" if scope == "manufacturing" else
            "sum of common-industry chain-volume levels; chain components are not strictly additive"
        )
        data["row_type"] = "annual_path"
        endpoint = {
            "scope": scope,
            "year": 2024,
            "row_type": "endpoint_2000_2024",
            "real_consumer_compensation_log_change": float(
                np.log(
                    data.loc[data.year.eq(2024), "real_consumer_compensation_per_hour_2024_twd"].iloc[0]
                    / data.loc[data.year.eq(2000), "real_consumer_compensation_per_hour_2024_twd"].iloc[0]
                )
            ),
            "labor_productivity_log_change": float(data.labor_productivity_log_change.sum()),
            "employee_compensation_share_log_change": float(data.employee_compensation_share_log_change.sum()),
            "price_wedge_log_change": float(data.price_wedge_log_change.sum()),
            "identity_residual": float(data.identity_residual.fillna(0).sum()),
            "labor_share_definition": "employee compensation / nominal value added; excludes self-employed mixed income",
            "real_output_aggregation_note": data.real_output_aggregation_note.iloc[0],
        }
        endpoint_frame = pd.DataFrame([endpoint]).reindex(columns=data.columns)
        frames.append(pd.concat([data, endpoint_frame], ignore_index=True))
    result = pd.concat(frames, ignore_index=True)
    endpoint = result.loc[result.row_type.eq("endpoint_2000_2024")].copy()
    endpoint["endpoint_identity_check"] = endpoint.real_consumer_compensation_log_change - (
        endpoint.labor_productivity_log_change
        + endpoint.employee_compensation_share_log_change
        - endpoint.price_wedge_log_change
    )
    if endpoint.endpoint_identity_check.abs().max() >= 1e-10:
        raise AssertionError("Endpoint productivity-wage identity does not close below 1e-10")
    return result


def read_official_manufacturing_productivity(path: Path) -> pd.DataFrame:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.reader(stream))
    records = []
    for row in rows:
        if len(row) < 13:
            continue
        match = re.fullmatch(r"(19|20)\d{2}", row[1].strip())
        if match and 2000 <= int(match.group(0)) <= 2024:
            records.append(
                {
                    "year": int(match.group(0)),
                    "official_manufacturing_productivity_2021_100": float(row[12].replace(",", "")),
                }
            )
    result = (
        pd.DataFrame(records)
        .drop_duplicates("year", keep="first")
        .sort_values("year")
        .reset_index(drop=True)
    )
    if len(result) != 25:
        raise ValueError("Official manufacturing productivity series must contain 25 annual observations")
    return result


def read_official_productivity_workbook(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, header=None)
    records = []
    for row in raw.itertuples(index=False, name=None):
        match = re.match(r"(\d{2,3})", str(row[1]).strip()) if len(row) > 3 else None
        if match:
            year = int(match.group(1)) + 1911
            if 2000 <= year <= 2024 and pd.notna(row[3]):
                records.append(
                    {
                        "year": year,
                        "workbook_manufacturing_productivity_2021_100": float(row[3]),
                    }
                )
    result = pd.DataFrame(records).sort_values("year").reset_index(drop=True)
    if len(result) != 25:
        raise ValueError("DGBAS productivity workbook must contain 25 annual observations")
    return result


def productivity_external_validation(
    decomposition: pd.DataFrame,
    official: pd.DataFrame,
    workbook: pd.DataFrame,
    output: pd.DataFrame,
) -> pd.DataFrame:
    manufacturing = decomposition.loc[
        decomposition.scope.eq("manufacturing") & decomposition.row_type.eq("annual_path"),
        ["year", "real_output_per_employee_hour_twd", "identity_residual"],
    ].copy()
    manufacturing["reconstructed_value_added_per_hour_2021_100"] = (
        manufacturing.real_output_per_employee_hour_twd
        / manufacturing.loc[manufacturing.year.eq(2021), "real_output_per_employee_hour_twd"].iloc[0]
        * 100.0
    )
    comparison = manufacturing.merge(official, on="year", validate="one_to_one").merge(
        workbook, on="year", validate="one_to_one"
    )
    comparison["reconstructed_vs_official_relative_error"] = (
        comparison.reconstructed_value_added_per_hour_2021_100
        - comparison.official_manufacturing_productivity_2021_100
    ).abs() / comparison.official_manufacturing_productivity_2021_100
    comparison["workbook_vs_yearbook_relative_error"] = (
        comparison.workbook_manufacturing_productivity_2021_100
        - comparison.official_manufacturing_productivity_2021_100
    ).abs() / comparison.official_manufacturing_productivity_2021_100

    nominal = _pivot_field(output, "3", "nominal")
    real = _pivot_field(output, "4", "real")
    deflator = _pivot_field(output, "5", "reported_deflator")
    deflator_check = nominal.merge(real, on=["year", "industry_id"]).merge(
        deflator, on=["year", "industry_id"]
    )
    deflator_check = deflator_check.loc[deflator_check.industry_id.eq("7")]
    deflator_check["recomputed_deflator"] = deflator_check.nominal / deflator_check.real * 100.0
    deflator_check["relative_error"] = (
        deflator_check.recomputed_deflator - deflator_check.reported_deflator
    ).abs() / deflator_check.reported_deflator

    endpoint_residual = decomposition.loc[
        decomposition.row_type.eq("endpoint_2000_2024"), "identity_residual"
    ].abs().max()
    rows = [
        {
            "check": "official_workbook_vs_MOL_yearbook_productivity",
            "maximum_relative_error": comparison.workbook_vs_yearbook_relative_error.max(),
            "mean_relative_error": comparison.workbook_vs_yearbook_relative_error.mean(),
            "threshold": 1e-12,
            "passed": bool(comparison.workbook_vs_yearbook_relative_error.max() < 1e-12),
            "interpretation": "same official manufacturing productivity index in two dissemination products",
        },
        {
            "check": "reconstructed_value_added_per_hour_vs_official_volume_productivity",
            "maximum_relative_error": comparison.reconstructed_vs_official_relative_error.max(),
            "mean_relative_error": comparison.reconstructed_vs_official_relative_error.mean(),
            "threshold": np.nan,
            "passed": np.nan,
            "interpretation": "reported, not gated: value added/hour and production-volume/hour are different concepts",
        },
        {
            "check": "manufacturing_output_deflator_recomputation",
            "maximum_relative_error": deflator_check.relative_error.max(),
            "mean_relative_error": deflator_check.relative_error.mean(),
            "threshold": 5e-4,
            "passed": bool(deflator_check.relative_error.max() < 5e-4),
            "interpretation": "nominal divided by chain-real output reproduces the published 2021=100 deflator",
        },
        {
            "check": "productivity_wage_identity",
            "maximum_relative_error": endpoint_residual,
            "mean_relative_error": decomposition.identity_residual.dropna().abs().mean(),
            "threshold": 1e-10,
            "passed": bool(endpoint_residual < 1e-10),
            "interpretation": "annual and endpoint log identities close algebraically",
        },
    ]
    result = pd.DataFrame(rows)
    gated = result.loc[result.threshold.notna()]
    if not gated.passed.all():
        raise AssertionError(f"Productivity external-validation gate failed: {gated.loc[~gated.passed].to_dict('records')}")
    return result


def productivity_comparison_path(
    decomposition: pd.DataFrame,
    official: pd.DataFrame,
) -> pd.DataFrame:
    manufacturing = decomposition.loc[
        decomposition.scope.eq("manufacturing") & decomposition.row_type.eq("annual_path")
    ].copy()
    manufacturing["reconstructed_value_added_per_hour_2021_100"] = (
        manufacturing.real_output_per_employee_hour_twd
        / manufacturing.loc[manufacturing.year.eq(2021), "real_output_per_employee_hour_twd"].iloc[0]
        * 100.0
    )
    return manufacturing.merge(official, on="year", validate="one_to_one")
