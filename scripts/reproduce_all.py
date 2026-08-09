from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wagedecomp_tw.counterfactual import frozen_share_path
from wagedecomp_tw.decomposition import (
    chained_decomposition,
    covid_prediction_checks,
    decompose,
    hours_mechanism_table,
    industry_contributions,
    institutional_period_decompositions,
    nominal_then_deflate_bridge,
    official_comparison_table,
    total_regular_industry_difference,
)
from wagedecomp_tw.inference import block_bootstrap_chained_paths, block_bootstrap_decomposition
from wagedecomp_tw.ingest import (
    build_major_panel,
    build_middle_panel,
    build_source_manifest,
    build_vintage_major_panel,
    read_annual_cpi,
    read_official_real_wage,
)
from wagedecomp_tw.migrant import (
    manufacturing_contribution_bounds,
    manufacturing_native_wage_bounds,
    migrant_mapping_validation,
    migrant_share_paths,
    read_migrant_counts,
    read_minimum_monthly_wage,
)
from wagedecomp_tw.panel import aggregate_wage, common_industry_path, enrich_panel
from wagedecomp_tw.productivity import (
    productivity_comparison_path,
    productivity_external_validation,
    productivity_wage_decomposition,
    read_dgbas_sdmx_json,
    read_official_manufacturing_productivity,
    read_official_productivity_workbook,
)
from wagedecomp_tw.provenance import write_results_manifest


RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
METRICS = ["regular_monthly", "total_monthly", "regular_hourly", "total_hourly"]
REAL_METRICS = [f"real_{metric}" for metric in METRICS]
METHODS = ["laspeyres", "paasche", "tornqvist"]


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.10f", lineterminator="\n")


def build_paper() -> Path:
    """Compile the manuscript and repair its searchable Unicode mapping."""
    paper = ROOT / "paper"
    final_pdf = paper / "wage_stagnation_decomposition_tw.pdf"
    configured_tectonic = os.environ.get("TECTONIC")
    xelatex = shutil.which("xelatex")
    tectonic = configured_tectonic or shutil.which("tectonic")
    if xelatex:
        command = [xelatex, "-interaction=nonstopmode", "-halt-on-error", "main.tex"]
        subprocess.run(command, cwd=paper, check=True)
        subprocess.run(command, cwd=paper, check=True)
    elif tectonic and Path(tectonic).is_file():
        subprocess.run([tectonic, "--keep-logs", "main.tex"], cwd=paper, check=True)
    else:
        raise RuntimeError(
            "No TeX engine found. Install XeLaTeX or Tectonic, or set TECTONIC to the Tectonic executable."
        )
    shutil.copy2(paper / "main.pdf", final_pdf)
    subprocess.run([sys.executable, str(paper / "fixtounicode.py"), str(final_pdf)], cwd=paper, check=True)
    return final_pdf


def external_checks(
    major: pd.DataFrame,
    major_official: pd.DataFrame,
    vintage: pd.DataFrame,
    vintage_official: pd.DataFrame,
    middle: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for label, panel, official in (
        ("latest_major_vs_official", major, major_official),
        ("same_vintage_major_vs_official", vintage, vintage_official),
    ):
        for metric in ["employees", "regular_monthly", "total_monthly", "regular_hours", "total_hours"]:
            computed = panel.groupby("year").employees.sum() if metric == "employees" else aggregate_wage(panel, metric)
            reference = official.set_index("year")[f"official_{metric}"]
            comparison = pd.DataFrame({"computed": computed, "reference": reference}).dropna()
            error = (comparison.computed - comparison.reference).abs() / comparison.reference.abs()
            rows.append(
                {
                    "check": label,
                    "metric": metric,
                    "maximum_relative_error": error.max(),
                    "year_of_maximum": int(error.idxmax()),
                    "mean_relative_error": error.mean(),
                    "periods_checked": len(error),
                    "threshold": 0.005,
                    "passed": bool(error.max() < 0.005),
                }
            )
    for metric in ["employees", "regular_monthly", "total_monthly", "regular_hours", "total_hours"]:
        computed = middle.groupby("year").employees.sum() if metric == "employees" else aggregate_wage(middle, metric)
        reference = vintage.groupby("year").employees.sum() if metric == "employees" else aggregate_wage(vintage, metric)
        comparison = pd.DataFrame({"computed": computed, "reference": reference}).dropna()
        error = (comparison.computed - comparison.reference).abs() / comparison.reference.abs()
        rows.append(
            {
                "check": "middle_vs_same_vintage_major",
                "metric": metric,
                "maximum_relative_error": error.max(),
                "year_of_maximum": int(error.idxmax()),
                "mean_relative_error": error.mean(),
                "periods_checked": len(error),
                "threshold": 0.005,
                "passed": bool(error.max() < 0.005),
            }
        )
    result = pd.DataFrame(rows)
    if not result.passed.all():
        raise AssertionError(f"External consistency gate failed: {result.loc[~result.passed].to_dict('records')}")
    return result


def aggregate_value(panel: pd.DataFrame, metric: str, year: int, common: set[str] | None = None) -> float:
    rows = panel.loc[panel.year.eq(year)]
    if common is not None:
        rows = rows.loc[rows.industry.isin(common)]
    return float(np.average(rows[metric], weights=rows.employees))


def main_tables(
    major: pd.DataFrame,
    vintage: pd.DataFrame,
    middle: pd.DataFrame,
    major_official: pd.DataFrame,
    official_real: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    main = [decompose(major, metric, 2000, 2024, method) for metric in REAL_METRICS for method in METHODS]
    bridge = nominal_then_deflate_bridge(major, "regular_monthly", 2000, 2024)
    for row in main:
        row["deflation_order"] = "deflate_then_decompose"
        row["price"] = 0.0
    main.append(
        {
            "start_year": 2000,
            "end_year": 2024,
            "metric": "regular_monthly",
            "method": "laspeyres",
            "scale": "2024_twd",
            "n_industries": 16,
            "deflation_order": "decompose_then_deflate_bridge",
            "residual": 0.0,
            **bridge,
        }
    )
    tables["table_03_main_decomposition.csv"] = pd.DataFrame(main)

    sensitivity = []
    for start in (2000, 2008, 2016):
        for metric in REAL_METRICS:
            for method in METHODS:
                sensitivity.append(decompose(major, metric, start, 2024, method))
    tables["table_07_period_sensitivity.csv"] = pd.DataFrame(sensitivity)

    grain_rows = []
    for start, end, regime in ((2016, 2020, "TSIC10"), (2021, 2024, "TSIC11")):
        start_set = set(middle.loc[middle.year.eq(start), "industry"])
        end_set = set(middle.loc[middle.year.eq(end), "industry"])
        common_middle = start_set & end_set
        common_panel = middle.loc[middle.industry.isin(common_middle) & middle.year.between(start, end)].copy()
        common_panel["weighted_wage"] = common_panel.employees * common_panel.real_regular_monthly
        coarse = common_panel.groupby(["year", "table_group"], as_index=False).agg(
            employees=("employees", "sum"), weighted_wage=("weighted_wage", "sum")
        )
        coarse["real_regular_monthly"] = coarse.weighted_wage / coarse.employees
        coarse["industry"] = coarse.table_group.map(lambda value: f"major_group_{int(value):02d}")
        for method in METHODS:
            major_result = decompose(coarse, "real_regular_monthly", start, end, method)
            middle_result = decompose(common_panel, "real_regular_monthly", start, end, method)
            for grain, result in (("major", major_result), ("middle", middle_result)):
                grain_rows.append({"regime": regime, "grain": grain, **result})
    tables["table_04_granularity_comparison.csv"] = pd.DataFrame(grain_rows)

    common = set(major.loc[major.year.eq(2000), "industry"]) & set(major.loc[major.year.eq(2024), "industry"])
    monthly_hourly = []
    for composition in ("regular", "total"):
        monthly = f"real_{composition}_monthly"
        hourly = f"real_{composition}_hourly"
        w0m, w1m = (aggregate_value(major, monthly, year, common) for year in (2000, 2024))
        w0h, w1h = (aggregate_value(major, hourly, year, common) for year in (2000, 2024))
        monthly_log = 100.0 * np.log(w1m / w0m)
        hourly_log = 100.0 * np.log(w1h / w0h)
        monthly_hourly.append(
            {
                "composition": composition,
                "start_year": 2000,
                "end_year": 2024,
                "monthly_log_growth_percent": monthly_log,
                "hourly_log_growth_percent": hourly_log,
                "hours_accounting_gap_percentage_points": hourly_log - monthly_log,
                "monthly_level_growth_percent": (w1m / w0m - 1.0) * 100.0,
                "hourly_level_growth_percent": (w1h / w0h - 1.0) * 100.0,
            }
        )
    tables["table_05_monthly_hourly.csv"] = pd.DataFrame(monthly_hourly)

    composition_rows = []
    for frequency in ("monthly", "hourly"):
        for composition in ("regular", "total"):
            metric = f"real_{composition}_{frequency}"
            w0, w1 = (aggregate_value(major, metric, year, common) for year in (2000, 2024))
            composition_rows.append(
                {
                    "frequency": frequency,
                    "composition": composition,
                    "start_value_2024_twd": w0,
                    "end_value_2024_twd": w1,
                    "level_change_2024_twd": w1 - w0,
                    "log_growth_percent": 100.0 * np.log(w1 / w0),
                }
            )
    tables["table_06_regular_total.csv"] = pd.DataFrame(composition_rows)

    counterfactual = pd.concat(
        [frozen_share_path(major, "real_regular_monthly", base) for base in (2000, 2008, 2016)],
        ignore_index=True,
    )
    tables["table_08_counterfactual.csv"] = counterfactual

    chained_frames = []
    for method in ("laspeyres", "paasche"):
        chained = chained_decomposition(major, "real_regular_monthly", 2000, 2024, method)
        bootstrap = block_bootstrap_chained_paths(
            major,
            "real_regular_monthly",
            2000,
            2024,
            method=method,
            replications=10000,
            block_length=3,
            seed=20260809,
        )
        for component in ("within", "shift", "interaction", "total"):
            interval = bootstrap.loc[bootstrap.component.eq(component)].set_index("year")
            for field in ("estimate_mean", "ci_lower", "ci_upper", "p_value_two_sided"):
                chained[f"cumulative_{component}_bootstrap_{field}"] = chained.year.map(interval[field])
        chained["valid_replications"] = 10000
        chained["seed"] = 20260809
        chained["block_length"] = 3
        chained["total_ci_width_exceeds_abs_cumulative_change"] = (
            chained.cumulative_total_bootstrap_ci_upper
            - chained.cumulative_total_bootstrap_ci_lower
        ) > chained.cumulative_total.abs()
        chained_frames.append(chained)
    chained_table = pd.concat(chained_frames, ignore_index=True)
    tables["table_10_chained_decomposition.csv"] = chained_table

    comparison_rows = []
    endpoint_results = {
        method: decompose(major, "real_regular_monthly", 2000, 2024, method)
        for method in ("laspeyres", "paasche")
    }
    chained_results = {
        method: chained_table.loc[chained_table.method.eq(method)].iloc[-1]
        for method in ("laspeyres", "paasche")
    }
    endpoint_method_gap = abs(
        float(endpoint_results["laspeyres"]["within"])
        - float(endpoint_results["paasche"]["within"])
    )
    chained_method_gap = abs(
        float(chained_results["laspeyres"].cumulative_within)
        - float(chained_results["paasche"].cumulative_within)
    )
    for method in ("laspeyres", "paasche"):
        endpoint = endpoint_results[method]
        chained = chained_results[method]
        row = {
            "metric": "real_regular_monthly",
            "start_year": 2000,
            "end_year": 2024,
            "method": method,
            "scale": "2024_twd",
            "endpoint_laspeyres_paasche_within_gap_2024_twd": endpoint_method_gap,
            "chained_laspeyres_paasche_within_gap_2024_twd": chained_method_gap,
            "method_gap_reduction_percent": (1.0 - chained_method_gap / endpoint_method_gap) * 100.0,
        }
        for component in ("within", "shift", "interaction", "total"):
            endpoint_value = float(endpoint[component])
            chained_value = float(chained[f"cumulative_{component}"])
            row[f"endpoint_{component}"] = endpoint_value
            row[f"chained_{component}"] = chained_value
            row[f"chained_minus_endpoint_{component}"] = chained_value - endpoint_value
        row["interaction_absolute_shrinkage_2024_twd"] = abs(float(endpoint["interaction"])) - abs(
            float(chained.cumulative_interaction)
        )
        row["interaction_absolute_shrinkage_percent"] = (
            1.0 - abs(float(chained.cumulative_interaction)) / abs(float(endpoint["interaction"]))
        ) * 100.0
        comparison_rows.append(row)
    tables["table_11_chained_vs_endpoint.csv"] = pd.DataFrame(comparison_rows)

    periods = [
        (2000, 2002, "2000-2002", "WTO_accession_2002"),
        (2002, 2009, "2002-2009", "post_global_financial_crisis_2009"),
        (2009, 2017, "2009-2017", "one_fixed_day_off_first_stage_2017"),
        (2017, 2020, "2017-2020", "covid_19_onset_2020"),
        (2020, 2022, "2020-2022", "covid_19_to_reopening"),
        (2022, 2024, "2022-2024", "post_covid_normalization"),
    ]
    tables["table_12_institutional_periods.csv"] = institutional_period_decompositions(
        major, "real_regular_monthly", periods
    )
    tables["table_13_covid_predictions.csv"] = covid_prediction_checks(major)
    tables["table_14_industry_contributions.csv"] = industry_contributions(
        major, "real_regular_monthly", 2000, 2024
    )

    total_aggregate = pd.DataFrame(
        [decompose(major, "real_total_monthly", 2000, 2024, method) for method in METHODS]
    )
    total_aggregate.insert(0, "row_type", "aggregate_total_wage")
    total_industry_difference = total_regular_industry_difference(major, 2000, 2024)
    tables["table_15_total_wage_decomposition.csv"] = pd.concat(
        [total_aggregate, total_industry_difference], ignore_index=True, sort=False
    )

    total_sensitivity = []
    for start in (2000, 2001, 2002):
        for end in (2022, 2023, 2024):
            for method in METHODS:
                total_sensitivity.append(decompose(major, "real_total_monthly", start, end, method))
    tables["table_16_total_wage_endpoint_sensitivity.csv"] = pd.DataFrame(total_sensitivity)
    tables["table_17_hours_mechanism.csv"] = hours_mechanism_table(major)
    tables["table_18_official_comparison.csv"] = official_comparison_table(
        major, major_official, official_real
    )
    return tables


def phase3_tables(major: pd.DataFrame, cpi: pd.DataFrame) -> dict[str, pd.DataFrame]:
    migrants = read_migrant_counts(RAW / "mol_foreign_workers_by_work.csv")
    minimum_wage = read_minimum_monthly_wage(RAW / "mol_major_economic_indicators.csv")
    shares = migrant_share_paths(migrants, major)
    mapping = migrant_mapping_validation(migrants, shares)

    wage_paths = []
    for industry in ("manufacturing", "construction", "water_waste"):
        data = major.loc[
            major.industry.eq(industry), ["year", "industry", "real_regular_monthly"]
        ].copy()
        data["real_regular_wage_index_2000_100"] = (
            data.real_regular_monthly
            / data.loc[data.year.eq(2000), "real_regular_monthly"].iloc[0]
            * 100.0
        )
        wage_paths.append(data)
    matched = major.loc[
        major.industry.isin({"manufacturing", "construction", "water_waste"})
    ].copy()
    matched["payroll"] = matched.employees * matched.real_regular_monthly
    matched = matched.groupby("year", as_index=False).agg(
        employees=("employees", "sum"), payroll=("payroll", "sum")
    )
    matched["industry"] = "matched_productive_industries"
    matched["real_regular_monthly"] = matched.payroll / matched.employees
    matched["real_regular_wage_index_2000_100"] = (
        matched.real_regular_monthly
        / matched.loc[matched.year.eq(2000), "real_regular_monthly"].iloc[0]
        * 100.0
    )
    wage_paths.append(
        matched[["year", "industry", "real_regular_monthly", "real_regular_wage_index_2000_100"]]
    )
    migrant_paths = shares.merge(
        pd.concat(wage_paths, ignore_index=True),
        on=["year", "industry"],
        how="left",
        validate="one_to_one",
    )
    migrant_paths["row_type"] = "industry_path"
    mapping = mapping.copy()
    mapping["row_type"] = "mapping_validation"
    table19 = pd.concat([migrant_paths, mapping], ignore_index=True, sort=False)

    bounds = manufacturing_native_wage_bounds(shares, major, minimum_wage)
    bounds["row_type"] = "annual_hypothetical_bounds"
    contribution = manufacturing_contribution_bounds(bounds, major)
    contribution["row_type"] = "endpoint_contribution_bounds"
    table20 = pd.concat([bounds, contribution], ignore_index=True, sort=False)

    output = read_dgbas_sdmx_json(RAW / "dgbas_national_accounts_output.json")
    compensation = read_dgbas_sdmx_json(
        RAW / "dgbas_national_accounts_compensation.json"
    )
    productivity = productivity_wage_decomposition(output, compensation, major, cpi)
    common = set(major.loc[major.year.eq(2000), "industry"]) & set(
        major.loc[major.year.eq(2024), "industry"]
    )
    survey_rows = []
    for scope, industries in (
        ("industry_services_common_16", common),
        ("manufacturing", {"manufacturing"}),
    ):
        data = major.loc[major.industry.isin(industries)].copy()
        data["payroll"] = data.employees * data.real_regular_hourly
        annual = data.groupby("year", as_index=False).agg(
            employees=("employees", "sum"), payroll=("payroll", "sum")
        )
        annual["scope"] = scope
        annual["survey_real_regular_hourly_index_2000_100"] = (
            annual.payroll / annual.employees
        ) / (
            annual.loc[annual.year.eq(2000), "payroll"].iloc[0]
            / annual.loc[annual.year.eq(2000), "employees"].iloc[0]
        ) * 100.0
        survey_rows.append(
            annual[["scope", "year", "survey_real_regular_hourly_index_2000_100"]]
        )
    productivity = productivity.merge(
        pd.concat(survey_rows, ignore_index=True),
        on=["scope", "year"],
        how="left",
        validate="many_to_one",
    )
    productivity["index_base_note"] = "CPI and output-price indexes both rebased to 2024=100 before ratios"

    official_productivity = read_official_manufacturing_productivity(
        RAW / "mol_major_economic_indicators.csv"
    )
    workbook_productivity = read_official_productivity_workbook(
        RAW / "dgbas_manufacturing_labor_productivity.xlsx"
    )
    productivity_validation = productivity_external_validation(
        productivity, official_productivity, workbook_productivity, output
    )
    productivity_validation["row_type"] = "validation_summary"
    productivity_path = productivity_comparison_path(productivity, official_productivity)
    productivity_path["row_type"] = "manufacturing_comparison_path"
    table22 = pd.concat([productivity_validation, productivity_path], ignore_index=True, sort=False)

    return {
        "table_19_migrant_trends_validation.csv": table19,
        "table_20_migrant_hypothetical_bounds.csv": table20,
        "table_21_productivity_decomposition.csv": productivity,
        "table_22_productivity_validation.csv": table22,
    }


def save_figure(fig: plt.Figure, filename: str) -> None:
    path = FIGURES / filename
    fig.savefig(path, bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None, "Creator": "wagedecomp_tw"})
    plt.close(fig)


def figures(major: pd.DataFrame, vintage: pd.DataFrame, middle: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> None:
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    path_month = common_industry_path(major, "real_regular_monthly", 2000)
    path_hour = common_industry_path(major, "real_regular_hourly", 2000)
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    for data, label, style, color in ((path_month, "Real monthly regular earnings", "-", "0.15"), (path_hour, "Real hourly regular earnings", "--", "0.55")):
        indexed = data.value / data.loc[data.year.eq(2000), "value"].iloc[0] * 100.0
        ax.plot(data.year, indexed, style, color=color, linewidth=1.8, label=label)
    ax.axhline(100, color="0.7", linewidth=0.7)
    ax.set(ylabel="Index (2000=100)", xlabel="Year")
    ax.legend(frameon=False)
    save_figure(fig, "figure_01_real_monthly_hourly.pdf")

    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    for metric, label, style, color in (("real_regular_monthly", "Regular", "-", "0.15"), ("real_total_monthly", "Total", "--", "0.55")):
        data = common_industry_path(major, metric, 2000)
        indexed = data.value / data.loc[data.year.eq(2000), "value"].iloc[0] * 100.0
        ax.plot(data.year, indexed, style, color=color, linewidth=1.8, label=label)
    ax.set(ylabel="Index (2000=100)", xlabel="Year")
    ax.legend(frameon=False)
    save_figure(fig, "figure_02_regular_total.pdf")

    shares = major.assign(share=major.employees / major.groupby("year").employees.transform("sum"))
    top = shares.groupby("industry").employees.sum().nlargest(6).index
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for industry, color in zip(top, np.linspace(0.1, 0.75, len(top))):
        data = shares.loc[shares.industry.eq(industry)]
        ax.plot(data.year, data.share * 100.0, color=str(color), linewidth=1.4, label=industry.replace("_", " "))
    for seam in (2006, 2011, 2016, 2021):
        ax.axvline(seam, color="0.75", linewidth=0.6, linestyle=":")
    ax.set(ylabel="Employment share (%)", xlabel="Year")
    ax.legend(frameon=False, ncol=2, fontsize=7)
    save_figure(fig, "figure_03_employment_shares.pdf")

    decomp = tables["table_03_main_decomposition.csv"]
    decomp = decomp.loc[(decomp.metric == "real_regular_monthly") & (decomp.deflation_order == "deflate_then_decompose")]
    components = ["within", "shift", "interaction", "residual"]
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    x = np.arange(len(decomp))
    bottom = np.zeros(len(decomp))
    for component, shade in zip(components, ("0.2", "0.45", "0.65", "0.82")):
        values = np.where(decomp.total.to_numpy() != 0, decomp[component].to_numpy() / decomp.total.to_numpy() * 100.0, np.nan)
        ax.bar(x, values, bottom=bottom, label=component, color=shade)
        bottom += np.nan_to_num(values)
    ax.set_xticks(x, decomp.method)
    ax.set_ylabel("Share of total change (%)")
    ax.legend(frameon=False, ncol=4, fontsize=7)
    save_figure(fig, "figure_04_decomposition_methods.pdf")

    grain = tables["table_04_granularity_comparison.csv"]
    grain = grain.loc[grain.method.eq("laspeyres")]
    labels = [f"{r.regime}\n{r.grain}" for r in grain.itertuples()]
    x = np.arange(len(grain))
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    ax.bar(x - 0.18, grain.within / grain.total * 100.0, 0.36, color="0.25", label="within")
    ax.bar(x + 0.18, grain["shift"] / grain.total * 100.0, 0.36, color="0.65", label="shift")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Share of total change (%)")
    ax.legend(frameon=False)
    save_figure(fig, "figure_05_granularity.pdf")

    cf = tables["table_08_counterfactual.csv"]
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    actual = cf.loc[cf.base_year.eq(2000)]
    ax.plot(actual.year, actual.actual, color="0.1", linewidth=1.8, label="Actual")
    for base, style, color in ((2000, "--", "0.35"), (2008, "-.", "0.55"), (2016, ":", "0.75")):
        data = cf.loc[cf.base_year.eq(base)]
        ax.plot(data.year, data.counterfactual, style, color=color, linewidth=1.4, label=f"Frozen {base} shares")
    ax.set(ylabel="2024 TWD per month", xlabel="Year")
    ax.legend(frameon=False, fontsize=8)
    save_figure(fig, "figure_06_counterfactual.pdf")

    chained = tables["table_10_chained_decomposition.csv"]
    chained = chained.loc[chained.method.eq("laspeyres")]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8), sharex=True)
    for ax, component, title in zip(
        axes.ravel(),
        ("within", "shift", "interaction", "total"),
        ("Within-industry", "Employment-share shift", "Interaction", "Total change"),
    ):
        estimate = chained[f"cumulative_{component}"].to_numpy(float)
        lower = chained[f"cumulative_{component}_bootstrap_ci_lower"].to_numpy(float)
        upper = chained[f"cumulative_{component}_bootstrap_ci_upper"].to_numpy(float)
        ax.fill_between(
            chained.year,
            lower,
            upper,
            color="0.85",
            linewidth=0,
            label="95% path-resampling interval",
        )
        ax.plot(chained.year, estimate, color="0.15", linewidth=1.6, label="Observed cumulative path")
        ax.axhline(0, color="0.6", linewidth=0.7)
        ax.set_title(title, fontsize=9)
        ax.set_ylabel("2024 TWD")
    for ax in axes[-1]:
        ax.set_xlabel("Year")
    axes[0, 0].legend(frameon=False, fontsize=7)
    save_figure(fig, "figure_07_chained_cumulative.pdf")

    contributions = tables["table_14_industry_contributions.csv"].sort_values("within_2024_twd")
    y = np.arange(len(contributions))
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.barh(y - 0.18, contributions.within_2024_twd, height=0.36, color="0.25", label="Within")
    ax.barh(y + 0.18, contributions.shift_2024_twd, height=0.36, color="0.7", label="Shift")
    ax.axvline(0, color="0.5", linewidth=0.7)
    ax.set_yticks(y, contributions.industry.str.replace("_", " "), fontsize=7)
    ax.set_xlabel("Contribution (2024 TWD per month)")
    ax.legend(frameon=False)
    save_figure(fig, "figure_08_industry_contributions.pdf")

    hours = tables["table_17_hours_mechanism.csv"]
    industry_hours = hours.loc[hours.row_type.eq("annual_industry_path")]
    aggregate_hours = hours.loc[hours.row_type.eq("annual_aggregate_path")]
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.8), sharex=True)
    for industry in industry_hours.industry.dropna().unique():
        data = industry_hours.loc[industry_hours.industry.eq(industry)]
        axes[0].plot(data.year, data.normal_hours, color="0.78", linewidth=0.65)
        axes[1].plot(data.year, data.overtime_hours, color="0.78", linewidth=0.65)
    axes[0].plot(aggregate_hours.year, aggregate_hours.normal_hours, color="0.1", linewidth=1.8, label="Employment-weighted common sample")
    axes[1].plot(aggregate_hours.year, aggregate_hours.overtime_hours, color="0.1", linewidth=1.8, label="Employment-weighted common sample")
    for ax, ylabel in zip(axes, ("Normal hours per month", "Overtime hours per month")):
        ax.axvline(2017, color="0.35", linestyle="--", linewidth=0.8)
        ax.axvline(2018, color="0.55", linestyle=":", linewidth=0.8)
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False, fontsize=7)
    axes[1].set_xlabel("Year")
    save_figure(fig, "figure_09_hours_by_industry.pdf")

    official = tables["table_18_official_comparison.csv"]
    official = official.loc[official.row_type.eq("annual_sequence")]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.7), sharey=False)
    for ax, composition, title in zip(axes, ("regular", "total"), ("Regular earnings", "Total earnings")):
        data = official.loc[official.composition.eq(composition)].sort_values("year")
        common_index = data.common_sample_real_2024_twd / data.common_sample_real_2024_twd.iloc[0] * 100.0
        published_index = data.official_published_real_rebased_2024_twd / data.official_published_real_rebased_2024_twd.iloc[0] * 100.0
        ax.plot(data.year, common_index, color="0.15", linewidth=1.7, label="Common-industry sample")
        ax.plot(data.year, published_index, color="0.55", linestyle="--", linewidth=1.7, label="Official published real series")
        ax.axhline(100, color="0.75", linewidth=0.7)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Year")
        ax.set_ylabel("Index (2000=100)")
        ax.legend(frameon=False, fontsize=7)
    save_figure(fig, "figure_10_official_comparison.pdf")

    migrant = tables["table_19_migrant_trends_validation.csv"]
    migrant = migrant.loc[migrant.row_type.eq("industry_path")]
    migrant_labels = {
        "manufacturing": "Manufacturing",
        "construction": "Construction",
        "water_waste": "Water/waste",
        "matched_productive_industries": "Matched three-industry total",
    }
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.8), sharex=True)
    for industry, shade in zip(migrant_labels, ("0.1", "0.38", "0.62", "0.78")):
        data = migrant.loc[migrant.industry.eq(industry)].sort_values("year")
        axes[0].plot(
            data.year,
            data.migrant_share * 100.0,
            color=shade,
            linewidth=1.6,
            label=migrant_labels[industry],
        )
        axes[1].plot(
            data.year,
            data.real_regular_wage_index_2000_100,
            color=shade,
            linewidth=1.6,
            label=migrant_labels[industry],
        )
    axes[0].set_ylabel("Migrant / employees (%)")
    axes[0].legend(frameon=False, fontsize=7, ncol=2)
    axes[1].set(ylabel="Real regular wage index (2000=100)", xlabel="Year")
    axes[1].legend(frameon=False, fontsize=7, ncol=2)
    save_figure(fig, "figure_11_migrant_shares_wages.pdf")

    migrant_bounds = tables["table_20_migrant_hypothetical_bounds.csv"]
    migrant_bounds = migrant_bounds.loc[
        migrant_bounds.row_type.eq("annual_hypothetical_bounds")
    ].sort_values("year")
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.fill_between(
        migrant_bounds.year.to_numpy(float),
        migrant_bounds.native_real_regular_monthly_lower_2024_twd.to_numpy(float),
        migrant_bounds.native_real_regular_monthly_upper_2024_twd.to_numpy(float),
        color="0.82",
        linewidth=0,
        label="Hypothetical native-wage interval",
    )
    ax.plot(
        migrant_bounds.year,
        migrant_bounds.observed_real_regular_monthly_2024_twd,
        color="0.12",
        linewidth=1.8,
        label="Observed manufacturing average",
    )
    ax.set(xlabel="Year", ylabel="Monthly regular wage (2024 TWD)")
    ax.legend(frameon=False, fontsize=8)
    save_figure(fig, "figure_12_migrant_hypothetical_bounds.pdf")

    productivity = tables["table_21_productivity_decomposition.csv"]
    annual_productivity = productivity.loc[productivity.row_type.eq("annual_path")]
    endpoint_productivity = productivity.loc[
        productivity.row_type.eq("endpoint_2000_2024")
    ]
    productivity_validation = tables["table_22_productivity_validation.csv"]
    productivity_path = productivity_validation.loc[
        productivity_validation.row_type.eq("manufacturing_comparison_path")
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.2))
    for ax, scope, title in (
        (axes[0, 0], "industry_services_common_16", "Industry/services common 16"),
        (axes[0, 1], "manufacturing", "Manufacturing"),
    ):
        data = annual_productivity.loc[annual_productivity.scope.eq(scope)].sort_values("year")
        ax.plot(data.year, data.labor_productivity_index_2000_100, color="0.1", linewidth=1.6, label="Value added/hour")
        ax.plot(data.year, data.real_consumer_compensation_index_2000_100, color="0.45", linestyle="--", linewidth=1.6, label="Real employee compensation/hour")
        ax.plot(data.year, data.survey_real_regular_hourly_index_2000_100, color="0.7", linestyle=":", linewidth=1.6, label="Survey real regular wage/hour")
        ax.set(title=title, xlabel="Year", ylabel="Index (2000=100)")
        ax.legend(frameon=False, fontsize=6.5)
    axes[1, 0].plot(
        productivity_path.year,
        productivity_path.reconstructed_value_added_per_hour_2021_100,
        color="0.1",
        linewidth=1.6,
        label="Reconstructed value added/hour",
    )
    axes[1, 0].plot(
        productivity_path.year,
        productivity_path.official_manufacturing_productivity_2021_100,
        color="0.6",
        linestyle="--",
        linewidth=1.6,
        label="Official production-volume/hour index",
    )
    axes[1, 0].set(title="Manufacturing concept comparison", xlabel="Year", ylabel="Index (2021=100)")
    axes[1, 0].legend(frameon=False, fontsize=6.5)

    labels = ["Productivity", "Labor share", "− price wedge", "Real compensation"]
    x = np.arange(2)
    width = 0.19
    scope_labels = ["Common 16", "Manufacturing"]
    endpoint_productivity = endpoint_productivity.set_index("scope").loc[
        ["industry_services_common_16", "manufacturing"]
    ]
    bar_values = [
        endpoint_productivity.labor_productivity_log_change.to_numpy(float),
        endpoint_productivity.employee_compensation_share_log_change.to_numpy(float),
        -endpoint_productivity.price_wedge_log_change.to_numpy(float),
        endpoint_productivity.real_consumer_compensation_log_change.to_numpy(float),
    ]
    for index, (label, values, shade) in enumerate(zip(labels, bar_values, ("0.15", "0.38", "0.62", "0.82"))):
        axes[1, 1].bar(x + (index - 1.5) * width, values * 100.0, width, color=shade, label=label)
    axes[1, 1].axhline(0, color="0.5", linewidth=0.7)
    axes[1, 1].set_xticks(x, scope_labels)
    axes[1, 1].set(title="2000–2024 accounting components", ylabel="Log percentage points")
    axes[1, 1].legend(frameon=False, fontsize=6.2, ncol=2)
    fig.tight_layout()
    save_figure(fig, "figure_13_productivity_wage_gap.pdf")


def main() -> int:
    for directory in (INTERIM, TABLES, FIGURES):
        directory.mkdir(parents=True, exist_ok=True)
    source_manifest = build_source_manifest(RAW, ROOT / "data" / "source_manifest.csv")
    cpi = read_annual_cpi(RAW / "cpi_basic_classification.xml")
    official_real = read_official_real_wage(RAW / "official_real_wage_series.csv")
    major_raw, major_official = build_major_panel(RAW)
    vintage_raw, vintage_official = build_vintage_major_panel(RAW)
    middle_raw = build_middle_panel(RAW)
    source_exclusions = list(middle_raw.attrs.get("source_exclusions", []))
    major = enrich_panel(major_raw, cpi)
    vintage = enrich_panel(vintage_raw, cpi)
    middle = enrich_panel(middle_raw, cpi)

    write_csv(cpi.assign(cpi_2024_100=cpi.cpi / cpi.loc[cpi.year.eq(2024), "cpi"].iloc[0] * 100.0), INTERIM / "annual_cpi.csv")
    write_csv(major, INTERIM / "major_industry_panel.csv")
    write_csv(vintage, INTERIM / "vintage_major_industry_panel.csv")
    write_csv(middle, INTERIM / "middle_industry_panel.csv")

    exclusions = [
        {
            "rule_id": "coverage_education_pre2009",
            "stage": "endpoint_common_sample",
            "reason": "Education is officially unavailable before 2009 and is not imputed",
            "rows_before": len(major),
            "rows_excluded": 0,
            "rows_after": len(major),
            "affected_period": "2000-2008",
            "notes": "Long-run endpoint decompositions use 16 industries common to both endpoints",
        }
    ]
    for index, item in enumerate(source_exclusions, 1):
        exclusions.append(
            {
                "rule_id": f"official_combined_workbook_{index:02d}",
                "stage": "middle_ingest",
                "reason": "Adjacent-table or exact duplicate rows embedded in the official workbook",
                "rows_before": "not_applicable",
                "rows_excluded": item["rows_excluded"],
                "rows_after": "not_applicable",
                "affected_period": item["year"],
                "notes": f"table_group={item['table_group']}",
            }
        )
    write_csv(pd.DataFrame(exclusions), ROOT / "data" / "exclusion_log.csv")

    source_summary = source_manifest.groupby(["notes", "classification_version"], as_index=False).agg(
        files=("file", "size"), bytes=("bytes", "sum"), first_coverage=("coverage", "first")
    )
    write_csv(source_summary, TABLES / "table_01_data_sources.csv")
    validation = external_checks(major, major_official, vintage, vintage_official, middle)
    write_csv(validation, TABLES / "table_02_external_validation.csv")

    tables = main_tables(major, vintage, middle, major_official, official_real)
    inference = block_bootstrap_decomposition(major, "real_regular_monthly", 2000, 2024)
    tables["table_09_inference.csv"] = inference
    cpi_rebased = cpi.assign(
        cpi_2024_100=cpi.cpi / cpi.loc[cpi.year.eq(2024), "cpi"].iloc[0] * 100.0
    )
    tables.update(phase3_tables(major, cpi_rebased))
    for filename, frame in tables.items():
        write_csv(frame, TABLES / filename)
    for path in FIGURES.glob("*"):
        if path.is_file():
            path.unlink()
    figures(major, vintage, middle, tables)
    write_results_manifest(ROOT / "results")
    paper_path = build_paper()
    print(
        json.dumps(
            {
                "sources": len(source_manifest),
                "major_rows": len(major),
                "middle_rows": len(middle),
                "external_gate": bool(validation.passed.all()),
                "paper": str(paper_path.relative_to(ROOT)),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
