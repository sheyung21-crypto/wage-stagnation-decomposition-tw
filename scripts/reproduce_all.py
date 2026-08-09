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
from wagedecomp_tw.decomposition import decompose, nominal_then_deflate_bridge
from wagedecomp_tw.inference import block_bootstrap_decomposition
from wagedecomp_tw.ingest import (
    build_major_panel,
    build_middle_panel,
    build_source_manifest,
    build_vintage_major_panel,
    read_annual_cpi,
)
from wagedecomp_tw.panel import aggregate_wage, common_industry_path, enrich_panel
from wagedecomp_tw.provenance import write_results_manifest


RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
PAPER_FIGURES = ROOT / "paper" / "figures"
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


def main_tables(major: pd.DataFrame, vintage: pd.DataFrame, middle: pd.DataFrame) -> dict[str, pd.DataFrame]:
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
    return tables


def save_figure(fig: plt.Figure, filename: str) -> None:
    path = FIGURES / filename
    fig.savefig(path, bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None, "Creator": "wagedecomp_tw"})
    plt.close(fig)
    shutil.copyfile(path, PAPER_FIGURES / filename)


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


def main() -> int:
    for directory in (INTERIM, TABLES, FIGURES, PAPER_FIGURES):
        directory.mkdir(parents=True, exist_ok=True)
    source_manifest = build_source_manifest(RAW, ROOT / "data" / "source_manifest.csv")
    cpi = read_annual_cpi(RAW / "cpi_basic_classification.xml")
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

    tables = main_tables(major, vintage, middle)
    inference = block_bootstrap_decomposition(major, "real_regular_monthly", 2000, 2024)
    tables["table_09_inference.csv"] = inference
    for filename, frame in tables.items():
        write_csv(frame, TABLES / filename)
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
