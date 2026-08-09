from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def macros() -> dict[str, float]:
    text = (ROOT / "paper" / "main.tex").read_text(encoding="utf-8")
    return {name: float(value) for name, value in re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{(-?[0-9.]+)\}", text)}


def verify_table_mapping() -> int:
    text = (ROOT / "paper" / "main.tex").read_text(encoding="utf-8")
    pairs = [
        (int(number), filename.replace("\\_", "_"))
        for number, filename in re.findall(
            r"\\resultmap\{(\d{2})\}\{([^}]+\.csv)\}", text
        )
    ]
    if len(pairs) != len(set(pairs)):
        raise AssertionError("Duplicate paper table mappings")
    numbers = [number for number, _ in pairs]
    filenames = [filename for _, filename in pairs]
    if len(numbers) != len(set(numbers)) or len(filenames) != len(set(filenames)):
        raise AssertionError("Paper table numbers and CSV filenames must both be one-to-one")
    expected_files = sorted(path.name for path in (ROOT / "results" / "tables").glob("table_*.csv"))
    if sorted(filenames) != expected_files:
        missing = sorted(set(expected_files) - set(filenames))
        extra = sorted(set(filenames) - set(expected_files))
        raise AssertionError(f"Paper/CSV mapping mismatch; missing={missing}, extra={extra}")
    if sorted(numbers) != list(range(1, len(expected_files) + 1)):
        raise AssertionError("Paper table numbering is not contiguous and complete")
    for number, filename in pairs:
        expected_prefix = f"table_{number:02d}_"
        if not filename.startswith(expected_prefix):
            raise AssertionError(f"Paper table {number} points to {filename}")

    manifest = json.loads((ROOT / "results" / "results_manifest.json").read_text(encoding="utf-8"))
    manifest_mapping = manifest["paper_number_mapping"]["tables"]
    expected_manifest = {
        str(number): f"results/tables/{filename}" for number, filename in pairs
    }
    if manifest_mapping != expected_manifest:
        raise AssertionError("Results manifest table mapping disagrees with the paper")
    return len(pairs)


def main() -> int:
    values = macros()
    main_table = pd.read_csv(ROOT / "results" / "tables" / "table_03_main_decomposition.csv")
    main_row = main_table.loc[
        main_table.metric.eq("real_regular_monthly")
        & main_table.method.eq("laspeyres")
        & main_table.deflation_order.eq("deflate_then_decompose")
    ].iloc[0]
    monthly_hourly = pd.read_csv(ROOT / "results" / "tables" / "table_05_monthly_hourly.csv").set_index("composition")
    counterfactual = pd.read_csv(ROOT / "results" / "tables" / "table_08_counterfactual.csv")
    cf2000 = counterfactual.loc[counterfactual.base_year.eq(2000) & counterfactual.year.eq(2024)].iloc[0]
    inference = pd.read_csv(ROOT / "results" / "tables" / "table_09_inference.csv").set_index("component")
    chained = pd.read_csv(ROOT / "results" / "tables" / "table_10_chained_decomposition.csv")
    chained = chained.loc[chained.method.eq("laspeyres") & chained.year.eq(2024)].iloc[0]
    chained_comparison = pd.read_csv(ROOT / "results" / "tables" / "table_11_chained_vs_endpoint.csv")
    chained_comparison = chained_comparison.loc[chained_comparison.method.eq("laspeyres")].iloc[0]
    industry = pd.read_csv(ROOT / "results" / "tables" / "table_14_industry_contributions.csv")
    official = pd.read_csv(ROOT / "results" / "tables" / "table_18_official_comparison.csv")
    official = official.loc[official.row_type.eq("growth_gap_decomposition")]
    official_summary = official.drop_duplicates("composition").set_index("composition")
    expected = {
        "MainGrowth": round(float(monthly_hourly.loc["regular", "monthly_log_growth_percent"]), 2),
        "HourlyGrowth": round(float(monthly_hourly.loc["regular", "hourly_log_growth_percent"]), 2),
        "HoursGap": round(float(monthly_hourly.loc["regular", "hours_accounting_gap_percentage_points"]), 2),
        "MainWithin": round(float(main_row.within)),
        "MainShift": round(float(main_row["shift"])),
        "MainInteraction": round(float(main_row.interaction)),
        "MainTotal": round(float(main_row.total)),
        "CounterfactualGap": round(float(cf2000.difference)),
        "ShiftPValue": round(float(inference.loc["shift", "p_value_two_sided"]), 3),
        "ChainWithin": round(float(chained.cumulative_within)),
        "ChainShift": round(float(chained.cumulative_shift)),
        "ChainInteraction": round(float(chained.cumulative_interaction)),
        "InteractionShrink": round(float(chained_comparison.interaction_absolute_shrinkage_percent), 2),
        "ChainMethodGap": round(float(chained_comparison.chained_laspeyres_paasche_within_gap_2024_twd)),
        "IndustryTopThree": round(float(industry.top3_within_concentration_percent.iloc[0]), 2),
        "TotalGrowth": round(float(monthly_hourly.loc["total", "monthly_log_growth_percent"]), 2),
        "OfficialRegularGrowth": round(float(official_summary.loc["regular", "official_published_log_growth_percent"]), 2),
        "OfficialTotalGrowth": round(float(official_summary.loc["total", "official_published_log_growth_percent"]), 2),
        "RegularOfficialGap": round(float(official_summary.loc["regular", "observed_official_minus_common_growth_gap_percentage_points"]), 2),
        "TotalOfficialGap": round(float(official_summary.loc["total", "observed_official_minus_common_growth_gap_percentage_points"]), 2),
    }
    missing = sorted(set(expected) - set(values))
    if missing:
        raise AssertionError(f"Missing paper number macros: {missing}")
    mismatches = {key: (values[key], expected_value) for key, expected_value in expected.items() if values[key] != expected_value}
    if mismatches:
        raise AssertionError(f"Paper numbers disagree with results: {mismatches}")
    table_count = verify_table_mapping()
    print(f"paper-number verification passed: {len(expected)} claims; {table_count} bidirectional table mappings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

