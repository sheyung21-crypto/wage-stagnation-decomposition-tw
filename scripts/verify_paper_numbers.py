from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def macros() -> dict[str, float]:
    text = (ROOT / "paper" / "main.tex").read_text(encoding="utf-8")
    return {name: float(value) for name, value in re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{(-?[0-9.]+)\}", text)}


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
    }
    missing = sorted(set(expected) - set(values))
    if missing:
        raise AssertionError(f"Missing paper number macros: {missing}")
    mismatches = {key: (values[key], expected_value) for key, expected_value in expected.items() if values[key] != expected_value}
    if mismatches:
        raise AssertionError(f"Paper numbers disagree with results: {mismatches}")
    print(f"paper-number verification passed: {len(expected)} claims")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

