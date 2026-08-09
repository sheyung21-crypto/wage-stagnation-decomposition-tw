from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"

EXPECTED_TABLES = {
    "table_01_data_sources.csv",
    "table_02_external_validation.csv",
    "table_03_main_decomposition.csv",
    "table_04_granularity_comparison.csv",
    "table_05_monthly_hourly.csv",
    "table_06_regular_total.csv",
    "table_07_period_sensitivity.csv",
    "table_08_counterfactual.csv",
    "table_09_inference.csv",
    "table_10_chained_decomposition.csv",
    "table_11_chained_vs_endpoint.csv",
    "table_12_institutional_periods.csv",
    "table_13_covid_predictions.csv",
    "table_14_industry_contributions.csv",
    "table_15_total_wage_decomposition.csv",
    "table_16_total_wage_endpoint_sensitivity.csv",
    "table_17_hours_mechanism.csv",
    "table_18_official_comparison.csv",
    "table_19_migrant_trends_validation.csv",
    "table_20_migrant_hypothetical_bounds.csv",
    "table_21_productivity_decomposition.csv",
    "table_22_productivity_validation.csv",
}
EXPECTED_FIGURES = {
    "figure_01_real_monthly_hourly.pdf",
    "figure_02_regular_total.pdf",
    "figure_03_employment_shares.pdf",
    "figure_04_decomposition_methods.pdf",
    "figure_05_granularity.pdf",
    "figure_06_counterfactual.pdf",
    "figure_07_chained_cumulative.pdf",
    "figure_08_industry_contributions.pdf",
    "figure_09_hours_by_industry.pdf",
    "figure_10_official_comparison.pdf",
    "figure_11_migrant_shares_wages.pdf",
    "figure_12_migrant_hypothetical_bounds.pdf",
    "figure_13_productivity_wage_gap.pdf",
}


def test_phase3_output_inventory_and_manifest_hashes() -> None:
    assert {path.name for path in TABLES.glob("*.csv")} == EXPECTED_TABLES
    assert {path.name for path in FIGURES.glob("*.pdf")} == EXPECTED_FIGURES

    manifest = json.loads((ROOT / "results" / "results_manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["paper_number_mapping"]["tables"]) == {str(i) for i in range(1, 23)}
    assert set(manifest["paper_number_mapping"]["figures"]) == {str(i) for i in range(1, 14)}
    for item in manifest["files"]:
        path = ROOT / item["path"]
        assert path.stat().st_size == item["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]


def test_chained_csv_identities_and_bootstrap_contract() -> None:
    frame = pd.read_csv(TABLES / "table_10_chained_decomposition.csv")
    annual = frame[["annual_within", "annual_shift", "annual_interaction"]].sum(axis=1)
    cumulative = frame[["cumulative_within", "cumulative_shift", "cumulative_interaction"]].sum(axis=1)
    assert np.allclose(annual, frame["annual_total"], rtol=1e-10, atol=1e-8)
    assert np.allclose(cumulative, frame["cumulative_total"], rtol=1e-10, atol=1e-8)
    assert set(frame["valid_replications"]) == {10000}
    assert set(frame["seed"]) == {20260809}
    assert set(frame["block_length"]) == {3}


def test_industry_contributions_and_prediction_statuses() -> None:
    contributions = pd.read_csv(TABLES / "table_14_industry_contributions.csv")
    main = pd.read_csv(TABLES / "table_03_main_decomposition.csv")
    endpoint = main.loc[
        main["metric"].eq("real_regular_monthly") & main["method"].eq("laspeyres")
    ].iloc[0]
    for column, component in (
        ("within_2024_twd", "within"),
        ("shift_2024_twd", "shift"),
        ("interaction_2024_twd", "interaction"),
        ("industry_total_change_2024_twd", "total"),
    ):
        assert np.isclose(contributions[column].sum(), endpoint[component], atol=1e-8)
    assert contributions["top3_concentration_exceeds_50_percent"].all()

    predictions = pd.read_csv(TABLES / "table_13_covid_predictions.csv")
    assert set(predictions["prediction_id"]) == {"P1", "P2", "P3"}
    assert predictions["supported"].all()


def test_total_wage_has_nine_endpoint_pairs_and_three_methods() -> None:
    frame = pd.read_csv(TABLES / "table_16_total_wage_endpoint_sensitivity.csv")
    assert frame[["start_year", "end_year"]].drop_duplicates().shape[0] == 9
    assert set(frame["method"]) == {"laspeyres", "paasche", "tornqvist"}
    assert len(frame) == 27


def test_phase3_bounds_productivity_and_single_figure_location() -> None:
    bounds = pd.read_csv(TABLES / "table_20_migrant_hypothetical_bounds.csv")
    endpoint = bounds.loc[bounds.row_type.eq("endpoint_contribution_bounds")].iloc[0]
    assert endpoint.identified_status == "hypothetical bounds; no point estimate"
    assert endpoint.composition_removed_within_contribution_lower_2024_twd <= endpoint.composition_removed_within_contribution_upper_2024_twd
    assert endpoint.composition_adjustment_lower_2024_twd > 0
    assert "midpoint" not in " ".join(bounds.columns).lower()

    productivity = pd.read_csv(TABLES / "table_21_productivity_decomposition.csv")
    endpoint_productivity = productivity.loc[productivity.row_type.eq("endpoint_2000_2024")]
    assert endpoint_productivity.identity_residual.abs().max() < 1e-10

    validation = pd.read_csv(TABLES / "table_22_productivity_validation.csv")
    gated = validation.loc[
        validation.row_type.eq("validation_summary") & validation.threshold.notna()
    ]
    assert gated.passed.astype(str).str.lower().eq("true").all()
    assert not list((ROOT / "paper" / "figures").glob("*.pdf"))
