from pathlib import Path

import numpy as np
import pandas as pd

from wagedecomp_tw.productivity import (
    productivity_external_validation,
    productivity_wage_decomposition,
    read_dgbas_sdmx_json,
    read_official_manufacturing_productivity,
    read_official_productivity_workbook,
)


ROOT = Path(__file__).resolve().parents[1]


def test_sdmx_decoder_and_productivity_identity() -> None:
    raw = ROOT / "data" / "raw"
    panel = pd.read_csv(ROOT / "data" / "interim" / "major_industry_panel.csv")
    cpi = pd.read_csv(ROOT / "data" / "interim" / "annual_cpi.csv")
    output = read_dgbas_sdmx_json(raw / "dgbas_national_accounts_output.json")
    compensation = read_dgbas_sdmx_json(
        raw / "dgbas_national_accounts_compensation.json"
    )
    result = productivity_wage_decomposition(output, compensation, panel, cpi)

    assert len(output) == 3 * 23 * 25
    assert len(compensation) == 2 * 21 * 25
    annual = result.loc[result.row_type.eq("annual_path")]
    endpoint = result.loc[result.row_type.eq("endpoint_2000_2024")]
    assert annual.identity_residual.dropna().abs().max() < 1e-10
    lhs = endpoint.real_consumer_compensation_log_change
    rhs = (
        endpoint.labor_productivity_log_change
        + endpoint.employee_compensation_share_log_change
        - endpoint.price_wedge_log_change
    )
    assert np.allclose(lhs, rhs, rtol=0, atol=1e-10)
    assert set(endpoint.scope) == {"industry_services_common_16", "manufacturing"}


def test_productivity_external_validation_contract() -> None:
    raw = ROOT / "data" / "raw"
    panel = pd.read_csv(ROOT / "data" / "interim" / "major_industry_panel.csv")
    cpi = pd.read_csv(ROOT / "data" / "interim" / "annual_cpi.csv")
    output = read_dgbas_sdmx_json(raw / "dgbas_national_accounts_output.json")
    compensation = read_dgbas_sdmx_json(
        raw / "dgbas_national_accounts_compensation.json"
    )
    decomposition = productivity_wage_decomposition(output, compensation, panel, cpi)
    official = read_official_manufacturing_productivity(
        raw / "mol_major_economic_indicators.csv"
    )
    workbook = read_official_productivity_workbook(
        raw / "dgbas_manufacturing_labor_productivity.xlsx"
    )
    checks = productivity_external_validation(decomposition, official, workbook, output)
    assert checks.loc[checks.threshold.notna(), "passed"].all()
    concept_check = checks.loc[
        checks.check.eq("reconstructed_value_added_per_hour_vs_official_volume_productivity")
    ].iloc[0]
    assert np.isnan(concept_check.threshold)
    assert concept_check.maximum_relative_error > 0
