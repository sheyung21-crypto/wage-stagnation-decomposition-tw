import numpy as np
import pandas as pd

from wagedecomp_tw.decomposition import (
    chained_decomposition,
    decompose,
    industry_contributions,
    laspeyres_arrays,
    paasche_arrays,
    tornqvist_arrays,
)


def test_array_identities() -> None:
    s0 = np.array([0.4, 0.6])
    s1 = np.array([0.5, 0.5])
    w0 = np.array([100.0, 200.0])
    w1 = np.array([120.0, 210.0])
    for function in (laspeyres_arrays, paasche_arrays, tornqvist_arrays):
        result = function(s0, s1, w0, w1)
        assert np.isclose(
            result["within"] + result["shift"] + result["interaction"] + result["residual"],
            result["total"],
            atol=1e-12,
        )


def test_dataframe_decomposition() -> None:
    frame = pd.DataFrame(
        {
            "year": [2000, 2000, 2024, 2024],
            "industry": ["a", "b", "a", "b"],
            "employees": [40, 60, 50, 50],
            "wage": [100.0, 200.0, 120.0, 210.0],
        }
    )
    result = decompose(frame, "wage", 2000, 2024, "laspeyres")
    assert np.isclose(result["total"], 5.0)
    assert result["n_industries"] == 2


def test_chained_decomposition_annual_and_cumulative_identities() -> None:
    frame = pd.DataFrame(
        {
            "year": [2000, 2000, 2001, 2001, 2002, 2002],
            "industry": ["a", "b"] * 3,
            "employees": [40, 60, 45, 55, 50, 50],
            "wage": [100.0, 200.0, 110.0, 205.0, 120.0, 210.0],
        }
    )
    for method in ("laspeyres", "paasche"):
        result = chained_decomposition(frame, "wage", 2000, 2002, method)
        annual_sum = result.annual_within + result.annual_shift + result.annual_interaction
        cumulative_sum = (
            result.cumulative_within
            + result.cumulative_shift
            + result.cumulative_interaction
        )
        assert np.allclose(annual_sum, result.annual_total, rtol=1e-10, atol=1e-10)
        assert np.allclose(cumulative_sum, result.cumulative_total, rtol=1e-10, atol=1e-10)
        assert np.isclose(result.cumulative_total.iloc[-1], 5.0)


def test_industry_contributions_add_to_endpoint_components() -> None:
    frame = pd.DataFrame(
        {
            "year": [2000, 2000, 2024, 2024],
            "industry_code": ["A", "B", "A", "B"],
            "industry": ["a", "b", "a", "b"],
            "employees": [40, 60, 50, 50],
            "wage": [100.0, 200.0, 120.0, 210.0],
        }
    )
    result = industry_contributions(frame, "wage", 2000, 2024)
    endpoint = decompose(frame, "wage", 2000, 2024, "laspeyres")
    assert np.isclose(result.within_2024_twd.sum(), endpoint["within"])
    assert np.isclose(result.shift_2024_twd.sum(), endpoint["shift"])
    assert np.isclose(result.interaction_2024_twd.sum(), endpoint["interaction"])
    assert np.isclose(result.industry_total_change_2024_twd.sum(), endpoint["total"])

