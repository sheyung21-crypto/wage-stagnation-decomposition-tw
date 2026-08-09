import numpy as np
import pandas as pd

from wagedecomp_tw.decomposition import decompose, laspeyres_arrays, paasche_arrays, tornqvist_arrays


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

