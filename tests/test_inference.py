import numpy as np
import pandas as pd

from wagedecomp_tw.inference import block_bootstrap_chained_paths, circular_block_indices


def test_bootstrap_indices_are_reproducible_and_circular() -> None:
    first = circular_block_indices(8, replications=25, block_length=3, seed=20260809)
    second = circular_block_indices(8, replications=25, block_length=3, seed=20260809)
    assert np.array_equal(first, second)
    assert first.shape == (25, 8)
    assert first.min() >= 0 and first.max() < 8


def test_chained_bootstrap_is_reproducible_and_complete() -> None:
    frame = pd.DataFrame(
        {
            "year": np.repeat(np.arange(2000, 2005), 2),
            "industry": ["a", "b"] * 5,
            "employees": [40, 60, 42, 58, 44, 56, 46, 54, 48, 52],
            "wage": [100, 200, 103, 202, 106, 204, 109, 206, 112, 208],
        }
    )
    first = block_bootstrap_chained_paths(
        frame, "wage", 2000, 2004, replications=50, seed=7
    )
    second = block_bootstrap_chained_paths(
        frame, "wage", 2000, 2004, replications=50, seed=7
    )
    pd.testing.assert_frame_equal(first, second)
    assert set(first.component) == {"within", "shift", "interaction", "total"}
    assert set(first.year) == set(range(2000, 2005))
    assert first.valid_replications.eq(50).all()

