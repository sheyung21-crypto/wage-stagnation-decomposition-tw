import numpy as np

from wagedecomp_tw.inference import circular_block_indices


def test_bootstrap_indices_are_reproducible_and_circular() -> None:
    first = circular_block_indices(8, replications=25, block_length=3, seed=20260809)
    second = circular_block_indices(8, replications=25, block_length=3, seed=20260809)
    assert np.array_equal(first, second)
    assert first.shape == (25, 8)
    assert first.min() >= 0 and first.max() < 8

