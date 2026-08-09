import numpy as np
import pandas as pd

from wagedecomp_tw.deflate import rebase_cpi, to_reference_prices


def test_cpi_rebasing_and_deflation() -> None:
    cpi = pd.Series({2000: 80.0, 2024: 120.0})
    rebased = rebase_cpi(cpi, 2024)
    assert rebased.loc[2024] == 100.0
    real = to_reference_prices(pd.Series([80.0, 120.0]), pd.Series([2000, 2024]), rebased)
    assert np.allclose(real, [120.0, 120.0])

