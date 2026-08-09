from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wagedecomp_tw.migrant import (
    manufacturing_contribution_bounds,
    manufacturing_native_wage_bounds,
    migrant_mapping_validation,
    migrant_share_paths,
    read_migrant_counts,
    read_minimum_monthly_wage,
)


ROOT = Path(__file__).resolve().parents[1]


def test_official_migrant_paths_and_bounds() -> None:
    raw = ROOT / "data" / "raw"
    panel = pd.read_csv(ROOT / "data" / "interim" / "major_industry_panel.csv")
    migrants = read_migrant_counts(raw / "mol_foreign_workers_by_work.csv")
    minimum = read_minimum_monthly_wage(raw / "mol_major_economic_indicators.csv")
    minimum_by_year = minimum.set_index("year").minimum_monthly_wage_twd
    assert minimum_by_year.loc[2000] == 15_840
    assert minimum_by_year.loc[2024] == 27_470
    assert minimum.minimum_monthly_wage_twd.between(10_000, 100_000).all()
    shares = migrant_share_paths(migrants, panel)
    validation = migrant_mapping_validation(migrants, shares)
    bounds = manufacturing_native_wage_bounds(shares, panel, minimum)
    contribution = manufacturing_contribution_bounds(bounds, panel)

    assert shares.migrant_share.between(0, 0.5, inclusive="left").all()
    assert validation.matched_fraction_of_productive_migrants.between(0, 1).all()
    assert np.all(bounds.native_regular_monthly_lower_twd <= bounds.native_regular_monthly_upper_twd)
    assert "midpoint" not in " ".join(bounds.columns).lower()
    assert contribution.identified_status.str.contains("no point estimate").all()


def test_migrant_share_gate_stops_above_fifty_percent() -> None:
    migrants = pd.DataFrame(
        {
            "year": [2024],
            "manufacturing": [60.0],
            "construction": [0.0],
            "waste": [0.0],
        }
    )
    panel = pd.DataFrame(
        {
            "year": [2024, 2024, 2024],
            "industry": ["manufacturing", "construction", "water_waste"],
            "employees": [100.0, 100.0, 100.0],
        }
    )
    with pytest.raises(ValueError, match="50%"):
        migrant_share_paths(migrants, panel)
