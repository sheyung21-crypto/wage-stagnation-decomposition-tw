import pandas as pd

from wagedecomp_tw.crosswalk import aggregate_with_crosswalk, assert_conservation


def test_crosswalk_conserves_employment_and_payroll() -> None:
    frame = pd.DataFrame(
        {"year": [2020, 2020, 2020], "industry": ["a", "b", "c"], "employees": [10.0, 20.0, 30.0], "wage": [2.0, 3.0, 4.0]}
    )
    mapping = pd.DataFrame({"industry": ["a", "b", "c"], "target": ["x", "x", "y"]})
    result = aggregate_with_crosswalk(frame, mapping)
    assert_conservation(frame, result)
    assert result.employees.sum() == 60.0

