from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    path = ROOT / "results" / "tables" / "table_02_external_validation.csv"
    frame = pd.read_csv(path)
    required = {
        "latest_major_vs_official",
        "same_vintage_major_vs_official",
        "middle_vs_same_vintage_major",
    }
    if set(frame.check) != required:
        raise AssertionError("External validation does not contain all required checks")
    if not frame.passed.astype(bool).all():
        raise AssertionError("At least one external check is marked failed")
    if not (frame.maximum_relative_error < frame.threshold).all():
        raise AssertionError("External consistency threshold was exceeded")
    print(f"external verification passed: {len(frame)} checks; max error={frame.maximum_relative_error.max():.6%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

