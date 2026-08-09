from __future__ import annotations

import unicodedata

import numpy as np
import pandas as pd


def normalize_label(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = text.split("\n", 1)[0]
    for mark in (" ", "　", "、", "，", ",", "﹑"):
        text = text.replace(mark, "")
    return text.strip()


MAJOR_LABEL_ALIASES = {
    normalize_label(value)
    for value in (
        "工業及服務業",
        "工業",
        "服務業",
        "總計",
        "礦業及土石採取業",
        "製造業",
        "電力及燃氣供應業",
        "用水供應及污染整治業",
        "營建工程業",
        "營造業",
        "批發及零售業",
        "運輸及倉儲業",
        "住宿及餐飲業",
        "資訊及通訊傳播業",
        "出版、影音製作、傳播及資通訊服務業",
        "出版影音及資通訊業",
        "金融及保險業",
        "不動產業",
        "專業、科學及技術服務業",
        "支援服務業",
        "教育業",
        "教育服務業",
        "醫療保健服務業",
        "醫療保健及社會工作服務業",
        "藝術、娛樂及休閒服務業",
        "其他服務業",
    )
}


def is_major_label(value: object) -> bool:
    normalized = normalize_label(value)
    return normalized in MAJOR_LABEL_ALIASES or normalized.startswith("教育業(")


def aggregate_with_crosswalk(
    frame: pd.DataFrame,
    mapping: pd.DataFrame,
    source_col: str = "industry",
    target_col: str = "target",
) -> pd.DataFrame:
    merged = frame.merge(
        mapping[[source_col, target_col]], on=source_col, how="left", validate="many_to_one"
    )
    if merged[target_col].isna().any():
        missing = sorted(merged.loc[merged[target_col].isna(), source_col].unique())
        raise ValueError(f"Unmapped industries: {missing}")
    merged["payroll"] = merged["employees"] * merged["wage"]
    grouped = merged.groupby(["year", target_col], as_index=False).agg(
        employees=("employees", "sum"), payroll=("payroll", "sum")
    )
    grouped["wage"] = grouped["payroll"] / grouped["employees"]
    return grouped


def assert_conservation(before: pd.DataFrame, after: pd.DataFrame, atol: float = 1e-8) -> None:
    lhs = before.assign(payroll=before.employees * before.wage).groupby("year")[["employees", "payroll"]].sum()
    rhs = after.groupby("year")[["employees", "payroll"]].sum()
    if not np.allclose(lhs.values, rhs.values, rtol=0.0, atol=atol):
        raise AssertionError("Crosswalk aggregation does not conserve employment and payroll")
