"""Checks for time_split in slotpred/split.py.

    .venv/Scripts/python.exe -m pytest tests/test_split.py -v
"""
import pandas as pd

from slotpred.config import SIM_END, SIM_START, TEST_START, TRAIN_END, VALID_END, VALID_START
from slotpred.split import SPLITS, time_split


def _days():
    dates = pd.date_range(SIM_START, SIM_END, freq="D").strftime("%Y-%m-%d")
    return pd.DataFrame({"date": dates, "row": range(len(dates))})


def test_three_splits_cover_every_row_once():
    df = _days()
    s = time_split(df)
    assert set(s) == set(SPLITS)
    ids = pd.concat([s[k]["row"] for k in SPLITS])
    assert len(ids) == len(df) and ids.is_unique


def test_boundaries_are_inclusive_and_in_order():
    s = time_split(_days())
    assert (s["train"]["date"].min(), s["train"]["date"].max()) == (SIM_START, TRAIN_END)
    assert (s["valid"]["date"].min(), s["valid"]["date"].max()) == (VALID_START, VALID_END)
    assert (s["test"]["date"].min(), s["test"]["date"].max()) == (TEST_START, SIM_END)
    assert s["train"]["date"].max() < s["valid"]["date"].min() < s["test"]["date"].min()


def test_returns_copies():
    df = _days()
    s = time_split(df)
    s["train"]["date"] = "changed"
    assert df["date"].iloc[0] == SIM_START


def test_custom_date_column():
    df = _days().rename(columns={"date": "visit_date"})
    assert len(time_split(df, date_col="visit_date")["valid"]) == 30
