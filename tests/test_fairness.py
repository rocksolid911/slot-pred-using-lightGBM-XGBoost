"""Checks for your two TODO(you) functions in slotpred/fairness.py.

    .venv/Scripts/python.exe -m pytest tests/test_fairness.py -v
"""
import numpy as np
import pandas as pd
import pytest

from slotpred.fairness import group_table, luck_band


# ---------------------------------------------------------------- luck_band

def test_luck_band_example():
    assert luck_band(0.86, 461) == pytest.approx(0.0323, abs=0.0005)


def test_luck_band_shrinks_like_one_over_root_n():
    assert luck_band(0.8, 400) == pytest.approx(2 * luck_band(0.8, 1600))


def test_luck_band_takes_many_n():
    out = luck_band(0.8, pd.Series([100, 400]))
    assert out.iloc[0] == pytest.approx(0.08) and out.iloc[1] == pytest.approx(0.04)


# ---------------------------------------------------------------- group_table

@pytest.fixture
def visits():
    # 1000 adults: 90% on time; 100 elderly: 60% on time. Overall 960 / 1100.
    adult = pd.DataFrame({"age_band": "adult", "on_time": [True] * 900 + [False] * 100,
                          "late_min": [0.0] * 900 + [2.0] * 100, "today_on_time": [True] * 700 + [False] * 300})
    old = pd.DataFrame({"age_band": "65+", "on_time": [True] * 60 + [False] * 40,
                        "late_min": [0.0] * 60 + [3.0] * 40, "today_on_time": [True] * 50 + [False] * 50})
    return pd.concat([adult, old], ignore_index=True)


def test_group_table_columns_and_order(visits):
    t = group_table(visits, "age_band")
    assert list(t.columns) == ["n", "on_time", "today", "late_min", "gap", "band", "flag"]
    assert list(t.index) == ["65+", "adult"]                      # lowest on-time first


def test_group_table_numbers(visits):
    t = group_table(visits, "age_band")
    overall = 960 / 1100
    assert t.loc["65+", "n"] == 100
    assert t.loc["65+", "on_time"] == pytest.approx(0.60)
    assert t.loc["65+", "today"] == pytest.approx(0.50)
    assert t.loc["65+", "late_min"] == pytest.approx(1.2)
    assert t.loc["65+", "gap"] == pytest.approx(0.60 - overall)
    assert t.loc["65+", "band"] == pytest.approx(2 * np.sqrt(overall * (1 - overall) / 100))


def test_group_table_flags_only_real_gaps(visits):
    t = group_table(visits, "age_band")
    assert bool(t.loc["65+", "flag"]) is True        # 27 points below, band about 7
    assert bool(t.loc["adult", "flag"]) is False     # above average
