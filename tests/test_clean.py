"""Checks for the TODO(you) rules in slotpred/clean.py.

Small hand-made cases test the exact behaviour (edges at 1 and 120 minutes, the Whitefield dates);
the last tests run your rules over the real export.

    .venv/Scripts/python.exe -m pytest tests/test_clean.py -v
"""
import numpy as np
import pandas as pd
import pytest

from slotpred import clean as c
from slotpred.config import DATA


# ---------------------------------------------------------------- normalise_doctor

@pytest.mark.parametrize("spelling", ["Sanjay Patil", "Dr. Sanjay Patil", "DR SANJAY PATIL",
                                      "Sanjay P.", " dr sanjay patil "])
def test_doctor_spellings_map_to_roster(spelling):
    assert c.normalise_doctor(spelling) == "Sanjay Patil"


def test_every_roster_doctor_in_every_style():
    for name in c.ROSTER:
        first, last = name.split()
        for s in (name, "Dr. " + name, "DR " + name.upper(), f"{first} {last[0]}.", " dr " + name.lower() + " "):
            assert c.normalise_doctor(s) == name, s


def test_unknown_doctor_gives_none_not_an_error():
    assert c.normalise_doctor("Dr. Gregory House") is None


# ---------------------------------------------------------------- normalise_visit_type

@pytest.mark.parametrize("spelling, expected", [
    ("new", "new"), ("New", "new"), ("NEW", "new"), ("new visit", "new"),
    ("follow-up", "follow-up"), ("Follow-up", "follow-up"), ("follow up", "follow-up"),
    ("FOLLOW-UP", "follow-up"), ("followup", "follow-up"),
    ("review", "review"), ("Review", "review"), ("report review", "review"),
])
def test_visit_type_spellings(spelling, expected):
    assert c.normalise_visit_type(spelling) == expected


def test_unknown_visit_type_gives_none():
    assert c.normalise_visit_type("emergency") is None


# ---------------------------------------------------------------- fix_age

def test_fix_age_removes_placeholder_keeps_real_values():
    age = pd.Series([0.0, 45.0, 999.0, np.nan, 92.0, -3.0, 150.0])
    fixed = c.fix_age(age)
    assert fixed.tolist()[:2] == [0.0, 45.0]           # babies are real
    assert fixed.iloc[4] == 92.0
    assert fixed.iloc[[2, 3, 5, 6]].isna().all()       # placeholder, missing, impossible
    assert age.iloc[2] == 999.0                        # a copy: the input is unchanged


# ---------------------------------------------------------------- add_minutes

def test_add_minutes():
    df = pd.DataFrame({"consult_start": ["2026-03-02 10:00:00", "2026-03-02 11:00:00"],
                       "consult_end": ["2026-03-02 10:12:30", None]}, index=[7, 9])
    m = c.add_minutes(df)
    assert m.loc[7] == pytest.approx(12.5)
    assert np.isnan(m.loc[9])
    assert list(m.index) == [7, 9]


# ---------------------------------------------------------------- target_ok

def _rows(**cols):
    n = max(len(v) for v in cols.values())
    base = dict(clinic=["HSR"] * n, date=["2026-02-10"] * n, minutes=[12.0] * n)
    base.update(cols)
    return pd.DataFrame(base)


def test_target_ok_duration_edges():
    df = _rows(minutes=[0.99, 1.0, 12.0, 120.0, 120.01, np.nan])
    assert c.target_ok(df).tolist() == [False, True, True, True, False, False]


def test_target_ok_whitefield_window():
    dates = ["2026-02-28", "2026-03-01", "2026-04-15", "2026-04-30", "2026-05-01"]
    whitefield = c.target_ok(_rows(clinic=["Whitefield"] * 5, date=dates))
    other = c.target_ok(_rows(clinic=["HSR"] * 5, date=dates))
    assert whitefield.tolist() == [True, False, False, False, True]
    assert other.all()                                  # only Whitefield had the process change


# ---------------------------------------------------------------- on the real export

needs_data = pytest.mark.skipif(not (DATA / "raw_emr_export.csv").exists(),
                                reason="run python -m slotpred.simulate first")


@needs_data
def test_every_export_spelling_is_recognised():
    raw = pd.read_csv(DATA / "raw_emr_export.csv")
    for v in raw["doctor"].unique():
        doc = c.normalise_doctor(v)
        assert doc in c.ROSTER, v
    for v in raw["visit_type"].unique():
        assert c.normalise_visit_type(v) in c.VISIT_TYPES, v


@needs_data
def test_mapped_doctor_matches_the_row_department():
    """A self-check that needs no hidden truth: a mapped name must belong to the row's department."""
    raw = pd.read_csv(DATA / "raw_emr_export.csv")
    pairs = raw[["doctor", "department"]].drop_duplicates()
    for spelling, dept in pairs.itertuples(index=False):
        assert c.ROSTER[c.normalise_doctor(spelling)] == dept, (spelling, dept)


@needs_data
def test_clean_pipeline():
    df, funnel = c.clean(pd.read_csv(DATA / "raw_emr_export.csv"))
    assert not df.duplicated().any()
    assert df["minutes"].between(c.MIN_MINUTES, c.MAX_MINUTES).all()
    assert set(df["doctor"]) <= set(c.ROSTER) and set(df["visit_type"]) <= set(c.VISIT_TYPES)
    assert (df["age"].dropna() <= 120).all()
    glitch = (df["clinic"] == c.GLITCH_CLINIC) & df["date"].between(c.GLITCH_FROM, c.GLITCH_TO)
    assert not glitch.any()
    assert 40_000 < len(df) < 42_500                   # ~2,500 of ~43,750 visits dropped
