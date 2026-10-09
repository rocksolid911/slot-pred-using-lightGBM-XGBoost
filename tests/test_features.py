"""Checks for the TODO(you) parts of slotpred/features.py, and for leakage in the model's inputs.

    .venv/Scripts/python.exe -m pytest tests/test_features.py -v
"""
import numpy as np
import pandas as pd

from slotpred import features as f


def _visits(**cols):
    n = max(len(v) for v in cols.values())
    base = dict(date=["2026-03-02"] * n, slot_time=["10:15"] * n, department=["Cardiology"] * n,
                visit_type=["new"] * n, doctor=["Anita Sharma"] * n, clinic=["HSR"] * n,
                age=[40.0] * n, interpreter=[False] * n, teleconsult=[False] * n, prior_reports=[1] * n,
                appointment_id=[f"A{i}" for i in range(n)], patient_id=["P1"] * n, patient_name=["X"] * n,
                check_in=["2026-03-02 10:05:00"] * n, consult_start=["2026-03-02 10:20:00"] * n,
                consult_end=["2026-03-02 10:35:00"] * n, minutes=[15.0] * n)
    base.update(cols)
    return pd.DataFrame(base)


# ---------------------------------------------------------------- add_time_features

def test_weekday_and_hour():
    df = _visits(date=["2026-03-02", "2026-03-07", "2026-08-31"], slot_time=["09:00", "14:45", "17:30"])
    out = f.add_time_features(df)
    assert out["weekday"].tolist() == [0, 5, 0]        # Monday, Saturday, Monday
    assert out["hour"].tolist() == [9, 14, 17]
    assert "weekday" not in df                          # a copy


# ---------------------------------------------------------------- impute_age

def test_impute_age_fills_and_flags():
    filled, flag = f.impute_age(pd.Series([3.0, np.nan, 70.0, np.nan]), fill=42.0)
    assert filled.tolist() == [3.0, 42.0, 70.0, 42.0]
    assert list(flag) == [0, 1, 0, 1]


def test_flag_separates_real_42_from_unknown():
    filled, flag = f.impute_age(pd.Series([42.0, np.nan]), fill=42.0)
    assert filled.iloc[0] == filled.iloc[1]             # same age after filling...
    assert flag.iloc[0] != flag.iloc[1]                 # ...the flag is the only way to tell them apart


# ---------------------------------------------------------------- fit_prep: training rows only

def test_fit_prep_learns_from_train_only():
    train = _visits(age=[10.0, 20.0, 30.0, np.nan], doctor=["Anita Sharma", "Rahul Menon"] * 2)
    test = _visits(age=[90.0, 90.0, 90.0], doctor=["Nikhil Verma"] * 3)
    state = f.fit_prep(train)
    assert state.age_fill == 20.0                       # median of 10, 20, 30; NaN ignored; test never seen
    assert state.categories["doctor"] == ["Anita Sharma", "Rahul Menon"]
    assert set(state.categories) == set(f.CATEGORICAL)


# ---------------------------------------------------------------- make_features

def test_features_are_booking_time_only():
    train = _visits(age=[10.0, np.nan])
    X = f.make_features(train, f.fit_prep(train))
    assert list(X.columns) == f.FEATURES
    assert not set(X.columns) & f.FORBIDDEN
    assert X["age"].notna().all() and X["age_missing"].tolist() == [0, 1]


def test_unseen_doctor_does_not_crash():
    train = _visits(doctor=["Anita Sharma", "Rahul Menon"])
    test = _visits(doctor=["Nikhil Verma", "Anita Sharma"])
    X = f.make_features(test, f.fit_prep(train))
    assert pd.isna(X["doctor"].iloc[0]) and X["doctor"].iloc[1] == "Anita Sharma"
    assert list(X["doctor"].cat.categories) == ["Anita Sharma", "Rahul Menon"]


def test_make_features_does_not_change_its_input():
    train = _visits(age=[np.nan])
    before = train.copy()
    f.make_features(train, f.fit_prep(_visits(age=[30.0])))
    pd.testing.assert_frame_equal(train, before)
