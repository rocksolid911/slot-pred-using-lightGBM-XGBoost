"""Checks for the TODO(you) functions in slotpred/metrics.py.

    .venv/Scripts/python.exe -m pytest tests/test_metrics.py -v
"""
import numpy as np
import pandas as pd
import pytest

from slotpred import metrics as m


# ---------------------------------------------------------------- pinball_loss

def test_pinball_the_two_visits_from_the_quick_check():
    assert m.pinball_loss([30.0], [20.0], 0.8) == pytest.approx(8.0)     # 10 late x 0.8
    assert m.pinball_loss([12.0], [20.0], 0.8) == pytest.approx(1.6)     # 8 idle x 0.2
    assert m.pinball_loss([30.0, 12.0], [20.0, 20.0], 0.8) == pytest.approx(4.8)   # the average


def test_pinball_is_zero_when_exact_and_accepts_series():
    y = pd.Series([10.0, 20.0, 30.0])
    assert m.pinball_loss(y, y, 0.67) == pytest.approx(0.0)


def test_pinball_weights_follow_q():
    # one minute late costs q, one minute idle costs 1 - q
    assert m.pinball_loss([21.0], [20.0], 0.67) == pytest.approx(0.67)
    assert m.pinball_loss([19.0], [20.0], 0.67) == pytest.approx(0.33)


# ---------------------------------------------------------------- coverage

def test_coverage():
    assert m.coverage([10, 20, 30, 40], [25, 25, 25, 40]) == pytest.approx(0.75)   # 40 <= 40 counts as on time


# ---------------------------------------------------------------- to_slot

def test_to_slot_rounds_up_to_the_grid():
    out = m.to_slot(np.array([16.6, 15.0, 15.2, 2.0, 20.01, 0.0]))
    assert out.tolist() == [20.0, 15.0, 20.0, 5.0, 25.0, 5.0]


def test_to_slot_ignores_tiny_float_noise():
    assert m.to_slot(np.array([0.1 * 3 * 50]))[0] == 15.0   # 15.000000000000002 books 15


def test_to_slot_other_grid():
    assert m.to_slot(np.array([11.0]), grid=10).tolist() == [20.0]


# ---------------------------------------------------------------- booked_metrics

def test_booked_metrics():
    y = np.array([30.0, 12.0, 20.0, 18.0])
    slot = np.array([20.0, 20.0, 20.0, 20.0])
    b = m.booked_metrics(y, slot)
    assert b["late_share"] == pytest.approx(0.25)          # only the 30-min visit
    assert b["late_min"] == pytest.approx(10 / 4)          # 10 late minutes spread over 4 visits
    assert b["idle_min"] == pytest.approx((8 + 0 + 2) / 4)
    assert b["slot_min"] == pytest.approx(20.0)


def test_evaluate_row():
    row = m.evaluate([30.0, 12.0], [16.6, 16.6], 0.8, name="demo")
    assert row["method"] == "demo" and row["q"] == 0.8
    assert row["booked_slot_min"] == pytest.approx(20.0)    # 16.6 booked as 20
    assert row["booked_on_time"] == pytest.approx(0.5)
