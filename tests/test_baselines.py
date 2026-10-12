"""Checks for GroupQuantile in slotpred/baselines.py (your fit and predict with backoff).

    .venv/Scripts/python.exe -m pytest tests/test_baselines.py -v
"""
import numpy as np
import pandas as pd
import pytest

from slotpred.baselines import FixedSlot, GlobalQuantile, GroupQuantile


def _train():
    # Dermatology follow-ups: 10 visits of 5..14 min. Neurology new: 10 visits of 30..39 min.
    # ENT new: only 2 visits (a tiny box).
    rows = [("Dermatology", "follow-up", "Divya Rao", 5.0 + i) for i in range(10)]
    rows += [("Neurology", "new", "Meera Pillai", 30.0 + i) for i in range(10)]
    rows += [("ENT", "new", "Rohan Desai", m) for m in (100.0, 2.0)]
    return pd.DataFrame(rows, columns=["department", "visit_type", "doctor", "minutes"])


def _rows(*boxes):
    return pd.DataFrame(list(boxes), columns=["department", "visit_type", "doctor"])


def test_templates():
    train = _train()
    assert FixedSlot(15).fit(train).predict(_rows(("ENT", "new", "x"))).tolist() == [15.0]
    assert GlobalQuantile(0.8).fit(train).predict(_rows(("ENT", "new", "x")))[0] == pytest.approx(train.minutes.quantile(0.8))


def test_each_box_gets_its_own_q_point():
    train = _train()
    b2 = GroupQuantile(["department", "visit_type"], 0.8).fit(train)
    pred = b2.predict(_rows(("Neurology", "new", "a"), ("Dermatology", "follow-up", "b")))
    assert pred[0] == pytest.approx(np.quantile(np.arange(30, 40), 0.8))   # 37.2
    assert pred[1] == pytest.approx(np.quantile(np.arange(5, 15), 0.8))    # 12.2


def test_returns_numpy_in_row_order_even_with_odd_index():
    train = _train()
    df = _rows(("Dermatology", "follow-up", "b"), ("Neurology", "new", "a"))
    df.index = [41, 7]
    pred = GroupQuantile(["department", "visit_type"], 0.5).fit(train).predict(df)
    assert isinstance(pred, np.ndarray) and len(pred) == 2
    assert pred[0] < pred[1]                                  # Dermatology first, as in df


def test_unseen_box_backs_off_to_overall_q_point():
    train = _train()
    b2 = GroupQuantile(["department", "visit_type"], 0.8).fit(train)
    pred = b2.predict(_rows(("Cardiology", "new", "z")))     # no Cardiology in training
    assert pred[0] == pytest.approx(train.minutes.quantile(0.8))


def test_small_box_backs_off_to_parent():
    train = _train()
    parent = GroupQuantile(["department"], 0.8)
    b3 = GroupQuantile(["department", "visit_type"], 0.8, min_n=5, parent=parent).fit(train)
    ent = b3.predict(_rows(("ENT", "new", "c")))[0]         # box has 2 visits < 5
    assert ent == pytest.approx(parent.predict(_rows(("ENT", "new", "c")))[0])
    neuro = b3.predict(_rows(("Neurology", "new", "a")))[0]   # box has 10 visits >= 5: trusted
    assert neuro == pytest.approx(np.quantile(np.arange(30, 40), 0.8))


def test_parent_is_fitted_by_the_child():
    train = _train()
    parent = GroupQuantile(["department"], 0.8)
    GroupQuantile(["department", "visit_type"], 0.8, min_n=5, parent=parent).fit(train)
    assert parent.predict(_rows(("ENT", "new", "c")))[0] == pytest.approx(train[train.department == "ENT"].minutes.quantile(0.8))


def test_fit_learns_from_train_only():
    train = _train()
    b2 = GroupQuantile(["department", "visit_type"], 0.8).fit(train)
    before = b2.predict(_rows(("Neurology", "new", "a")))[0]
    later = _rows(("Neurology", "new", "a")).assign(minutes=500.0)   # predicting never changes what was learned
    assert b2.predict(later)[0] == before
