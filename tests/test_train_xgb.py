"""Checks for slotpred/train_xgb.py: same contract as the LightGBM trainer.

    .venv/Scripts/python.exe -m pytest tests/test_train_xgb.py -v
"""
import numpy as np
import pandas as pd
import pytest

from slotpred import train_xgb as t
from slotpred.metrics import coverage, pinball_loss


def test_make_model_learns_the_q_point():
    p = t.make_model(0.75, learning_rate=0.2).get_params()
    assert p["objective"] == "reg:quantileerror" and p["quantile_alpha"] == 0.75
    assert p["learning_rate"] == 0.2 and t.PARAMS["learning_rate"] == 0.05


def _data(n, seed):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"kind": pd.Categorical(rng.choice(["quick", "slow"], n)), "reports": rng.integers(0, 6, n)})
    y = (np.where(X["kind"] == "slow", 30.0, 10.0) + 1.3 * X["reports"]) * rng.lognormal(-0.04, 0.28, n)
    return X, pd.Series(y)


def test_fit_one_stops_early_and_is_calibrated():
    Xtr, ytr = _data(2000, 0)
    Xva, yva = _data(500, 1)
    m = t.fit_one(0.8, Xtr, ytr, Xva, yva, n_estimators=600, early_stopping_rounds=20, min_child_weight=10)
    assert 0 < m.best_iteration < 599
    pred = m.predict(Xva)
    assert pinball_loss(yva, pred, 0.8) < pinball_loss(yva, np.full(len(yva), ytr.quantile(0.8)), 0.8)
    assert coverage(yva, pred) == pytest.approx(0.8, abs=0.06)
