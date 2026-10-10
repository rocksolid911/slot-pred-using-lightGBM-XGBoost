"""Checks for your pinball_push and MiniBoost.fit in slotpred/mini_boost.py.

    .venv/Scripts/python.exe -m pytest tests/test_mini_boost.py -v
"""
import numpy as np
import pytest

from slotpred.mini_boost import MiniBoost, pinball_push


def test_push_follows_the_score():
    push = pinball_push([30, 12, 20], [20, 20, 20], 0.8)
    assert np.allclose(push, [0.8, -0.2, 0.0])          # late pushes up 0.8, idle pushes down 0.2
    assert np.allclose(pinball_push([25.0], [20.0], 0.67), [0.67])


def _two_kinds(n=600, seed=0):
    """Quick visits (about 10 min) and slow visits (about 30 min), told apart by one column."""
    rng = np.random.default_rng(seed)
    slow = rng.random(n) < 0.5
    y = np.where(slow, 30.0, 10.0) * rng.lognormal(-0.04, 0.28, n)
    return slow.astype(float).reshape(-1, 1), y, slow


def test_boosting_lowers_the_score_round_after_round():
    X, y, _ = _two_kinds()
    m = MiniBoost(0.8, n_rounds=40, learning_rate=0.2, max_leaf_nodes=2, min_samples_leaf=20).fit(X, y)
    assert m.start_ == pytest.approx(np.quantile(y, 0.8))
    assert len(m.rounds_) == 40 and len(m.history_) == 40
    assert m.history_[-1] < m.history_[0]
    assert all(b <= a + 1e-9 for a, b in zip(m.history_, m.history_[1:]))   # never gets worse on train


def test_ends_near_each_kinds_q_point():
    X, y, slow = _two_kinds()
    m = MiniBoost(0.8, n_rounds=60, learning_rate=0.2, max_leaf_nodes=2, min_samples_leaf=20).fit(X, y)
    pred = m.predict(X)
    assert pred[slow].mean() == pytest.approx(np.quantile(y[slow], 0.8), rel=0.05)
    assert pred[~slow].mean() == pytest.approx(np.quantile(y[~slow], 0.8), rel=0.05)


def test_predict_with_fewer_rounds_matches_the_history():
    X, y, _ = _two_kinds()
    m = MiniBoost(0.8, n_rounds=10, learning_rate=0.2, max_leaf_nodes=2, min_samples_leaf=20).fit(X, y)
    from slotpred.metrics import pinball_loss
    assert pinball_loss(y, m.predict(X, n_rounds=5), 0.8) == pytest.approx(m.history_[4])
