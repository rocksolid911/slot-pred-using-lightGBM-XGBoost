"""Lesson A6, part 1: boosting by hand. A tiny LightGBM in about 20 lines, so the real one isn't a black box.

The recipe (notes/A06-lightgbm.md, ideas 1-2):
    start      every visit gets the same number: the q-point of all training visits (B1 from A5)
    each round 1. push: for each visit, which way should its prediction move?
                  +q       if the visit was LONGER than predicted (it would run late)
                  -(1-q)   if it was SHORTER (the doctor would sit idle)
                  0        if exactly right
               2. a small tree groups visits that need a similar push into boxes (its leaves)
               3. each box moves a small step (learning_rate) towards the q-point of its own errors
    predict    start + all the small steps of every round

LightGBM does exactly this, only faster and with smarter trees.

    .venv/Scripts/python.exe -m pytest tests/test_mini_boost.py -v
"""
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from .metrics import pinball_loss


# ---------------------------------------------------------------- TODO(you)

def pinball_push(y, pred, q):
    """For each visit, the direction that lowers the score (the 'gradient' in ML words).

        +q        where y > pred   (late: push the prediction up, strongly when q is high)
        -(1 - q)  where y < pred   (idle: push it down)
        0         where y == pred

    Returns a float numpy array. Example, q = 0.8: y = [30, 12, 20], pred = [20, 20, 20] -> [0.8, -0.2, 0.0]
    """
    # TODO(you)
    raise NotImplementedError


class MiniBoost:
    """A minimal quantile booster. X must be all-numeric (use to_numeric below)."""

    def __init__(self, q, n_rounds=100, learning_rate=0.1, max_leaf_nodes=8, min_samples_leaf=30):
        self.q, self.n_rounds, self.learning_rate = q, n_rounds, learning_rate
        self.max_leaf_nodes, self.min_samples_leaf = max_leaf_nodes, min_samples_leaf

    def fit(self, X, y):
        """TODO(you). The boosting loop. Store on self:

            self.start_     the q-point of y                       (np.quantile(y, self.q))
            self.rounds_    a list of (tree, values) pairs, one per round
            self.history_   the training score (pinball_loss) after each round

        Steps:
            y = np.asarray(y, float); pred = an array full of self.start_
            repeat self.n_rounds times:
                push   = pinball_push(y, pred, self.q)
                tree   = self._new_tree().fit(X, push)            # boxes of visits that need a similar push
                leaves = tree.apply(X)                            # which box each visit landed in
                values = leaf_values(leaves, y - pred, self.q)    # each box's q-point of its errors
                pred   = pred + self.learning_rate * step(leaves, values)
                remember (tree, values); append pinball_loss(y, pred, self.q) to the history
        Return self.
        """
        # TODO(you)
        raise NotImplementedError

    # ------------------------------------------------------------ Claude

    def _new_tree(self):
        return DecisionTreeRegressor(max_leaf_nodes=self.max_leaf_nodes, min_samples_leaf=self.min_samples_leaf,
                                     random_state=0)

    def predict(self, X, n_rounds=None):
        """start + learning_rate x (each round's box value). n_rounds: use only the first n rounds."""
        pred = np.full(len(X), self.start_, dtype=float)
        for tree, values in self.rounds_[:n_rounds]:
            pred = pred + self.learning_rate * step(tree.apply(X), values)
        return pred


def leaf_values(leaves, errors, q):
    """{box id: q-point of the errors (actual - prediction) of the visits in that box}.
    This is how far the box would have to move so that a fraction q of its visits fit."""
    s = pd.Series(np.asarray(errors, float)).groupby(np.asarray(leaves)).quantile(q)
    return s.to_dict()


def step(leaves, values):
    """Each visit's move: the value of the box it landed in."""
    return np.array([values[leaf] for leaf in leaves], dtype=float)


def to_numeric(X: pd.DataFrame) -> np.ndarray:
    """sklearn trees need numbers: category columns become their codes (unseen = -1)."""
    out = X.copy()
    for col in out.columns:
        if isinstance(out[col].dtype, pd.CategoricalDtype):
            out[col] = out[col].cat.codes
    return out.to_numpy(dtype=float)
