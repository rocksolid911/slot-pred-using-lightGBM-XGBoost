"""Lesson A5: how we score any slot prediction, from the simplest baseline to LightGBM.

Words (notes/A05-baselines.md):
    actual  y     how long the visit really took (the `minutes` column)
    pred    p     the predicted P-point for that visit, e.g. 16.6 minutes
    q             the P value as a fraction: 0.67, 0.75 or 0.80. A BUSINESS setting from the PO (A1),
                  never something we choose or learn; config.Q_CANDIDATES lists them
    late          minutes the visit ran past its slot   = max(y - p, 0)
    idle          minutes the doctor sat free            = max(p - y, 0)
    slot          what gets booked: p rounded UP to the 5-minute grid

    .venv/Scripts/python.exe -m pytest tests/test_metrics.py -v
"""
import numpy as np

from .config import SLOT_GRID


# ---------------------------------------------------------------- TODO(you)

def pinball_loss(y, pred, q):
    """The average score over all visits (lower is better):

        score of one visit = late x q + idle x (1 - q)

    y, pred: arrays (or Series) of the same length. Returns one float.
    Example, q = 0.8, pred = 20: a 30-min visit scores 10 x 0.8 = 8.0; a 12-min visit scores 8 x 0.2 = 1.6;
    the average of the two is 4.8.
    """
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    late = np.maximum(y - pred, 0)
    idle = np.maximum(pred - y, 0)
    return np.mean(late * q + idle * (1 - q))


def coverage(y, pred):
    """The share of visits that finished within their prediction (y <= pred), as a float between 0 and 1.
    For a well-made P80 it should be close to 0.80."""
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    return np.mean(y <= pred)


def to_slot(pred, grid=SLOT_GRID):
    """Round each prediction UP to the booking grid: 16.6 -> 20, 15 -> 15, 15.2 -> 20, 2 -> 5.
    Never below one block (grid minutes). Returns a float numpy array.

    Watch out: 15.000000001 from floating-point arithmetic should still book 15, not 20.
    Hint: np.ceil(pred / grid - 1e-9) * grid, then np.maximum with grid.
    """
    slot = np.asarray(pred, float)
    return np.maximum(np.ceil(slot / grid - 1e-9) * grid, grid)


def booked_metrics(y, slot):
    """What the clinic feels once the slots are booked. Return a dict:

        late_share   share of visits that ran past their slot (y > slot)
        late_min     average late minutes per visit   (visits that finished early count as 0)
        idle_min     average idle minutes per visit   (visits that ran late count as 0)
        slot_min     average booked slot length
    """
    y, slot = np.asarray(y, float), np.asarray(slot, float)
    late = np.maximum(y - slot, 0)
    idle = np.maximum(slot - y, 0)
    return {
        "late_share": np.mean(y > slot),
        "late_min": np.mean(late),
        "idle_min": np.mean(idle),
        "slot_min": np.mean(slot),
    }


# ---------------------------------------------------------------- Claude

def evaluate(y, pred, q, name=""):
    """One leaderboard row: the model's own P-point (pinball, coverage) and the booked slot (late, idle)."""
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    booked = booked_metrics(y, to_slot(pred))
    return {"method": name, "q": q, "pinball": pinball_loss(y, pred, q), "coverage": coverage(y, pred),
            "booked_on_time": 1 - booked["late_share"], **{f"booked_{k}": v for k, v in booked.items()}}
