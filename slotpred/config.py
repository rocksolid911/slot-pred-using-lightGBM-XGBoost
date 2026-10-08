"""Paths, dates and constants shared by the whole project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"

SEED = 7

# Three-way split by time (see notes/00-ml-workflow.md, stage 3).
# Train on the past, tune on the next month, test once on the future.
SIM_START = "2026-01-01"
TRAIN_END = "2026-05-31"
VALID_START, VALID_END = "2026-06-01", "2026-06-30"
TEST_START, SIM_END = "2026-07-01", "2026-08-31"

SLOT_GRID = 5     # schedulers book in 5-minute blocks
# Which quantile to book is a BUSINESS setting, not a constant (lesson A1, "What the PO answered").
# The PO rejected 4:1, so the true ratio is probably 2:1-3:1. We train all candidates and let the
# PO choose from a trade-off table in A8. Never hard-code 0.8 anywhere else; use these names.
Q_CANDIDATES = (0.67, 0.75, 0.80)
Q_SAFE = 0.75     # provisional default until the PO picks from the A8 table
