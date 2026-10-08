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
Q_SAFE = 0.8      # P80, from the 4:1 overrun:idle cost ratio (lesson A1)
