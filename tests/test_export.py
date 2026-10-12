"""The EMR export must never contain the simulator's hidden truth.

The model only ever sees data/raw_emr_export.csv. If a hidden column (the doctor's speed,
the expected minutes, the noise) slipped into it, a model could "learn" the answer
instead of the pattern. That's leakage from the simulator.
"""
import numpy as np
import pandas as pd

from slotpred import _messy_export

HIDDEN = {"doctor_speed", "expected_minutes", "noise", "actual_minutes", "weekday"}


def _tiny_world(n=200):
    rng = np.random.default_rng(0)
    start = pd.Timestamp("2026-03-02 10:00") + pd.to_timedelta(np.arange(n) * 15, unit="min")
    return pd.DataFrame({
        "appointment_id": [f"A{i:06d}" for i in range(n)], "date": "2026-03-02", "weekday": 0,
        "slot_time": "10:00", "clinic": "HSR", "doctor": "Deepak Rao", "department": "General Medicine",
        "doctor_speed": 1.0, "patient_id": "P00001", "patient_name": "Siddhant", "age": 34,
        "visit_type": "new", "prior_reports": 2, "teleconsult": False,
        "expected_minutes": 13.7, "noise": rng.lognormal(0, 0.28, n), "actual_minutes": 13.0,
        "check_in": start - pd.Timedelta(minutes=10), "consult_start": start,
        "consult_end": start + pd.Timedelta(minutes=13),
    })


def test_export_hides_the_truth():
    raw = _messy_export.export(_tiny_world(), np.random.default_rng(1))
    leaked = HIDDEN & set(raw.columns)
    assert not leaked, f"hidden simulator columns leaked into the export: {leaked}"


def test_export_has_no_duration_column():
    # the target must be derived from timestamps (A4), as in a real EMR
    raw = _messy_export.export(_tiny_world(), np.random.default_rng(1))
    assert not any("minute" in c or "duration" in c for c in raw.columns)
