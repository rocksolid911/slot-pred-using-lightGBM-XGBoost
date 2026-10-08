"""Generate 8 months of OPD consults and export them the way an EMR would.

    python -m slotpred.simulate

Writes
    data/raw_emr_export.csv     what the hospital would hand you (messy; explored in A3, cleaned in A4)
    data/_hidden_truth.csv      expected minutes per visit; only for scoring the oracle, never a feature
"""
from datetime import date, timedelta

import numpy as np
import pandas as pd

from . import _messy_export
from .config import DATA, SEED, SIM_END, SIM_START
from .world import DOCTORS, FIRST_NAMES, expected_minutes, sample_noise

SLOT_TIMES = [f"{h:02d}:{m:02d}" for h in range(9, 18) for m in (0, 15, 30, 45)]


def _make_patients(rng, n=6000):
    is_kid = rng.random(n) < 0.15
    return pd.DataFrame({
        "patient_id": [f"P{i:05d}" for i in range(n)],
        "patient_name": rng.choice(FIRST_NAMES, n),
        "age": np.where(is_kid, rng.integers(0, 16, n),
                        np.clip(rng.normal(46, 17, n), 16, 92).astype(int)),
        "interpreter": rng.random(n) < 0.06,
    })


def _make_appointments(rng, patients):
    kids = patients.index[patients.age < 16].to_numpy()
    adults = patients.index[patients.age >= 16].to_numpy()
    rows = []
    day, end = date.fromisoformat(SIM_START), date.fromisoformat(SIM_END)
    while day <= end:
        d = day.isoformat()
        if day.weekday() != 6:                                     # closed on Sunday
            for doc in DOCTORS:
                if d < doc.joined or rng.random() > 0.75:          # each doctor sits ~4-5 days a week
                    continue
                clinic = rng.choice(doc.clinics)
                n = min(rng.poisson(14), len(SLOT_TIMES))
                for t in sorted(rng.choice(SLOT_TIMES, n, replace=False)):
                    visit = rng.choice(["new", "follow-up", "review"], p=[0.35, 0.45, 0.20])
                    p = patients.loc[rng.choice(kids if doc.dept == "Paediatrics" else adults)]
                    if visit == "new":
                        reports = rng.poisson(1.2 + (1.5 if doc.dept in ("Cardiology", "Neurology") else 0))
                    elif visit == "follow-up":
                        reports = rng.poisson(0.6)
                    else:
                        reports = 1 + rng.poisson(2)
                    rows.append(dict(
                        date=d, weekday=day.weekday(), slot_time=t, clinic=clinic,
                        doctor=doc.name, department=doc.dept, doctor_speed=doc.speed,
                        patient_id=p.patient_id, patient_name=p.patient_name, age=int(p.age),
                        interpreter=bool(p.interpreter), visit_type=visit, prior_reports=int(reports),
                        teleconsult=bool(visit != "new" and doc.dept != "Paediatrics" and rng.random() < 0.10),
                    ))
        day += timedelta(days=1)
    df = pd.DataFrame(rows)
    df.insert(0, "appointment_id", [f"A{i:06d}" for i in range(len(df))])
    return df


def simulate(seed=SEED):
    """Return (world, raw): `world` is the clean truth, `raw` is the messy EMR export."""
    rng = np.random.default_rng(seed)
    world = _make_appointments(rng, _make_patients(rng))

    world["expected_minutes"] = [
        expected_minutes(dept=r.department, visit_type=r.visit_type, age=r.age, reports=r.prior_reports,
                         interpreter=r.interpreter, tele=r.teleconsult, doctor_speed=r.doctor_speed,
                         clinic=r.clinic, weekday=r.weekday, hour=int(r.slot_time[:2]), date=r.date)
        for r in world.itertuples()
    ]
    world["noise"] = sample_noise(rng, len(world))
    world["actual_minutes"] = world["expected_minutes"] * world["noise"]

    # The EMR records timestamps, not durations.
    booked = pd.to_datetime(world["date"] + " " + world["slot_time"])
    world["check_in"] = booked - pd.to_timedelta(rng.uniform(0, 20, len(world)), unit="min")
    world["consult_start"] = booked + pd.to_timedelta(rng.exponential(4, len(world)), unit="min")
    world["consult_end"] = world["consult_start"] + pd.to_timedelta(world["actual_minutes"], unit="min")

    raw = _messy_export.export(world, rng)
    return world, raw


def main():
    world, raw = simulate()
    DATA.mkdir(exist_ok=True)
    raw.to_csv(DATA / "raw_emr_export.csv", index=False)
    world[["appointment_id", "expected_minutes"]].to_csv(DATA / "_hidden_truth.csv", index=False)
    print(f"{len(world):,} visits, {world.date.min()} .. {world.date.max()}")
    print(f"wrote {DATA / 'raw_emr_export.csv'} ({len(raw):,} rows) and {DATA / '_hidden_truth.csv'}")
    return world, raw


if __name__ == "__main__":
    main()
