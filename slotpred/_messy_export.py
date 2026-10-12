"""SPOILER - don't read this file until you've finished lesson A3 (EDA).

It injects the data-quality problems a real EMR export has. In A3 you find them yourself
by exploring the data, as you would on a real project. Afterwards, compare your list with this one.
"""
import numpy as np
import pandas as pd

TS = "%Y-%m-%d %H:%M:%S"


def export(world: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    df = world.copy()
    n = len(df)
    start, end, check_in = (df[c].astype("datetime64[ns]") for c in ("consult_start", "consult_end", "check_in"))

    # 1. Doctor forgot to close the visit -> hours-long "consults"
    forgot = rng.random(n) < 0.010
    end[forgot] = start[forgot] + pd.to_timedelta(rng.uniform(120, 300, forgot.sum()), unit="min")

    # 2. Start and end clicked together -> near-zero consults
    double = rng.random(n) < 0.005
    end[double] = start[double] + pd.to_timedelta(rng.uniform(0, 0.8, double.sum()), unit="min")

    # 3. Whitefield, March-April: reception clicked "start" at check-in (target definition broken, see A1)
    glitch = (df["clinic"] == "Whitefield") & df["date"].between("2026-03-01", "2026-04-30")
    start[glitch] = check_in[glitch]

    df["check_in"] = check_in.dt.strftime(TS)
    df["consult_start"] = start.dt.strftime(TS)
    df["consult_end"] = end.dt.strftime(TS).where(rng.random(n) >= 0.003, None)   # 4. end missing

    # 5. Doctor names typed differently by different staff
    messy = rng.random(n) < 0.04
    styles = [lambda s: "Dr. " + s, lambda s: "DR " + s.upper(),
              lambda s: s.split()[0] + " " + s.split()[1][0] + ".", lambda s: " dr " + s.lower() + " "]
    df.loc[messy, "doctor"] = [styles[i](s) for i, s in zip(rng.integers(0, 4, messy.sum()), df.loc[messy, "doctor"])]

    # 6. Visit type spelled inconsistently
    spelling = {"new": ["New", "NEW", "new visit"], "follow-up": ["Follow-up", "follow up", "FOLLOW-UP", "followup"]}
    odd = rng.random(n) < 0.03
    df.loc[odd, "visit_type"] = [rng.choice(spelling[v]) for v in df.loc[odd, "visit_type"]]

    # 7. Age missing, and a placeholder 999 for "unknown"
    age = df["age"].astype(float)
    age[rng.random(n) < 0.02] = np.nan
    age[rng.random(n) < 0.001] = 999
    df["age"] = age

    cols = ["appointment_id", "date", "slot_time", "clinic", "doctor", "department", "patient_id",
            "patient_name", "age", "visit_type", "prior_reports", "teleconsult",
            "check_in", "consult_start", "consult_end"]
    raw = df[cols]

    # 8. Some rows exported twice
    raw = pd.concat([raw, raw.sample(frac=0.004, random_state=int(rng.integers(1e9)))])
    return raw.sample(frac=1, random_state=int(rng.integers(1e9))).reset_index(drop=True)
