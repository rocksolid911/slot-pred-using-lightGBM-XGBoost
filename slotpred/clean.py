"""Lesson A4, pipeline steps 1-5: RULES that turn the raw export into one clean row per visit.

Every function here is a rule: its output for a row depends only on that row and fixed constants,
so it is safe to run before the split (notes/A04-prepare.md, section 1). Nothing here learns from data.

Two kinds of rule:
    feature rules  normalise_doctor, normalise_visit_type, fix_age
                   also run at prediction time (A11), so the API sees inputs exactly as training did
    target rules   add_minutes, target_ok
                   need consult_end, which only exists after the visit: historical data only

    .venv/Scripts/python.exe -m pytest tests/test_clean.py -v
"""
import pandas as pd

from .eda import load_raw
from .world import DOCTORS

# The hospital's doctor roster: name -> department. In a real project this comes from HR.
# Only names and departments are public; the doctor's speed in world.py is hidden and never used.
ROSTER = {d.name: d.dept for d in DOCTORS}
VISIT_TYPES = ("new", "follow-up", "review")

# Domain rule from A3: a consult under 1 minute or over 2 hours is a recording error, not a visit.
MIN_MINUTES, MAX_MINUTES = 1.0, 120.0
# A3 problem log #7: Whitefield recorded consult_start at check-in during these months (inclusive).
GLITCH_CLINIC, GLITCH_FROM, GLITCH_TO = "Whitefield", "2026-03-01", "2026-04-30"
PLACEHOLDER_AGE = 999


# ---------------------------------------------------------------- feature rules (TODO(you))

def normalise_doctor(name):
    """Map one spelling of a doctor's name to its ROSTER name, or None if it matches nobody.

    Spellings seen in the export (A3 problem log #2), for the roster name "Sanjay Patil":
        "Sanjay Patil"   "Dr. Sanjay Patil"   "DR SANJAY PATIL"   "Sanjay P."   " dr sanjay patil "

    Return None (don't raise) for a name that matches no roster doctor: at prediction time the API
    may receive a doctor the roster doesn't know yet, and that must not crash the service.

    Hints: strip spaces and lower-case first; remove a leading "dr." or "dr "; then compare with each
    roster name in lower case, and with abbreviated first/last-name forms.
    """
    if not isinstance(name, str):
        return None
    name_clean = name.strip().lower()
    if name_clean.startswith("dr. "):
        name_clean = name_clean[4:]
    elif name_clean.startswith("dr "):
        name_clean = name_clean[3:]
    matches = []
    for roster_name in ROSTER:
        roster_name_clean = roster_name.lower()
        first_name, last_name = roster_name_clean.split()
        aliases = {
            roster_name_clean,
            f"{first_name[0]}. {last_name}",
            f"{first_name} {last_name[0]}.",
        }
        if name_clean in aliases:
            matches.append(roster_name)
    # "D. Rao" fits both Deepak Rao and Divya Rao: guessing would silently give the visit to the
    # wrong doctor, so an ambiguous name counts as unrecognised
    return matches[0] if len(matches) == 1 else None


def normalise_visit_type(value):
    """Map one spelling to "new" | "follow-up" | "review", or None if it isn't recognisable.

    Spellings seen in the export: "New", "NEW", "new visit", "Follow-up", "follow up", "FOLLOW-UP",
    "followup", "Review", "report review", and the clean forms themselves.
    """
    if not isinstance(value, str):
        return None
    value_clean = value.strip().lower()
    if value_clean in {"new", "new visit"}:
        return "new"
    if value_clean in {"follow-up", "follow up", "followup"}:
        return "follow-up"
    if value_clean in {"review", "report review"}:
        return "review"
    return None


def fix_age(age: pd.Series) -> pd.Series:
    """Return a copy of `age` where the placeholder PLACEHOLDER_AGE (and anything else that can't be a
    human age: below 0 or above 120) becomes NaN. Real values, including 0 (babies), stay unchanged.

    This only marks the value as missing. Filling it in is a LEARNED step (features.py, A4 step 7).
    """
    age_clean = pd.to_numeric(age, errors="coerce")
    age_clean = age_clean.mask((age_clean < 0) | (age_clean > 120))
    return age_clean


# ---------------------------------------------------------------- target rules (TODO(you))

def add_minutes(df: pd.DataFrame) -> pd.Series:
    """The target: consult length in minutes = consult_end - consult_start (A1).

    The timestamp columns arrive as text like "2026-03-02 10:12:30". A missing or invalid timestamp,
    or a consult_end earlier than consult_start, gives NaN. Return a float Series aligned with df.index.
    """
    consult_start = pd.to_datetime(df["consult_start"], errors="coerce")
    consult_end = pd.to_datetime(df["consult_end"], errors="coerce")
    minutes = (consult_end - consult_start).dt.total_seconds() / 60
    return minutes.mask(minutes < 0)


def target_ok(df: pd.DataFrame) -> pd.Series:
    """True where the row's `minutes` can be trusted as a training or evaluation target.

    False when any of these holds (A3 problem log #5, #6, #7, #8):
        minutes is missing
        minutes < MIN_MINUTES or minutes > MAX_MINUTES       (1 and 120 themselves are OK)
        clinic is GLITCH_CLINIC and GLITCH_FROM <= date <= GLITCH_TO

    `date` is text "YYYY-MM-DD"; strings in this format compare correctly with <= and >=.
    """
    minutes = df["minutes"]
    ok = ~minutes.isna() & minutes.between(MIN_MINUTES, MAX_MINUTES)
    ok &= ~((df["clinic"] == GLITCH_CLINIC) & df["date"].between(GLITCH_FROM, GLITCH_TO))
    return ok


# ---------------------------------------------------------------- the pipeline (Claude)

def normalise_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the feature rules to a whole table. Works on training data and on API requests alike."""
    out = df.copy()
    # map each distinct spelling once, then broadcast: ~100 calls instead of ~44,000
    out["doctor"] = out["doctor"].map({v: normalise_doctor(v) for v in out["doctor"].dropna().unique()})
    out["visit_type"] = out["visit_type"].map(
        {v: normalise_visit_type(v) for v in out["visit_type"].dropna().unique()})
    out["age"] = fix_age(out["age"])
    return out


def clean(raw: pd.DataFrame):
    """Steps 1-5 of the A4 pipeline. Returns (clean_df, funnel).

    funnel is a table with one row per step: rows in, rows removed, rows out. The removed counts
    should match your A3 problem log.
    """
    steps = []

    def log(step, before, after):
        steps.append(dict(step=step, rows_in=before, removed=before - after, rows_out=after))

    n0 = len(raw)
    df = raw.drop_duplicates().reset_index(drop=True)                 # 1 duplicates (log #1)
    log("1 drop exact duplicates", n0, len(df))

    df = normalise_labels(df)                                          # 2-3 labels, placeholder age
    unmapped = df["doctor"].isna() | df["visit_type"].isna()
    log("2 unrecognised doctor / visit type", len(df), len(df) - unmapped.sum())
    df = df[~unmapped]

    df = df.assign(minutes=add_minutes(df))                            # 4 the target
    ok = target_ok(df)                                                 # 5 untrustworthy targets
    # breakdown by reason; a row with several problems is counted under the first one only
    missing = df["minutes"].isna()
    glitch = (df["clinic"] == GLITCH_CLINIC) & df["date"].between(GLITCH_FROM, GLITCH_TO) & ~missing
    out_of_range = ~df["minutes"].between(MIN_MINUTES, MAX_MINUTES) & ~missing & ~glitch
    for label, mask in [("5a consult_end missing", missing),
                        ("5b Whitefield Mar-Apr process change", glitch),
                        ("5c under 1 or over 120 min", out_of_range)]:
        steps.append(dict(step=label, rows_in=None, removed=int(mask.sum()), rows_out=None))
    log("5 total untrustworthy targets", len(df), int(ok.sum()))
    df = df[ok].reset_index(drop=True)

    return df, pd.DataFrame(steps)


def load_clean():
    """The cleaned visits, straight from the raw export."""
    return clean(load_raw())[0]
