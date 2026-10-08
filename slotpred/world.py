"""The simulated hospital: who works where, and the hidden rule for consult length.

    actual minutes = expected_minutes(...) x noise

The model only ever sees the CSV export. The rule below is the "truth" it tries to learn.
Lesson: notes/A02-simulator.md
"""
from dataclasses import dataclass

import numpy as np

# --- Base minutes per department: (new, follow-up, report review) ---------------
DEPARTMENTS = {
    "Cardiology":       (20, 9, 13),
    "Orthopaedics":     (14, 8, 10),
    "Paediatrics":      (16, 9, 10),
    "General Medicine": (12, 7, 9),
    "Dermatology":      (10, 6, 7),
    "ENT":              (12, 7, 8),
    "Gynaecology":      (18, 10, 12),
    "Neurology":        (24, 12, 15),
}
VISIT_TYPES = ("new", "follow-up", "review")

# --- ADDITIVE effects: extra *work*, the same minutes whatever the base -------------
REPORT_MIN = 1.3          # minutes to read each prior report
REPORT_CAP = 8            # beyond 8 reports the doctor skims; no extra time
INTERPRETER_MIN = 4.0     # translating adds roughly fixed minutes

# --- MULTIPLICATIVE effects: change the *pace* of the whole consult ----------------
ELDERLY_AGE, ELDERLY_FACTOR = 65, 1.15      # everything is slower: history, exam, explaining
INFANT_AGE, INFANT_FACTOR = 3, 1.15         # paediatrics only: babies are slower to examine
TELE_FACTOR = 0.80                          # video: no physical exam
CLINIC_FACTOR = {"HSR": 1.00, "Whitefield": 0.95, "Koramangala": 0.90, "Jayanagar": 1.05, "Indiranagar": 0.97}
DAY_FACTOR = {0: 0.94, 1: 1.00, 2: 1.00, 3: 1.00, 4: 1.00, 5: 1.05}   # Mon rush .. Sat relaxed; closed Sun
EARLY_HOUR, EARLY_FACTOR = 10, 1.05         # before 10:00, the doctor is fresh and thorough
LATE_HOUR, LATE_FACTOR = 16, 0.92           # from 16:00, running behind and hurrying

# --- Drift: from July, Dermatology adds a fixed skin-check checklist (ADDITIVE) ----
DRIFT_DATE, DRIFT_DEPT, DRIFT_EXTRA_MIN = "2026-07-01", "Dermatology", 3.0

# --- Noise -----------------------------------------------------------------------
NOISE_SIGMA = 0.28


@dataclass(frozen=True)
class Doctor:
    name: str
    dept: str
    speed: float          # pace multiplier: > 1 means slower than peers
    clinics: tuple
    joined: str = "2025-01-01"


DOCTORS = [
    Doctor("Deepak Rao", "General Medicine", 1.00, ("HSR", "Koramangala")),
    Doctor("Divya Rao", "Dermatology", 0.90, ("HSR", "Indiranagar")),
    Doctor("Anita Sharma", "Cardiology", 1.15, ("HSR", "Jayanagar")),
    Doctor("Rahul Menon", "Cardiology", 0.90, ("Whitefield", "Koramangala")),
    Doctor("Priya Nair", "Paediatrics", 1.10, ("HSR", "Whitefield")),
    Doctor("Suresh Kumar", "Orthopaedics", 0.85, ("Jayanagar", "Koramangala")),
    Doctor("Kavita Iyer", "Gynaecology", 1.05, ("Indiranagar", "HSR")),
    Doctor("Arjun Reddy", "Orthopaedics", 1.10, ("Whitefield", "HSR")),
    Doctor("Meera Pillai", "Neurology", 1.00, ("Koramangala", "Jayanagar")),
    Doctor("Vikram Shetty", "ENT", 0.95, ("Indiranagar", "Whitefield")),
    Doctor("Neha Gupta", "General Medicine", 1.20, ("Jayanagar", "Whitefield")),
    Doctor("Sanjay Patil", "Paediatrics", 0.90, ("Koramangala", "Indiranagar")),
    Doctor("Lakshmi Krishnan", "Gynaecology", 0.95, ("Jayanagar", "Whitefield")),
    Doctor("Imran Khan", "General Medicine", 0.85, ("Indiranagar", "HSR")),
    Doctor("Farah Siddiqui", "Neurology", 1.15, ("HSR", "Whitefield")),
    Doctor("Rohan Desai", "ENT", 1.05, ("Koramangala", "HSR")),
    Doctor("Asha Hegde", "Dermatology", 1.10, ("Jayanagar", "Koramangala")),
    Doctor("Karthik Subramanian", "Cardiology", 1.00, ("Indiranagar", "Jayanagar")),
    Doctor("Pooja Bhat", "Paediatrics", 1.00, ("Jayanagar", "Indiranagar")),
    Doctor("Manoj Joshi", "Orthopaedics", 1.00, ("HSR", "Indiranagar")),
    # Joins in July (drift): slow and new; no training data exists for him.
    Doctor("Nikhil Verma", "Paediatrics", 1.35, ("HSR", "Whitefield"), joined="2026-07-01"),
]

FIRST_NAMES = """Siddhant Aarav Vivaan Aditya Arjun Reyansh Krishna Ishaan Rohan Kabir Ananya Diya Aadhya
Saanvi Myra Ira Kiara Anika Riya Meera Rahul Amit Sunil Ravi Vijay Ramesh Suresh Mahesh Ganesh
Lakshmi Sunita Geeta Kavya Pooja Neha Sneha Divya Shreya Priya Arun Varun Nikhil Kiran Manoj
Imran Ayesha Farhan Zoya Joseph Mary Thomas Anil Prakash Harish Shalini Nandini Vandana Uma
Venkatesh Srinivas Raghav Tanvi Ishita Aryan Dev Yash Om Tara Asha Bhavna Chetan Deepa Esha""".split()


def expected_minutes(*, dept, visit_type, age, reports, interpreter, tele,
                     doctor_speed, clinic, weekday, hour, date):
    """The hidden rule: the EXPECTED consult length in minutes, before noise.

    Arguments
        dept         e.g. "Cardiology"            (a key of DEPARTMENTS)
        visit_type   "new" | "follow-up" | "review"
        age          patient age in years (int)
        reports      number of prior reports to read (int)
        interpreter  True if an interpreter is needed
        tele         True for a video consult
        doctor_speed the doctor's pace multiplier (Doctor.speed)
        clinic       e.g. "HSR"                    (a key of CLINIC_FACTOR)
        weekday      0 = Monday .. 5 = Saturday    (a key of DAY_FACTOR)
        hour         hour of the booked slot, 9..17
        date         "YYYY-MM-DD" (strings in this format compare correctly with < and >=)

    The design (lesson A2):
        work  = base + min(reports, REPORT_CAP) * REPORT_MIN + interpreter
                                                                  <- ADDITIVE (extra work)
        pace  = elderly x infant x tele x doctor x clinic x day x time-of-day   <- MULTIPLICATIVE
        drift = +3 min for Dermatology from July                   <- ADDITIVE, after the pace

        expected = work x pace + drift

    Use the constants defined above; don't hard-code numbers.
    """
    report_work = min(reports, REPORT_CAP) * REPORT_MIN
    work = DEPARTMENTS[dept][VISIT_TYPES.index(visit_type)] + report_work + (INTERPRETER_MIN if interpreter else 0)
    
    pace = 1
    if age >= ELDERLY_AGE:
        pace *= ELDERLY_FACTOR
    if dept == "Paediatrics" and age < INFANT_AGE:
        pace *= INFANT_FACTOR
    if tele:
        pace *= TELE_FACTOR
    pace *= doctor_speed
    pace *= CLINIC_FACTOR[clinic]
    pace *= DAY_FACTOR[weekday]
    if hour < EARLY_HOUR:
        pace *= EARLY_FACTOR
    elif hour >= LATE_HOUR:
        pace *= LATE_FACTOR

    drift = 0
    if dept == DRIFT_DEPT and date >= DRIFT_DATE:
        drift += DRIFT_EXTRA_MIN

    expected = work * pace + drift
    return expected


def sample_noise(rng: np.random.Generator, n: int, sigma: float = NOISE_SIGMA) -> np.ndarray:
    """n log-normal noise multipliers with MEAN exactly 1 (so noise doesn't bias the rule).

    If Z ~ Normal(mu, sigma^2), then e^Z is log-normal with mean e^(mu + sigma^2/2).
    Pick mu so that the mean is 1.  Hint: rng.lognormal(mean=..., sigma=..., size=n),
    where `mean` is mu, the mean of Z, NOT the mean of the result.
    """
    mu = -0.5 * sigma**2
    return rng.lognormal(mean=mu, sigma=sigma, size=n)
