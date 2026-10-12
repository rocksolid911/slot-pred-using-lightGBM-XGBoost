"""Checks for your two TODO(you) functions in slotpred/world.py.

They test BEHAVIOUR (new > follow-up, reports are additive, ...), not exact numbers,
so you can change the constants in world.py and the tests still apply.

    .venv/Scripts/python.exe -m pytest tests/test_world.py -v
"""
import math

import numpy as np
import pytest

from slotpred import world as w


def minutes(**overrides):
    """expected_minutes for a 'neutral' visit: every pace multiplier is 1, so effects are easy to read."""
    args = dict(dept="General Medicine", visit_type="new", age=40, reports=0,
                tele=False, doctor_speed=1.0, clinic="HSR", weekday=1, hour=11, date="2026-03-10")
    args.update(overrides)
    return w.expected_minutes(**args)


# ---------------------------------------------------------------- expected_minutes

def test_neutral_visit_equals_base_minutes():
    for dept, (new, follow) in w.DEPARTMENTS.items():
        assert minutes(dept=dept, visit_type="new") == pytest.approx(new)
        assert minutes(dept=dept, visit_type="follow-up") == pytest.approx(follow)


def test_always_positive_and_plausible():
    for dept in w.DEPARTMENTS:
        for visit in w.VISIT_TYPES:
            for age in (1, 40, 80):
                for reports in (0, 3, 12):
                    m = minutes(dept=dept, visit_type=visit, age=age, reports=reports,
                                doctor_speed=1.35, weekday=5, hour=9)
                    assert 2 < m < 90, (dept, visit, age, reports, m)


def test_reports_are_additive_and_capped():
    one = minutes(reports=1) - minutes(reports=0)
    assert one == pytest.approx(w.REPORT_MIN)
    # additive: same extra minutes for a short and a long consult
    assert minutes(dept="Neurology", reports=1) - minutes(dept="Neurology", reports=0) == pytest.approx(one)
    # capped
    assert minutes(reports=w.REPORT_CAP + 5) == pytest.approx(minutes(reports=w.REPORT_CAP))


def test_elderly_is_multiplicative():
    for dept in ("Dermatology", "Neurology"):
        ratio = minutes(dept=dept, age=w.ELDERLY_AGE) / minutes(dept=dept, age=w.ELDERLY_AGE - 1)
        assert ratio == pytest.approx(w.ELDERLY_FACTOR)


def test_infant_factor_only_in_paediatrics():
    paed = minutes(dept="Paediatrics", age=1) / minutes(dept="Paediatrics", age=8)
    assert paed == pytest.approx(w.INFANT_FACTOR)
    assert minutes(dept="General Medicine", age=1) == pytest.approx(minutes(dept="General Medicine", age=8))


def test_age_boundaries():
    # "under 3" means a 3-year-old is NOT an infant; "65+" means a 65-year-old IS elderly
    assert minutes(dept="Paediatrics", age=w.INFANT_AGE) == pytest.approx(minutes(dept="Paediatrics", age=8))
    assert minutes(dept="Paediatrics", age=w.INFANT_AGE - 1) > minutes(dept="Paediatrics", age=8)
    assert minutes(age=w.ELDERLY_AGE) > minutes(age=w.ELDERLY_AGE - 1)


def test_pace_multipliers():
    base = minutes()
    assert minutes(tele=True) == pytest.approx(base * w.TELE_FACTOR)
    assert minutes(doctor_speed=1.2) == pytest.approx(base * 1.2)
    assert minutes(clinic="Koramangala") == pytest.approx(base * w.CLINIC_FACTOR["Koramangala"])
    assert minutes(weekday=0) == pytest.approx(base * w.DAY_FACTOR[0])
    assert minutes(hour=w.EARLY_HOUR - 1) == pytest.approx(base * w.EARLY_FACTOR)
    assert minutes(hour=w.LATE_HOUR) == pytest.approx(base * w.LATE_FACTOR)
    assert minutes(hour=w.EARLY_HOUR) == pytest.approx(base)          # 10:00 is not "early"


def test_slow_doctor_also_slows_extra_work():
    # pace multiplies the whole work, including reports
    slow = minutes(reports=2, doctor_speed=1.2)
    normal = minutes(reports=2)
    assert slow == pytest.approx(normal * 1.2)


def test_dermatology_drift_from_july():
    derm_before = minutes(dept="Dermatology", date="2026-06-30")
    derm_after = minutes(dept="Dermatology", date=w.DRIFT_DATE)
    assert derm_after - derm_before == pytest.approx(w.DRIFT_EXTRA_MIN)
    # drift is added after the pace: a slow dermatologist gets the same +3, not +3 x speed
    slow_gap = minutes(dept="Dermatology", date=w.DRIFT_DATE, doctor_speed=1.2) - \
        minutes(dept="Dermatology", date="2026-06-30", doctor_speed=1.2)
    assert slow_gap == pytest.approx(w.DRIFT_EXTRA_MIN)
    # other departments don't change
    assert minutes(dept="ENT", date="2026-08-01") == pytest.approx(minutes(dept="ENT", date="2026-06-30"))


# ---------------------------------------------------------------- sample_noise

@pytest.fixture
def noise():
    return w.sample_noise(np.random.default_rng(0), 400_000, sigma=0.28)


def test_noise_shape_and_positive(noise):
    assert noise.shape == (400_000,)
    assert (noise > 0).all()


def test_noise_mean_is_one(noise):
    assert noise.mean() == pytest.approx(1.0, abs=0.005)


def test_noise_is_right_skewed(noise):
    assert np.median(noise) < noise.mean()
    assert np.median(noise) == pytest.approx(math.exp(-0.28**2 / 2), rel=0.01)


def test_noise_p80_over_median(noise):
    # the ratio depends only on sigma: e^(0.8416 * sigma)
    ratio = np.quantile(noise, 0.8) / np.median(noise)
    assert ratio == pytest.approx(math.exp(0.8416 * 0.28), rel=0.01)


def test_zero_sigma_means_no_noise():
    assert np.allclose(w.sample_noise(np.random.default_rng(0), 10, sigma=0.0), 1.0)
