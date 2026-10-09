# A4 · Prepare: cleaning, features, and the split by time (lifecycle stage 3: Prepare)

> **Date:** 2026-10-09 · **Notebook:** [notebooks/A04_prepare.ipynb](../notebooks/A04_prepare.ipynb) · **Code:** [clean.py](../slotpred/clean.py), [split.py](../slotpred/split.py), [features.py](../slotpred/features.py), [prepare.py](../slotpred/prepare.py) · **Git tag:** `lesson-A4`

## In one line
**Rules may run anywhere; learned steps are fitted on the training rows only; and the split follows time**, so nothing from the future, and nothing recorded after booking, reaches the model.

## Intuition

### Two kinds of preparation step

| Kind | Definition | Examples | Before the split? |
|---|---|---|---|
| **Rule** | The output for a row depends only on **that row** plus fixed constants | drop duplicates, `"DR SANJAY PATIL"` → `Sanjay Patil`, 999 → missing, keep 1–120 min | ✅ yes |
| **Learned step** | The output depends on **statistics of many rows** | fill age with the median, per-doctor averages, percentile cut-offs, scaling | ❌ fit on **train only**, then apply everywhere |

**Test:** would the result for row 5 change if you deleted row 9,000? If yes, it's learned.

*Example:* "drop visits above the 99th percentile" is learned (the percentile comes from many rows). The 1–120 minute domain rule is a rule.

### Feature steps vs target steps

| | Feature steps | Target steps |
|---|---|---|
| Touch | inputs (doctor, age, hour…) | `minutes`, which needs `consult_end` |
| At training time | ✅ | ✅ |
| At prediction time (the API, A11) | ✅ **the same code** | ❌ the visit hasn't happened |

If training and prediction clean inputs differently, that's **training–serving skew**: the API sends `"Dr. Sanjay Patil"`, but the model only ever saw `Sanjay Patil`.

### The pipeline

```
raw ─► 1 duplicates ─► 2 labels ─► 3 age 999 ─► 4 minutes ─► 5 bad targets ─► 6 SPLIT BY TIME ─► 7 fit on train ─► 8 features
       └──────────────────── rules (row by row) ────────────────────┘                          └──── learned ────┘
```

| A3 log # | Problem | Step | Fix |
|---|---|---|---|
| 1 | duplicates | 1 | drop |
| 2 | doctor and visit-type spellings | 2 | map to 21 doctors and 3 visit types |
| 3 | age 999 | 3 | → missing |
| 4 | missing age | 7 | training-row median + `age_missing` flag |
| 5 | missing `consult_end` | 5 | drop (no target) |
| 6, 8 | under 1 or over 120 min | 5 | drop; never cap |
| 7 | Whitefield March–April | 5 | drop |

### Features: known at booking only

| Use | From | Never use |
|---|---|---|
| department, visit_type, doctor, clinic | normalised labels | `appointment_id` (it encodes the date) |
| age, age_missing | step 7 | `patient_id`, `patient_name` |
| interpreter, teleconsult, prior_reports | as they are | `check_in`, `consult_start`, `consult_end` |
| weekday, hour | `date`, `slot_time` | the raw `date` (test dates never appear in training) |

## Maths

### Why split by time
The test score should estimate **future** error:

$$
\text{target} = \mathbb{E}_{(x,y)\sim P_{\text{future}}}\big[L(y,\hat y)\big]
$$

A random split draws test rows from the mix of all months, $P_{\text{mix}} = \sum_m w_m P_m$, where $w_m$ is month $m$'s share of rows. If nothing changes over time, $P_{\text{mix}} = P_{\text{future}}$ and both estimates agree. **With drift they differ**, and the random split is optimistic, because rows from the test months sit in training.

**Worked example (the department × visit-type median, from the lesson's probe):**

| | Overall MAE | Dermatology MAE | Dermatology bias (actual − predicted) |
|---|---|---|---|
| Random 80/20 split | 3.57 | 2.84 | +0.68 min |
| Split by time (Jan–May → Jul–Aug) | 3.83 | 4.13 | **+3.66 min** |

The +3 minute Dermatology drift from 1 July (A2) is visible only with the split by time. Dr Nikhil Verma (joined 1 July, 500 visits) is a cold start only with the split by time, which is the deployed reality.

### Filling missing age: fill + flag, fitted on train

$$
\text{age}^{\text{filled}}_i=\begin{cases}\text{age}_i & \text{known}\\ m_{\text{train}}=\operatorname{median}(\text{age}_{\text{train}}) & \text{missing}\end{cases}
\qquad
\text{age\_missing}_i=\mathbb{1}[\text{age}_i\text{ missing}]
$$

| Symbol | Meaning |
|---|---|
| $m_{\text{train}}$ | median age of the **training** rows (42 here); the median because it's robust (A3) |
| $\text{age\_missing}_i$ | 1 if row $i$'s age was unknown; lets the model separate "really 42" from "unknown, filled with 42" |

**How much does fitting on all data leak?** It depends on the statistic:

| Statistic | Fit on train | Fit on all months | Effect |
|---|---|---|---|
| Median age (an input) | 42 | 42 | none here |
| Per-doctor median minutes (from the **target**), test MAE | 3.73 | 3.53 | looks 5% better than reality |

## Code
| Piece | Where | Written by |
|---|---|---|
| `normalise_doctor`, `normalise_visit_type`, `fix_age`, `add_minutes`, `target_ok` | [clean.py](../slotpred/clean.py) | me |
| `clean()` pipeline and funnel, `normalise_labels` | [clean.py](../slotpred/clean.py) | Claude |
| `time_split` | [split.py](../slotpred/split.py) | me |
| `random_split`, `describe` | [split.py](../slotpred/split.py) | Claude |
| `add_time_features`, `impute_age`, `fit_prep` | [features.py](../slotpred/features.py) | me |
| `make_features` (categories, unseen levels), `make_xy` | [features.py](../slotpred/features.py) | Claude |
| `prepare.main()`, `load_prepared()` | [prepare.py](../slotpred/prepare.py) | Claude |
| Experiments 1 and 2 | [A04_prepare.ipynb](../notebooks/A04_prepare.ipynb) | me |

## Our results
<!-- From your notebook run: the funnel, split sizes, both experiment tables. -->

## Gotchas
- Quick check Q4: the point of `age_missing` is **not** the fill value. After filling, "really 42" and "unknown" look identical; only the flag tells them apart.

## Self-check

<details><summary>Q1. "Drop visits above the 99th percentile of minutes": rule or learned? Where does it run?</summary>

Learned: the percentile comes from many rows. Compute it on the training rows only and apply that fixed number everywhere. It's also a target step, so it never runs at prediction time.
</details>

<details><summary>Q2. Why can't the API call <code>target_ok()</code>?</summary>

It needs `minutes`, which needs `consult_end`, which doesn't exist until the visit is over.
</details>

<details><summary>Q3. "Random split gives 3.57, better than your 3.83." Reply?</summary>

They measure different things. 3.57 is the error on a mix of months whose future rows leaked into training; 3.83 is the error on unseen future months, the deployed situation. The random split hides the Dermatology drift and the new doctor. The honest number is 3.83.
</details>

<details><summary>Q4. Why add <code>age_missing</code> instead of only filling with 42?</summary>

After filling, a patient who is really 42 and one whose age is unknown look identical. The flag lets the model learn whether "unknown" behaves differently (walk-ins, emergencies).
</details>

<details><summary>Q5. Why must the category list for <code>doctor</code> come from the training rows?</summary>

The model can only learn about levels it saw in training. A level first seen later (Nikhil Verma) becomes "unknown" instead of crashing, which is how the deployed model will meet every new doctor.
</details>

## In my words
<!-- Written by you, not Claude: 3 lines, as if teaching a friend. -->

## New terms
rule vs learned step, fit/transform, training–serving skew, split by time, random split, drift, cold start, imputation, missing-indicator flag, unseen category, funnel
