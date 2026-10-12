# A4 · Prepare: cleaning, features, and the split by time (lifecycle stage 3: Prepare)

> **Date:** 2026-10-09 · **Notebook:** [notebooks/A04_prepare.ipynb](../notebooks/A04_prepare.ipynb) · **Code:** [clean.py](../slotpred/clean.py), [split.py](../slotpred/split.py), [features.py](../slotpred/features.py), [prepare.py](../slotpred/prepare.py) · **Lab page:** [A4 Prepare Lab](https://claude.ai/artifact/1u2Bvgmm571WYeeZF7DTut) · **Git tag:** `lesson-A4`

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
| 2 | doctor and visit-type spellings | 2 | map to 21 doctors and 2 visit types |
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
| teleconsult, prior_reports | as they are | `check_in`, `consult_start`, `consult_end` |
| weekday, hour | `date`, `slot_time` | the raw `date` (test dates never appear in training) |

## Maths

### Why split by time
The test score should estimate **future** error:

$$
\text{target} = \mathbb{E}_{(x,y)\sim P_{\text{future}}}\big[L(y,\hat y)\big]
$$

A random split draws test rows from the mix of all months, $P_{\text{mix}} = \sum_m w_m P_m$, where $w_m$ is month $m$'s share of rows. If nothing changes over time, $P_{\text{mix}} = P_{\text{future}}$ and both estimates agree. **With drift they differ**, and the random split is optimistic, because rows from the test months sit in training.

**Worked example (the department × visit-type median, Experiment 1 below):**

| | Overall MAE | Dermatology MAE | Dermatology bias (actual − predicted) |
|---|---|---|---|
| Random 80/20 split | 3.33 | 2.65 | +0.54 min |
| Split by time (Jan–May → Jul–Aug) | 3.56 | 3.93 | **+3.58 min** |

The +3 minute Dermatology drift from 1 July (A2) is visible only with the split by time. Dr Nikhil Verma (joined 1 July, 532 visits) is a cold start only with the split by time, which is the deployed reality.

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
| Per-doctor median minutes (from the **target**), test MAE | 3.47 | 3.26 | looks 6% better than reality |

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

### The funnel, reconciled with the A3 problem log

| Step | Removed | A3 log | Why the counts differ |
|---|---|---|---|
| 1 exact duplicates | 177 | 177 | same |
| 2 unrecognised doctor / visit type | 0 | — | every spelling mapped |
| 5a `consult_end` missing | 120 | 120 | same |
| 5b Whitefield March–April | 1,704 | 1,712 | 5 were duplicates (gone in step 1); 3 had no `consult_end` (counted under 5a) |
| 5c under 1 or over 120 min | 668 | 690 | 4 were duplicates; 18 were also in the Whitefield window (counted under 5b) |
| **Total** | 44,514 → **41,845** clean visits | | |

### Splits

| Split | Rows | First | Last |
|---|---|---|---|
| train | 24,763 | 2026-01-01 | 2026-05-30 (31 May is a Sunday) |
| valid | 5,418 | 2026-06-01 | 2026-06-30 |
| test | 11,664 | 2026-07-01 | 2026-08-31 |

- `fit_prep` on train: age_fill = **42**; 8 departments, 2 visit types, **20** doctors (Nikhil Verma is absent), 5 clinics.
- Nikhil Verma: 0 / 0 / **532** visits; in `X_test` his `doctor` is NaN (an unseen level).
- `age_missing == 1`: 498 / 100 / 275 rows, all filled with 42.

### Experiment 1: random split vs split by time (department × visit-type median)

| | MAE | Derm MAE | Derm bias | Test rows |
|---|---|---|---|---|
| Split by time (Jan–May → Jul–Aug) | 3.56 | 3.93 | **+3.58** | 11,664 |
| Random 80/20 | **3.33** | 2.65 | +0.54 | 8,369 |
| Same random-split model, Jul–Aug rows only | 3.42 | 3.55 | +3.03 | 2,381 |

The random split's good score comes from easy pre-drift months in its test set. On the future months it's nearly as bad as the honest number; it only *hid* the drift. Row 3 is a little better than row 1 (3.42 vs 3.56, bias +3.03 vs +3.58) only because some July–August rows leaked into its training.

### Experiment 2: what the learned step may see

| Median fitted on… | Key | Test MAE | Derm MAE | Derm bias |
|---|---|---|---|---|
| train only (honest) | department × visit type | 3.56 | 3.93 | +3.58 |
| all months (leaky) | department × visit type | 3.51 | 3.50 | +2.92 |
| train only (honest) | doctor × visit type | 3.47 | 3.80 | +3.45 |
| all months (leaky) | doctor × visit type | 3.26 | 3.37 | +2.77 |

Age median: 42 on train, 42 on all months; no visible leak here, but the rule still applies.

## Gotchas
- **Ambiguous aliases map silently to the wrong row.** My extra `"S. Patil"` form made `"D. Rao"` match both Deepak Rao and Divya Rao, and the loop returned whichever came first. A matcher must return None when more than one name matches.
- **Don't name a variable after a function.** `time_split = median_baseline(...)` in the notebook replaced the imported `time_split()` function, so rerunning the split cell would crash.
- Parse fixed formats explicitly (`format="%H:%M"`, or `str[:2]`); letting pandas guess one value at a time is slow and fragile.
- Quick check Q4: the point of `age_missing` is **not** the fill value. After filling, "really 42" and "unknown" look identical; only the flag tells them apart.

## Self-check

<details><summary>Q1. "Drop visits above the 99th percentile of minutes": rule or learned? Where does it run?</summary>

Learned: the percentile comes from many rows. Compute it on the training rows only and apply that fixed number everywhere. It's also a target step, so it never runs at prediction time.
</details>

<details><summary>Q2. Why can't the API call <code>target_ok()</code>?</summary>

It needs `minutes`, which needs `consult_end`, which doesn't exist until the visit is over.
</details>

<details><summary>Q3. "Random split gives 3.33, better than your 3.56." Reply?</summary>

They measure different things. 3.33 is the error on a mix of months whose future rows leaked into training; 3.56 is the error on unseen future months, the deployed situation. The random split hides the Dermatology drift and the new doctor. The honest number is 3.56.

</details>

<details><summary>Q4. Why add <code>age_missing</code> instead of only filling with 42?</summary>

After filling, a patient who is really 42 and one whose age is unknown look identical. The flag lets the model learn whether "unknown" behaves differently (walk-ins, emergencies).
</details>

<details><summary>Q5. Why must the category list for <code>doctor</code> come from the training rows?</summary>

The model can only learn about levels it saw in training. A level first seen later (Nikhil Verma) becomes "unknown" instead of crashing, which is how the deployed model will meet every new doctor.
</details>

## In my words
the goal isn't low error on data you've seen, it's low average error on the visits that will arrive after deployment.for time-ordered data like slot bookings, split by time. A random split leaks the future into training and hides drift.train data for fitting, test data for evaluation, never both at once
- Rule (fixed, no data needed): normalise_doctor, normalise_visit_type or fix_age. They map "dr nikhil verma" to "Nikhil Verma" the same way for every row. A rule may run anywhere: before the split, in training, and in the API at prediction time.
- Learned step (computed from many rows): the age median used for filling, or the category list for doctor. It must be fitted on the training rows only, then the fixed result is applied to validation, test and live data
- target_ok checks minutes, which needs consult_end. That doesn't exist until the visit is over, so at booking time there is nothing to check. It's a target rule, used only to clean the training data.
- normalise_doctor works on a feature that exists at booking. If the API skipped it, "dr nikhil verma" would arrive unnormalised and be treated as an unknown level. That would be training–serving skew, so it must run at prediction time too.
- Every other doctor has a pace that isn't recorded anywhere. The model learns it from that doctor's past visits through the doctor feature, and dropping the feature would throw that away for the other 20 doctors.
- Nikhil is a cold start: his level wasn't in training, so he becomes "unknown" instead of crashing. That is how the deployed model will meet every new doctor, and the A12 monitoring step is meant to flag it.
- Duplicates of the same visit can land in both train and test.
- The model then gets scored on rows it has effectively already seen. That is leakage, and it makes the score look better than it is.
- The fix is to drop duplicates before splitting, which is how the pipeline does it (raw.drop_duplicates() in clean.py).
## New terms
rule vs learned step, fit/transform, training–serving skew, split by time, random split, drift, cold start, imputation, missing-indicator flag, unseen category, funnel
