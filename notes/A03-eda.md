# A3 · Exploring the messy export (lifecycle stage 2: Explore)

> **Date:** 2026-10-09 · **Notebook:** [notebooks/A03_eda.ipynb](../notebooks/A03_eda.ipynb) · **Helpers:** [slotpred/eda.py](../slotpred/eda.py) · **Lab page:** [A3 EDA Lab](https://claude.ai/artifact/Qwyogp2WiEn824VPTSAWMK) · **Git tag:** `lesson-A3`

## In one line
**Look before you clean, and write everything down with numbers.** Statistics find what's *unusual*, domain rules find what's *impossible*, and only time plots find *process changes*. None of them alone finds everything.

## Intuition

### Why EDA comes before cleaning
1. You can't fix a problem you haven't found; whatever you miss goes straight into the model.
2. Some "fixes" destroy real information. Deleting every visit over 60 minutes also deletes the genuine long neurology consults the model most needs.

**Rule: look, don't change.** Every finding goes into a **problem log** (problem, evidence, rows, cause, fix to try). Fixing is A4's job.

### The EDA checklist (any dataset)

| # | Check | Question | Tools |
|---|---|---|---|
| 1 | Shape and types | Is each column stored as what it really is? | `.shape`, `.dtypes`, `.head()` |
| 2 | Keys and duplicates | Does each ID mean exactly one thing? | `.duplicated()` |
| 3 | Missing values | How much is missing, and *which rows*? | `.isna().mean()`, `groupby` |
| 4 | Categories | Are the labels consistent? | `.nunique()`, `.value_counts()` |
| 5 | Numbers | Impossible or placeholder values? | `.describe()`, sorting |
| 6 | The target | Does what we predict look sensible? | log histogram, IQR fences, domain rules |
| 7 | Consistency | Stable across time and places? | groupby month × clinic, plots |
| 8 | Leakage smells | Is each column known at prediction time? | thinking |

## Maths

### Robust statistics
One bad row among 11 (a visit left open for 240 minutes):

| | Mean | Median |
|---|---|---|
| 10 real visits `8 … 27` | 14.0 | 12.5 |
| + one 240-min mistake | **34.5** (+146%) | **13.0** |

The median depends only on the *middle position*, so a few extreme rows can't drag it. EDA describes data with **medians, quantiles and the IQR**.

### IQR fences, and why on the log scale
$$
\text{IQR} = Q_3 - Q_1, \qquad \text{flag outside } \big[\,Q_1 - 1.5\,\text{IQR},\; Q_3 + 1.5\,\text{IQR}\,\big]
$$
The rule assumes symmetric data. Consult times are right-skewed (log-normal, A2), so apply it to $\log(\text{minutes})$ and convert the fences back with $e^x$.

**Worked example** (the 10 visits):
- raw: $Q_1 = 10.25$, $Q_3 = 15.5$, IQR $= 5.25$ → fences **[2.4, 23.4]**, which flags the genuine 27-minute visit ❌
- log: $Q_1 = 2.327$, $Q_3 = 2.740$, IQR $= 0.413$ → $[e^{1.707}, e^{3.359}]$ = **[5.5, 28.8]** min, so 27 is fine ✅

### Three kinds of missing data

| Type | Missingness depends on… | Example | Drop the rows? |
|---|---|---|---|
| **MCAR** | nothing | random export glitch | ✅ safe; you only lose data |
| **MAR** | something you **can see** (another column) | Whitefield's broken months (clinic + month) | ⚠️ the overall mix changes, but relationships *given that column* stay right, so a model using the column learns correctly |
| **MNAR** | the **missing value itself** | end time missing *because* the visit was very long | ❌ biased: the model would book slots that are too short everywhere |

**General rule:** dropping rows biases a model only when the reason for dropping is related to the **target** (beyond what the features explain).

## Code
| Piece | Where | Written by |
|---|---|---|
| `load_raw`, `log_hist`, `monthly_median` | [slotpred/eda.py](../slotpred/eda.py) | Claude |
| Checks 2–7 investigation, target derivation, IQR fences, timestamp-order checks | [A03_eda.ipynb](../notebooks/A03_eda.ipynb) | me |
| Problem log, leakage table, wrap-up | same notebook | me |

## Our results

### My problem log

| # | Problem | Evidence | Rows | Likely cause | Fix to try in A4 |
|---|---|---|---|---|---|
| 1 | Exact duplicate rows | 175 rows copy another row; all repeated `appointment_id`s identical | 175 (0.4%) | Export ran twice | Drop duplicates, keep one |
| 2 | Inconsistent labels | `doctor`: 105 spellings for 21 doctors; `visit_type`: 12 for 3 | ~1.4k visit types + ~4% of doctors | Free-text entry | Normalise and map to the 21 doctors and 3 visit types |
| 3 | Placeholder age 999 | max age 999 (41 rows); age 0 in 407 rows **(all Paediatrics: real babies)** | 41 | "Unknown" code | 999 → missing, then impute |
| 4 | Missing age | 863 NaN (1.96%), flat across clinics; also why `age` is float | 863 | MCAR-like | Impute (+ missing flag) |
| 5 | Missing `consult_end` | 100 NaN (0.23%), flat across clinics and months | 100 | MCAR-like | Drop from training (no target) |
| 6 | Impossible durations | 214 under 1 min (min 2 s); 455 over 120 min (up to ~300) | 669 | **Two causes:** start and end clicked together (< 1 min); doctor forgot to close the visit (> 2 h) | Drop the target (don't cap) |
| 7 | Whitefield process change | `consult_start == check_in` in 1,753 rows, all Whitefield, March (936) + April (817); median 27–28 vs ~12 | 1,753 | Start time recorded at check-in, so target = wait + consult | Drop these rows from training |
| 8 | End == start | 4 rows with 0 minutes | 4 | Same as the < 1 min rows in #6 | Covered by #6 |

### Answer key (`slotpred/_messy_export.py`, now safe to read)

| Injected problem | Rows | My log |
|---|---|---|
| Forgot to close (120–300 min) | ~1% | #6 ✅ |
| Start and end clicked together (< 1 min) | ~0.5% | #6, #8 ✅ |
| Whitefield Mar–Apr: start = check-in | 1,753 | #7 ✅ |
| `consult_end` missing | 100 | #5 ✅ |
| Doctor-name variants | ~4% | #2 ✅ |
| Visit-type spellings | ~3% | #2 ✅ |
| Age missing, and 999 | 863 + 41 | #3, #4 ✅ |
| Duplicate rows | 175 | #1 ✅ |

**8 of 8 found.**

### Outlier rules scored against the hidden truth (43,822 rows with a target; 2,389 wrong)

| Rule | Keeps | Flagged | Errors caught | Genuine visits thrown away | Errors missed (Whitefield) |
|---|---|---|---|---|---|
| Raw IQR | ≤ 31.3 min | 2,204 | 1,018 | **1,186** | 1,371 (1,157) |
| Log IQR | 2.8–52.7 min | 782 | 703 | 79 | 1,686 (1,686) |
| Domain (1–120 min) | 1–120 min | 669 | **669** | **0** | 1,720 (1,720) |

**No rule catches the Whitefield rows**: they look like plausible 20–35-minute visits. Only the time plot (Check 7) and the timestamp check found them.

## Corrections from review
- **Capping is worse than dropping.** Capping a forgotten visit at 120 min invents a fake 120-minute consult; its true length is unknown. Drop its target.
- **Dropping Whitefield's rows is not biased**, contrary to my first answer. The rows are chosen by clinic and month (MAR), which are unrelated to how long each visit really took, so the relationships the model learns stay correct; we only lose data (two months of one clinic). Re-estimating the start from a "typical duration" would **invent the target** (the model would learn our guesses), so don't.
- **Why `age` is float:** NaN is a float value and an int column can't hold it, so pandas stores the whole column as float64. Nothing is "filled".
- **`appointment_id`:** unique means useless as a feature. It also runs in time order (A000014 on 1 Jan, A043622 on 31 Aug), so it secretly encodes the date.
- **`date`, `slot_time`:** known at booking, but use what they *mean* (weekday, hour) rather than the raw date. Every test date is later than every training date, so a model can't use a date it has never seen. (A4.)
- **Impute using training rows only** (A4): medians computed on all data leak test information into training.
- **Look, don't change:** my Check 1 cell modified `raw` and left a hidden `slot_minutes` column. In EDA, work on a copy.

## Gotchas
- Raw-scale IQR on skewed data throws away genuine long visits (1,186 here). Use the log scale, and domain rules for what is impossible.
- Statistics find unusual values; **systematic** errors (a process change) look normal. Plot medians over time and by place.
- Age 0 looked suspicious, but all 407 rows are Paediatrics: real babies. Check before calling something an error.
- "Too good" or "too strange" numbers deserve a groupby by clinic and month before any fix.

## Self-check

<details><summary>Q1. Why use the median rather than the mean in EDA?</summary>

One bad row can drag the mean a long way (14.0 → 34.5 with one 240-min row); the median barely moves (12.5 → 13.0).
</details>

<details><summary>Q2. Why apply the IQR rule to log(minutes)?</summary>

Consult times are right-skewed; on the raw scale the upper fence is too low and flags genuine long visits. Logs remove the skew.
</details>

<details><summary>Q3. Which outlier rule caught the Whitefield problem?</summary>

None. The rows look plausible. The monthly median per clinic plot and the check `consult_start == check_in` found it.
</details>

<details><summary>Q4. End times are missing mostly for very long visits. MCAR, MAR or MNAR? Can you drop them?</summary>

MNAR. Dropping them removes the long visits, so the model would learn that visits are shorter than they are and book slots that are too short.
</details>

<details><summary>Q5. Rows are missing for one clinic in two months. Does dropping them bias the model?</summary>

No: missingness depends on observed columns (MAR), unrelated to the visit's length. The model learns correctly from what's left; you lose data, not correctness.
</details>

<details><summary>Q6. Is `check_in` a valid feature?</summary>

No. It's recorded when the patient arrives, after booking.
</details>

## In my words
<!-- Siddhant: 3 lines, your own words. -->

## New terms
EDA, problem log, robust statistics, IQR, fences, domain rule, MCAR, MAR, MNAR, placeholder value, process change, hidden state
