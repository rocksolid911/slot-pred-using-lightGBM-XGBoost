# EDA playbook: exploring any dataset

A reusable flow for exploratory data analysis (lifecycle stage 2 in [00-ml-workflow.md](00-ml-workflow.md)). It works for any table, and the last part adapts it to other kinds of data. Every step shows **what we did in this project** ([A3](A03-eda.md)), so you have a worked example next to the general rule.

## What EDA is for

**EDA = understanding the data, and writing down its problems, before changing anything.** It answers three questions:

1. **Can I trust it?** Duplicates, missing values, impossible values, inconsistent labels, broken processes.
2. **What does it look like?** Distributions, especially of the target; how things change over time and across groups.
3. **What can I legally use?** Which columns exist at prediction time (leakage), and which look predictive.

The output is not a cleaned dataset. It's a **problem log** plus **decisions for the Prepare stage**.

## Seven rules

| # | Rule | Why | In our project |
|---|---|---|---|
| 1 | **Look, don't change** | Fixing while exploring destroys the evidence and mixes stages | My Check 1 cell modified `raw` and left a hidden column; lesson learned |
| 2 | **Write every finding with numbers** | "Some ages are odd" can't be acted on; "41 rows have age 999" can | The problem log in [A03_eda.ipynb](../notebooks/A03_eda.ipynb) |
| 3 | **Use robust statistics** | One bad row can move the mean a lot; the median barely moves | 14.0 → 34.5 mean vs 12.5 → 13.0 median with one 240-min row |
| 4 | **Ask *where*, not just *how many*** | The same missing rate can be harmless (random) or fatal (one clinic, or only long visits) | Missing age was flat across clinics, so random |
| 5 | **Plot over time and across groups** | Process changes look like valid data in any single summary | The Whitefield step appeared only in the monthly plot |
| 6 | **Unusual ≠ wrong** | Statistics find the unusual; only domain knowledge says what's impossible | The raw IQR rule threw away 1,186 genuine visits; the domain rule threw away none |
| 7 | **Don't form ideas from the test set** | Patterns you notice in the test period leak your judgement into the model | See the note under step 7 |

## The flow

```
1 Context ─► 2 Structure ─► 3 Completeness ─► 4 Validity ─► 5 Target ─► 6 Consistency ─► 7 Relationships ─► 8 Leakage audit ─► 9 Write-up
 (what is      (types,        (missing,          (labels,       (derive,     (time, groups,     (target vs           (known at          (problem log,
  a row?)       keys, dups)    placeholders)      ranges)        shape,       cross-field        features)            prediction        decisions,
                                                                 outliers)    rules)                                  time?)            questions)
```

Steps 2–4 are about **trust**, 5–7 about **shape and signal**, 8 about **what's allowed**. In practice you loop back: a finding in step 6 often sends you back to step 3.

A quick start for steps 2–4 on any table: `from slotpred.eda import first_look; first_look(df, date_col="date")`. It prints duplicates, missing % and distinct counts per column, numeric ranges (placeholders show up as the max), the values of text columns (spelling variants show up here) and rows per month.

---

### 1 · Context: what is a row, and when is the prediction made?

**Questions**
- What does **one row** represent (a visit? a patient? a day)? Can one real thing appear in several rows?
- What is the **target**, and how is it recorded? Is there a precise definition?
- What is the **prediction moment**? Everything after it is off-limits as a feature.
- Is there a **data dictionary**? Who can answer questions about the source system?

**Red flags:** nobody can define the target precisely; the same column means different things in different sources.

**In our project:** one row = one consult. Target = `consult_end − consult_start`, the time the doctor is occupied (A1). Prediction moment = **booking**, days before the visit. The hospital's reference list: 21 doctors, 8 departments, 5 clinics, 3 visit types.

### 2 · Structure: types, keys, duplicates

**Questions:** Is each column stored as what it really is? Is the ID unique? Are rows duplicated?

```python
df.shape, df.dtypes, df.head()
df.duplicated().sum()                       # exact duplicate rows
df["id"].duplicated().sum()                 # repeated IDs: exact copies, or conflicting versions?
```

**Red flags:** dates stored as text; numbers stored as float for no reason (usually hiding missing values); an ID that isn't unique; repeated IDs with *different* contents (conflicting versions are worse than copies).

**In our project:** timestamps were text; `age` was float because of NaN; **175 exact duplicate rows**, all identical copies.

### 3 · Completeness: missing values and placeholders

**Questions:** How much is missing per column? **Where** is it missing? Are there fake values meaning "unknown"?

```python
df.isna().mean().sort_values(ascending=False)                     # missing share per column
df.assign(miss=df["col"].isna()).groupby("group")["miss"].mean()  # where is it missing?
df["num"].value_counts().head(); df["num"].describe()             # 999, -1, 0, "N/A", "unknown"
```

**Decide the mechanism**, because it decides the fix:

| Mechanism | Missingness depends on… | Test | Safe to drop rows? |
|---|---|---|---|
| **MCAR** | nothing | Rate is flat across every group and over time | Yes; you only lose data |
| **MAR** | an observed column | Rate differs by group or period | Yes for the model's relationships (it can use that column), but the overall mix shifts |
| **MNAR** | the missing value itself | You often can't test it from the data; ask how the data is recorded | **No**: biased |

**In our project:** age missing 1.96%, flat across clinics (MCAR); `consult_end` missing 0.23%, flat across months (MCAR); **age 999** in 41 rows (placeholder). Age 0 looked suspicious, but all 407 rows were Paediatrics: real babies.

### 4 · Validity: categories, ranges, units

**Questions:** Do category labels match the reference list? Are numbers inside physically possible ranges? Are units consistent?

```python
df["cat"].nunique(); df["cat"].value_counts()             # more levels than expected = spelling variants
df.groupby("child")["parent"].nunique()                    # hierarchy: each doctor in exactly one department?
df["num"].describe(percentiles=[.01, .99])                 # extremes; 1% and 99% hide less than min and max
```

**Red flags:** more distinct values than the reference list; case or spacing variants; a child category under two parents; values outside physical limits; a mix of units (minutes and hours, cm and inches).

**In our project:** **105 doctor spellings** for 21 doctors; **12 visit-type spellings** for 3.

### 5 · The target

**Questions:** Can the target be computed for every row? What's its shape: skewed, multi-modal, heavy-tailed? Which values are *impossible*, and which are just *unusual*?

```python
y = (end - start).dt.total_seconds() / 60          # derive it, then count NaN, <= 0, below/above domain limits
np.log(y).describe()                               # skewed targets: look on the log scale
# IQR fences on log(y): unusual. Domain rule: impossible.
```

| Rule | Finds | Use it to |
|---|---|---|
| IQR fences on the raw scale | Unusual values, assuming symmetric data | Avoid on skewed targets: it flags genuine long values |
| IQR fences on log scale | Unusual values in skewed data | Make a "look at these" list |
| **Domain rule** | **Impossible** values | Remove what can't be real |

For **classification** targets, check class balance and label consistency instead.

**In our project:** 100 rows with no target; **214 under 1 minute** (start and end clicked together) and **455 over 2 hours** (visit never closed). The domain rule caught all 669 impossible rows and no genuine ones.

### 6 · Consistency: over time, across groups, across fields

**Questions:** Is the data stable month to month? Does any group behave differently in a way that looks like a recording change? Do related fields obey their logical rules?

```python
# median of the target per month x group: a step = a process change
df.groupby([df["date"].dt.to_period("M"), "group"])["y"].median().unstack()
# cross-field rules: the order of events, totals that should add up, ...
(df["start"] < df["check_in"]).sum(); (df["end"] <= df["start"]).sum(); (df["start"] == df["check_in"]).sum()
# volume over time: gaps or jumps in rows per day or month
```

**Red flags:** a step change in one group; a jump in volume; fields that are suddenly equal (one copied into the other); rules broken only in one place or period.

**In our project:** **Whitefield, March–April**: `consult_start == check_in` in 1,753 rows, which doubled the median (12 → 27 min). *No outlier rule caught it*, because every row looked plausible.

### 7 · Relationships: what looks predictive?

**Questions:** How does the target vary with each feature? Which features look strong, which useless? Are features near-duplicates of each other? How many levels does each category have (high cardinality)?

```python
df.groupby("feature")["y"].median().sort_values()           # one feature at a time
df.pivot_table(index="a", columns="b", values="y", aggfunc="median")   # interactions
df[numeric_cols].corr(method="spearman")                    # rank correlation: robust to skew
```

**Red flags:** a feature that predicts the target *almost perfectly* (suspect leakage first); two features carrying the same information; categories with thousands of rare levels.

**⚠️ Explore relationships on the training period only.** Data-quality checks (steps 2–6) can cover all the data, because problems must be fixed everywhere. But *ideas* about which features matter should come from training data only, or you're quietly tuning on the test set.

**In our project:** partly done so far: the department × visit-type medians in the A2 sanity checks, and the monthly medians. We'll do this properly when building features in A4, on **January–May only**.

### 8 · Leakage audit

For **every column**, answer: *is it known at the prediction moment?*

| Smell | Example | Why it's a problem |
|---|---|---|
| Recorded after the prediction moment | `check_in`, `consult_start`, `consult_end` | Not available when booking |
| The target, or built from it | duration computed from `consult_end` | Answers the question with the answer |
| IDs | `appointment_id`, `patient_id` | Identify rows rather than describe them; IDs in time order secretly encode the date; patient IDs let the model memorise people |
| Too good to be true | a feature with near-perfect correlation | Usually a hidden copy of the target |
| Duplicates across the split | the same visit in train and test | The model is tested on what it has seen |
| Aggregates over all data | "doctor's average minutes" over every row | Includes the row's own target and the future |

**In our project:** allowed at booking: date and slot time (as weekday and hour), clinic, doctor, department, age, interpreter, prior reports, visit type, teleconsult. Not allowed: `check_in`, `consult_start`, `consult_end`, both IDs, the patient's name.

### 9 · Write-up

Three outputs:

1. **Problem log:** each problem with evidence (numbers, an example), rows affected, likely cause and a fix to try.
2. **Decisions for Prepare:** what to drop, map, impute, flag or derive, and in what order.
3. **Questions for the data owners:** "Why did Whitefield record start at check-in in March–April?" Only they can confirm a cause.

---

## From problem to fix: a decision guide

| Problem | Usual fix | Watch out for |
|---|---|---|
| Exact duplicates | Drop, keep one | Conflicting duplicates need a rule (latest? most complete?) |
| Placeholder values (999, -1) | Convert to missing | Then treat as missing data |
| Missing, MCAR | Drop rows, or impute plus a "was missing" flag | Compute impute values on **training rows only** |
| Missing, MAR | Impute using the related column, or drop | The overall mix shifts; check groups still have enough data |
| Missing, MNAR | Don't just drop. Investigate the source; consider modelling it | Dropping removes exactly the cases you care about |
| Inconsistent labels | Normalise (trim, lower-case) and map to the reference list | Ambiguous variants (initials shared by two people) need a manual check |
| Impossible target | Drop the **target** for those rows | Don't cap: capping invents values |
| Process change for a period or group | Drop the affected rows, or flag them and fix the source | Never re-estimate the target from a "typical" value |
| Leaky column | Remove it from the features | Also remove anything derived from it |

## Adapting the flow to other kinds of data

The nine steps stay the same; the checks change.

| Data | Extra things to check |
|---|---|
| **Classification** | Class balance; label noise (the same input with different labels); rare classes |
| **Time series** | Gaps and duplicate timestamps; seasonality (weekday, month); trend; sudden level shifts; time zones |
| **Text** (Track B) | Length distribution in tokens; duplicates and near-duplicates; languages; empty or boilerplate entries; personal data |
| **LLM training data** (Track B) | Deduplication; **contamination** (test examples inside the training corpus); prompt and answer format consistency; answer distribution (e.g. are "15 min" answers over-represented?) |
| **Images** | Resolution and format mix; corrupted files; label quality; near-duplicate images across train and test |
| **Joined tables** | Row counts before and after each join (a join that multiplies rows is a bug); unmatched keys |

## Common mistakes

- Skipping EDA and finding the problems only after training, when the model looks strangely good or bad.
- "Cleaning as you go" inside the EDA notebook.
- Trusting the mean and standard deviation of dirty data.
- Throwing away everything an outlier rule flags.
- Checking only overall summaries and missing problems that live in one group or one period.
- Forming feature ideas from the test period.
- Not writing anything down, then having to redo the whole investigation in the Prepare stage.
