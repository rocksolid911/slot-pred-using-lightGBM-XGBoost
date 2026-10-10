# Glossary

One line per term, plus the lesson where it is explained properly. Keep the list alphabetical.

| Term | Meaning | Lesson |
|---|---|---|
| Additive effect | A factor that adds fixed minutes whatever the base (interpreter +4) | [A2](A02-simulator.md) |
| Backoff | If a box has too few training visits (or none, like a new doctor), use the bigger box it belongs to instead | [A5](A05-baselines.md) |
| Baseline | The simple rule a model must beat to be worth deploying (here: a fixed 15-min slot) | [A1](A01-framing.md) |
| Booked slot | The predicted P-point rounded **up** to the 5-minute grid; what the clinic actually books (16.6 → 20) | [A5](A05-baselines.md) |
| Box (group) | One combination of columns, e.g. (Dermatology, follow-up); a lookup-table baseline books one q-point per box | [A5](A05-baselines.md) |
| Cold start | Predicting for something with no history (a new doctor); the model can't have learned its pattern | [A2](A02-simulator.md) |
| Conditional quantile | A quantile that depends on the inputs: the P80 *for this doctor, visit type, …* | [A1](A01-framing.md) |
| Coverage | Share of visits finishing within the predicted P80; should be ≈ 0.80. Gameable on its own | [A1](A01-framing.md) |
| Domain rule | A limit set by knowledge of the real world ("a consult can't take under 1 minute or over 2 hours"): finds the impossible, not just the unusual | [A3](A03-eda.md) |
| Drift | The world changes after training (new doctor, new protocol), so the model's accuracy drops | [A2](A02-simulator.md) |
| EDA | Exploratory data analysis: understanding the data, and logging its problems, before changing anything | [A3](A03-eda.md) |
| Feature | An input to the model; it must be known at prediction (booking) time | [A1](A01-framing.md) |
| Fit / transform | Learn numbers from the training rows once (fit), then apply them unchanged to every split and to live requests (transform) | [A4](A04-prepare.md) |
| Funnel | A table of how many rows each cleaning step removed, each row counted once under its first reason | [A4](A04-prepare.md) |
| Hidden notebook state | A variable or column left over from a cell that was edited or deleted; results change after a restart | [A3](A03-eda.md) |
| Imputation | Filling in a missing value (here: the training-row median age); a learned step | [A4](A04-prepare.md) |
| IQR fences | Q1 − 1.5·IQR and Q3 + 1.5·IQR; values outside are "unusual". Use log(x) for skewed data | [A3](A03-eda.md) |
| Irreducible error | Error even a perfect model has, because of the noise; the oracle's score | [A2](A02-simulator.md) |
| Leaderboard | A table of every method's scores on the same split (here June), so they can be compared fairly | [A5](A05-baselines.md) |
| Leakage | Using information that won't exist at prediction time; looks great in tests, fails in production | [A1](A01-framing.md) |
| Learned step | A preparation step whose output depends on statistics of many rows (a median, a percentile, a category list); fit it on the training rows only | [A4](A04-prepare.md) |
| Log-normal | e^Z where Z is normal: always positive, right-skewed | [A2](A02-simulator.md) |
| MAE | Mean absolute error: average number of minutes off | [A1](A01-framing.md) |
| MAR | Missing at random *given observed columns* (e.g. one clinic, two months); dropping keeps relationships given those columns intact | [A3](A03-eda.md) |
| MCAR | Missing completely at random; dropping only loses data | [A3](A03-eda.md) |
| Missing-indicator flag | A 0/1 column (`age_missing`) recording that a value was filled in, so the model can tell "really 42" from "unknown" | [A4](A04-prepare.md) |
| MNAR | Missing because of the value itself (long visits never closed); dropping biases the model | [A3](A03-eda.md) |
| Multiplicative effect | A factor that scales the whole time (slow doctor ×1.2) | [A2](A02-simulator.md) |
| Noise | The random part of the outcome that no feature can explain | [A2](A02-simulator.md) |
| Oracle (floor) | The best possible score, computed from the simulator's hidden expected minutes; only for scoring, never a feature | [A5](A05-baselines.md) |
| Oracle | A model that knows the hidden rule exactly but not the noise; the best possible score | [A2](A02-simulator.md) |
| P50 / median | Value that half the outcomes fall within: the typical visit | [A1](A01-framing.md) |
| P80 | Value that 80% of outcomes fall within: the safe slot when overrun is 4× worse than idle | [A1](A01-framing.md) |
| Pinball loss | Quantile loss: q per minute under, (1−q) per minute over; minimised by the q-th quantile | [A1](A01-framing.md) |
| Placeholder value | A fake number meaning "unknown" (age 999); must become missing before modelling | [A3](A03-eda.md) |
| Problem log | The EDA output: each data problem with evidence, rows affected, cause and a fix to try | [A3](A03-eda.md) |
| Process change | A change in how data is recorded (Whitefield start = check-in); looks like valid data, shows up as a step in a time plot | [A3](A03-eda.md) |
| Quantile | Value that a fraction q of outcomes fall at or below | [A1](A01-framing.md) |
| Random split | Shuffling rows into train and test regardless of date; on time-ordered data with drift it leaks the future and looks too good | [A4](A04-prepare.md) |
| Right-skewed | Mostly small values with a long tail of big ones; pulls the mean above the median | [A1](A01-framing.md) |
| Robust statistics | Median, quantiles, IQR: statistics a few bad rows can't drag around | [A3](A03-eda.md) |
| Rule (preparation) | A step whose output for a row depends only on that row plus fixed constants; safe before the split and at prediction time | [A4](A04-prepare.md) |
| Sigma (σ) of the noise | How much same-kind visits vary; 0.28 ≈ ±25–30%. Measured from data as the SD of log(actual) − log(predicted) | [A2](A02-simulator.md) |
| Simulator | Code that generates data from rules we chose, so the truth is known | [A2](A02-simulator.md) |
| Split by time | Train on the earliest months, validate on the next, test on the latest; mirrors deployment, where the model always predicts the future | [A4](A04-prepare.md) |
| Target | What the model predicts: `consult_end − consult_start` in minutes | [A1](A01-framing.md) |
| Training–serving skew | Inputs prepared differently at training time and at prediction time, so the deployed model sees data it never learned from | [A4](A04-prepare.md) |
| Unseen category | A level that appears after training (Nikhil Verma); it becomes missing in the categorical column instead of crashing | [A4](A04-prepare.md) |
| z-score | (value − mean) / SD; each z marks a fixed share of a normal distribution (0.8416 → 80%) | [A2](A02-simulator.md) |
