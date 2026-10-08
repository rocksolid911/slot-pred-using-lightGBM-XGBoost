# Glossary

One line per term, plus the lesson where it is explained properly. Keep the list alphabetical.

| Term | Meaning | Lesson |
|---|---|---|
| Baseline | The simple rule a model must beat to be worth deploying (here: a fixed 15-min slot) | [A1](A01-framing.md) |
| Conditional quantile | A quantile that depends on the inputs: the P80 *for this doctor, visit type, …* | [A1](A01-framing.md) |
| Coverage | Share of visits finishing within the predicted P80; should be ≈ 0.80. Gameable on its own | [A1](A01-framing.md) |
| Feature | An input to the model; it must be known at prediction (booking) time | [A1](A01-framing.md) |
| Leakage | Using information that won't exist at prediction time; looks great in tests, fails in production | [A1](A01-framing.md) |
| MAE | Mean absolute error: average number of minutes off | [A1](A01-framing.md) |
| P50 / median | Value that half the outcomes fall within: the typical visit | [A1](A01-framing.md) |
| P80 | Value that 80% of outcomes fall within: the safe slot when overrun is 4× worse than idle | [A1](A01-framing.md) |
| Pinball loss | Quantile loss: q per minute under, (1−q) per minute over; minimised by the q-th quantile | [A1](A01-framing.md) |
| Quantile | Value that a fraction q of outcomes fall at or below | [A1](A01-framing.md) |
| Right-skewed | Mostly small values with a long tail of big ones; pulls the mean above the median | [A1](A01-framing.md) |
| Target | What the model predicts: `consult_end − consult_start` in minutes | [A1](A01-framing.md) |
