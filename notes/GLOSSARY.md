# Glossary

One line per term, plus the lesson where it is explained properly. Keep the list alphabetical.

| Term | Meaning | Lesson |
|---|---|---|
| Baseline | The simple rule a model must beat to be worth deploying (here: a fixed 15-min slot) | [A1](A01-framing.md) |
| Cold start | Predicting for something with no history (a new doctor); the model can't have learned its pattern | [A2](A02-simulator.md) |
| Conditional quantile | A quantile that depends on the inputs: the P80 *for this doctor, visit type, …* | [A1](A01-framing.md) |
| Additive effect | A factor that adds fixed minutes whatever the base (interpreter +4) | [A2](A02-simulator.md) |
| Coverage | Share of visits finishing within the predicted P80; should be ≈ 0.80. Gameable on its own | [A1](A01-framing.md) |
| Drift | The world changes after training (new doctor, new protocol), so the model's accuracy drops | [A2](A02-simulator.md) |
| Feature | An input to the model; it must be known at prediction (booking) time | [A1](A01-framing.md) |
| Irreducible error | Error even a perfect model has, because of the noise; the oracle's score | [A2](A02-simulator.md) |
| Leakage | Using information that won't exist at prediction time; looks great in tests, fails in production | [A1](A01-framing.md) |
| Log-normal | e^Z where Z is normal: always positive, right-skewed | [A2](A02-simulator.md) |
| MAE | Mean absolute error: average number of minutes off | [A1](A01-framing.md) |
| Multiplicative effect | A factor that scales the whole time (slow doctor ×1.2) | [A2](A02-simulator.md) |
| Noise | The random part of the outcome that no feature can explain | [A2](A02-simulator.md) |
| Oracle | A model that knows the hidden rule exactly but not the noise; the best possible score | [A2](A02-simulator.md) |
| P50 / median | Value that half the outcomes fall within: the typical visit | [A1](A01-framing.md) |
| P80 | Value that 80% of outcomes fall within: the safe slot when overrun is 4× worse than idle | [A1](A01-framing.md) |
| Pinball loss | Quantile loss: q per minute under, (1−q) per minute over; minimised by the q-th quantile | [A1](A01-framing.md) |
| Quantile | Value that a fraction q of outcomes fall at or below | [A1](A01-framing.md) |
| Right-skewed | Mostly small values with a long tail of big ones; pulls the mean above the median | [A1](A01-framing.md) |
| Sigma (σ) of the noise | How much same-kind visits vary; 0.28 ≈ ±25–30%. Measured from data as the SD of log(actual) − log(predicted) | [A2](A02-simulator.md) |
| Simulator | Code that generates data from rules we chose, so the truth is known | [A2](A02-simulator.md) |
| Target | What the model predicts: `consult_end − consult_start` in minutes | [A1](A01-framing.md) |
| z-score | (value − mean) / SD; each z marks a fixed share of a normal distribution (0.8416 → 80%) | [A2](A02-simulator.md) |
