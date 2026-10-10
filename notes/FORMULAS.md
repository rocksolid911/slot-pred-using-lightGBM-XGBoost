# Formula sheet

Every formula from the project on one page. Each entry gives the formula, what it is for, and the lesson that derives it.

## Track A

**Quantile** ([A1](A01-framing.md)): the value that a fraction $q$ of outcomes fall within.
$$Q(q) = \min \{ v : P(y \le v) \ge q \}$$

**Best slot from costs** ([A1](A01-framing.md)): book this quantile.
$$q^* = \frac{c_{over}}{c_{over} + c_{idle}} \qquad (4:1 \to 0.8,\; 9:1 \to 0.9)$$

**Pinball loss** ([A1](A01-framing.md)): the loss a quantile model minimises.
$$L_q(y,\hat y) = \max\big(q(y-\hat y),\,(q-1)(y-\hat y)\big)$$

**Loss → statistic** ([A1](A01-framing.md)): squared error → mean; absolute error → median; pinball $q$ → $q$-th quantile.

**MAE** ([A1](A01-framing.md))
$$\text{MAE} = \frac{1}{n}\sum_i \lvert y_i - \hat y_i \rvert$$

**Coverage** ([A1](A01-framing.md)): should be ≈ 0.80 for a P80 model.
$$\text{coverage} = \frac{1}{n}\sum_i \mathbb{1}[\,y_i \le \hat y^{(0.8)}_i\,]$$

**Simulator rule** ([A2](A02-simulator.md)): extra work is added, pace multiplies, drift is added last.
$$\text{expected} = (\text{base} + 1.3\min(r,8) + 4\cdot\text{interp}) \times \text{pace} + 3\cdot[\text{Derm, from July}]$$

**Log-normal noise** ([A2](A02-simulator.md))
$$Z\sim\mathcal N(\mu,\sigma^2),\quad \text{noise}=e^Z,\quad E[e^Z]=e^{\mu+\sigma^2/2}\;\Rightarrow\;\mu=-\sigma^2/2 \text{ for mean } 1$$

**Quantiles of log-normal noise** ([A2](A02-simulator.md)): percentiles carry through $e^Z$; the mean doesn't.
$$\text{median}=e^{\mu},\qquad Q_q=e^{\mu+z_q\sigma},\qquad \frac{P80}{\text{median}}=e^{0.8416\,\sigma}\;(=1.27 \text{ for } \sigma=0.28)$$

**z-score** ([A2](A02-simulator.md))
$$z=\frac{x-\text{mean}}{\text{SD}},\qquad z_{0.5}=0,\; z_{0.8}=0.8416,\; z_{0.9}=1.2816,\; z_{0.95}=1.645$$

**Oracle MAE (irreducible error) for log-normal noise** ([A2](A02-simulator.md)): $\Phi$ is the normal CDF.
$$\text{MAE}_{oracle}=\text{expected}\times\big(2\Phi(\sigma)-1\big)\;=\;0.22\times\text{expected}\;(\sigma=0.28)$$

**Estimating σ from data** ([A2](A02-simulator.md))
$$\hat\sigma=\operatorname{SD}\big(\log y_i-\log\hat y_i\big)$$

**IQR fences** ([A3](A03-eda.md)): on skewed data apply them to $\log(x)$, then convert back with $e^x$.
$$\text{IQR}=Q_3-Q_1,\qquad \text{flag outside }\big[Q_1-1.5\,\text{IQR},\;Q_3+1.5\,\text{IQR}\big]$$

**When dropping rows biases a model** ([A3](A03-eda.md)): MCAR, never; MAR (missingness depends on observed columns), the overall mix shifts but $P(y\mid x)$ stays right; MNAR (depends on $y$ itself), biased.

**Why split by time** ([A4](A04-prepare.md)): the test score should estimate future error; a random split estimates it on a mix of months, which differs from the future whenever the data drifts.
$$\text{target}=\mathbb E_{(x,y)\sim P_{\text{future}}}\big[L(y,\hat y)\big],\qquad P_{\text{mix}}=\textstyle\sum_m w_m P_m\;\neq\;P_{\text{future}}\text{ under drift}$$

**Fill + flag, fitted on train** ([A4](A04-prepare.md))
$$\text{age}^{\text{filled}}_i=\begin{cases}\text{age}_i & \text{known}\ \operatorname{median}(\text{age}_{\text{train}}) & \text{missing}\end{cases},\qquad \text{age\_missing}_i=\mathbb 1[\text{age}_i\text{ missing}]$$

**Pinball loss as the score** ([A5](A05-baselines.md)): late minutes cost $q$, idle minutes cost $1-q$; averaged over the $n$ scored visits. Lower is better.
$$\text{score}=\frac1n\sum_{i=1}^{n}\Big(q\cdot\max(y_i-p_i,0)+(1-q)\cdot\max(p_i-y_i,0)\Big)$$

**How shaky a box's q-point is** ([A5](A05-baselines.md)): the jumpiness shrinks with the square root of the visits in the box (4× the visits → half the jump).
$$\text{SD}(\hat Q_q)\;\propto\;\frac{1}{\sqrt n}$$

## Track B
