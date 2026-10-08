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

**Estimating σ from data** ([A2](A02-simulator.md))
$$\hat\sigma=\operatorname{SD}\big(\log y_i-\log\hat y_i\big)$$

## Track B
