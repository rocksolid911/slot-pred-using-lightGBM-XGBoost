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

## Track B
