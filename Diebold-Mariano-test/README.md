# 1. Introduction of Diebold-Mariano test
*Edited by Elvis*

This document explains the theoretical basis and implementation of the Diebold–Mariano (1995) test used to compare the predictive accuracy of two competing forecasts. Comparing point estimates of a loss measure (MSE, MAE, and so on) alone is not enough to determine which model is superior; the DM test assesses whether the observed difference in predictive performance is **statistically significant**. Depending on the hypothesis, the test can evaluate either whether the two models simply **differ** in accuracy (two-sided) or whether one model is **significantly better** than the other (one-sided).

---

## 1. Test Statistic and Asymptotic Distribution

**Step 1 — Forecast errors.**
Let $y_t$ be the realized target and $\hat{f}_{1t}$, $\hat{f}_{2t}$ the forecasts from the two models:
$$e_{1t} = y_t - \hat{f}_{1t}, \qquad e_{2t} = y_t - \hat{f}_{2t}$$

**Step 2 — Loss differential and its sample mean.**
For a loss function $g(\cdot)$, the loss differential and its sample mean over $T$ out-of-sample points are:
$$d_t = g(e_{1t}) - g(e_{2t}), \qquad \bar{d} = \frac{1}{T}\sum_{t=1}^{T} d_t$$

The loss function is arbitrary; for example:
- **MSE:** $g(e) = e^2$  ⇒  $d_t = e_{1t}^2 - e_{2t}^2$
- **MAE:** $g(e) = |e|$  ⇒  $d_t = |e_{1t}| - |e_{2t}|$

**Step 3 — Asymptotic distribution.**
If the loss-differential series $d_t$ is covariance stationary and short memory, the Central Limit Theorem gives:
$$\sqrt{T}(\bar{d}-\mu) \xrightarrow{d} N\big(0,\, 2\pi f_d(0)\big)$$

where $f_d(0)$ is the spectral density of $d_t$ at frequency zero, and $2\pi f_d(0)$ equals the long-run variance of $d_t$ (the sum of all its autocovariances). The test statistic is:
$$S_1 = \frac{\bar{d}}{\sqrt{2\pi \hat{f}_d(0)/T}} \xrightarrow{a} N(0,1)$$

where $\hat{f}_d(0)$ is a **consistent estimate of the spectral density $f_d(0)$ at frequency zero**, obtained in practice from a HAC (Newey–West type) estimate of the long-run variance. Regardless of the distribution of the forecast errors themselves, $S_1$ converges to the standard normal $N(0,1)$ in large samples.

---

## 2. Estimating the Long-Run Variance

The denominator of $S_1$ requires an estimate of the long-run variance $2\pi f_d(0)$. Because $d_t$ is serially correlated, this estimate must incorporate the autocovariances of $d_t$ at each lag, not merely the lag-0 variance. In this implementation the estimate is obtained by regressing $d_t$ on a constant by OLS — so the estimated intercept equals $\bar{d}$ — and taking its HAC standard error, which equals $\sqrt{\hat{V}(\bar{d})}$; the ratio of the two is $S_1$.

- **Estimator:** a Newey–West HAC estimator with a Bartlett kernel, which guarantees a non-negative long-run variance estimate.
- **Truncation lag $= h - 1$:** because the loss differential of an optimal $h$-step-ahead forecast is at most $(h-1)$-dependent (Diebold & Mariano, 1995). When $h = 1$ the truncation lag is $0$, only the lag-$0$ variance enters, and the estimator reduces to the ordinary sample variance.

---

## 3. One-Sided vs. Two-Sided Tests

Since $S_1 \sim N(0,1)$, either a two-sided or a one-sided test may be used, exactly as with a standard Z-test.

**Two-sided** — used when the question is whether the two models **differ** in accuracy, with no prior on which is better:
- $H_0: \mathbb{E}[d_t] = 0$  vs.  $H_1: \mathbb{E}[d_t] \neq 0$
- At the 5% level, reject $H_0$ if $|S_1| > 1.96$.

**One-sided** — used when testing whether a given model is **significantly better** than the other. With the convention $d_t = g(e_{\text{benchmark}}) - g(e_{\text{proposed}})$:
- $H_0: \mathbb{E}[d_t] \leq 0$  vs.  $H_1: \mathbb{E}[d_t] > 0$  (proposed loss $<$ benchmark loss)
- At the 5% level, reject $H_0$ if $S_1 > 1.645$.

---

## 4. Loss Function

The DM test allows the per-period loss to be an **arbitrary function $g(\cdot)$** of the forecast error. Once the loss differential $d_t = g(e_{1t}) - g(e_{2t})$ has been formed, the only requirement is that $d_t$ be covariance stationary and short memory; standard Central Limit Theorem results then deliver the $N(0,1)$ asymptotic distribution, whatever the choice of $g$.

Intuitively, as long as the underlying forecast errors are stationary, any well-behaved transformation of them — squaring for MSE, absolute value for MAE, or an asymmetric loss — still produces a stationary loss-differential series $d_t$. Consequently the same statistic and the same asymptotic normal distribution apply regardless of the loss function; only the realized values of $d_t$ (and hence the resulting p-value) change.

