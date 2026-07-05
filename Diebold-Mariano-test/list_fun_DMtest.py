import numpy as np
import statsmodels.api as sm
from scipy.stats import norm


def diebold_mariano(y_true, f_model, f_benchmark, loss, h=1, two_sided=False):
    """
    Parameters
    ----------
    y_true : array-like
        Realized values r_t(h).
    f_model : array-like
        Forecasts from the candidate model being evaluated.
    f_benchmark : array-like
        Forecasts from the benchmark model.
    loss : {"mse", "mae"}
        Loss function used to form the loss differential (required).
          - "mse": squared-error loss  (g(e) = e^2)
          - "mae": absolute-error loss  (g(e) = |e|)
    h : int
        Forecast horizon. The HAC lag is set to max(h - 1, 0).
    two_sided : bool
        - False (default): one-sided test of whether the model beats the
          benchmark.  H0: E[d_t] <= 0 (model no better);
          H1: E[d_t] > 0 (model has lower loss than the benchmark).
        - True: two-sided test of whether the two forecasts differ.
          H0: E[d_t] = 0;  H1: E[d_t] != 0.
        The statistic dm_stat is identical either way; only the p-value differs.

    Returns
    -------
    dm_stat : float
        DM test statistic (~ N(0, 1) under H0). Positive => model beats benchmark.
    p_value : float
        One-sided p-value (H1: model better) if two_sided is False,
        otherwise the two-sided p-value.
    dbar : float
        Mean loss differential, mean(d_t).
    """

    y_true = np.asarray(y_true).ravel()
    f_model = np.asarray(f_model).ravel()
    f_benchmark = np.asarray(f_benchmark).ravel()

    if not (len(y_true) == len(f_model) == len(f_benchmark)):
        raise ValueError("Inputs must have the same length.")

    # Forecast errors
    e_model = y_true - f_model
    e_benchmark = y_true - f_benchmark

    # Loss differential (benchmark minus model): positive => model is better
    if loss == "mse":
        d = e_benchmark**2 - e_model**2
    else:  # "mae"
        d = np.abs(e_benchmark) - np.abs(e_model)

    # HAC lag length is fixed to max(h - 1, 0).
    max_lag = max(h - 1, 0)

    # Mean test: regress d on a constant with HAC (Newey-West) covariance.
    X = np.ones(len(d))
    results = sm.OLS(d, X).fit(cov_type="HAC", cov_kwds={"maxlags": max_lag})
    dm_stat = float(results.tvalues[0])

    # p-value under the asymptotic N(0, 1) distribution
    if two_sided:                       # H1: E[d] != 0 (forecasts differ)
        p_value = float(2 * (1 - norm.cdf(abs(dm_stat))))
    else:                               # H1: E[d] > 0 (model beats benchmark)
        p_value = float(1 - norm.cdf(dm_stat))

    return dm_stat, p_value, float(d.mean())
