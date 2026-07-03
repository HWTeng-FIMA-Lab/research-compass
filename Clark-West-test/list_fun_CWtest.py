import numpy as np
import matplotlib.pyplot as plt
# Import the required libraries
import statsmodels.api as sm
import numpy as np

import numpy as np
import statsmodels.api as sm
from scipy.stats import norm

def diebold_mariano_mspe(y_true, f_with, f_without, h=1, max_lag=None):
    """
    Diebold–Mariano (DM) test with MSPE loss.
 
    Convention:
        d_t = e_without^2 - e_with^2

    Positive mean(d_t) => model_with has lower MSPE.

    Parameters
    ----------
    y_true : array-like
        Realized values r_t(h)
    f_with : array-like
        Forecasts from model_with (feature selection)
    f_without : array-like
        Forecasts from model_without (no feature selection)
    h : int
        Forecast horizon
    max_lag : int or None
        HAC lag length. If None, uses h-1 (recommended for overlapping returns).

    Returns
    -------
    dm_stat : float
        DM test statistic
    p_value_onesided : float
        One-sided p-value (H1: model_with improves MSPE)
    dbar : float
        Mean loss differential
    """

    y_true = np.asarray(y_true).ravel()
    f_with = np.asarray(f_with).ravel()
    f_without = np.asarray(f_without).ravel()

    if not (len(y_true) == len(f_with) == len(f_without)):
        raise ValueError("Inputs must have the same length.")

    # Forecast errors
    e_with = y_true - f_with
    e_without = y_true - f_without

    # Loss differential (MSPE)
    d = e_without**2 - e_with**2

    # HAC lag selection
    if max_lag is None:
        max_lag = max(h - 1, 0)

    # Mean test via regression on constant with HAC covariance
    X = np.ones(len(d))
    model = sm.OLS(d, X)
    results = model.fit(cov_type='HAC', cov_kwds={'maxlags': max_lag})

    dm_stat = float(results.tvalues[0])

    # Right-sided test: H1: E[d] > 0 (model_with better)
    p_value = float(1 - norm.cdf(dm_stat))

    return dm_stat, p_value, float(d.mean())

    
# By chatGPT and checkeh by HWTeng 
import numpy as np
import statsmodels.api as sm
from scipy.stats import norm

# -----------------------------
# Clark–West (2007) test (nested)
# H1: "large" model improves MSPE over "small" model
# -----------------------------
def clark_west_nested(y_true, f_with, f_without, h=1, max_lag=None):
    """
    Clark–West adjusted MSPE test for nested models.
    Tests whether the larger model ("large") improves forecast accuracy vs the smaller model ("small").
    H1: E[d_cw] > 0.

    Parameters
    ----------
    y_true : array-like
        realized y
    f_with : array-like
        forecasts from smaller (nested) model
    f_without : array-like
        forecasts from larger model
    h : int
        forecast horizon (use h-1 HAC lag for overlapping h-step targets)
    max_lag : int or None
        HAC lag. If None, uses h-1.

    Returns
    -------
    cw_stat : float
    p_value_onesided : float
    dbar : float
    """
    y_true = np.asarray(y_true)
    f_with = np.asarray(f_with)
    f_without = np.asarray(f_without)

    e_with = y_true - f_with
    e_without = y_true - f_without

    # CW adjusted differential (positive mean => large improves over small)
    d_cw = (e_without**2) - (e_with**2) + (f_with - f_without)**2

    if max_lag is None:
        max_lag = max(h - 1, 0)

    # Mean test with HAC (Newey–West) SE: regress d_cw on a constant
    X = np.ones(len(d_cw))
    res = sm.OLS(d_cw, X).fit(cov_type="HAC", cov_kwds={"maxlags": max_lag})
    cw_stat = float(res.tvalues[0])

    # Right-sided p-value for H1: mean(d_cw) > 0
    p_value = float(1 - norm.cdf(cw_stat))
    return cw_stat, p_value, float(d_cw.mean())


# -----------------------------
# Synthetic DGP: stationary ARX with Uniform predictors
# x_{i,t} ~ Uniform(0,1), i=1..p, t=1..T
# y_t = phi*y_{t-1} + b1*x1_t + b2*x2_t (+ optionally b3*x3_t) + eps_t
# -----------------------------
def simulate_arx_uniform(T=500, p=10, phi=0.4, b1=1.2, b2=-0.9, b3=0.0, sigma=0.6, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 1.0, size=(T, p))
    eps = rng.normal(0.0, sigma, size=T)

    y = np.zeros(T)
    y[0] = eps[0]
    for t in range(1, T):
        y[t] = phi * y[t-1] + b1 * X[t, 0] + b2 * X[t, 1] + b3 * X[t, 2] + eps[t]
    return y, X


# -----------------------------
# Rolling one-step-ahead OLS forecasts
# model_with    = const + y_{t-1} + x1 + x2
# model_without = const + y_{t-1} + x1..xp
# -----------------------------
def rolling_forecasts(y, X, window=200):
    T, p = X.shape
    f_with, f_without, y_realized = [], [], []

    for t in range(window + 1, T):
        idx0, idx1 = t - window, t  # training y: idx0..t-1
        y_train = y[idx0:idx1]
        y_lag_train = y[idx0 - 1:idx1 - 1]  # y_{s-1} aligned with y_s

        # SMALL ("with"): const + y_{t-1} + x1 + x2
        Z_small = np.column_stack([y_lag_train, X[idx0:idx1, 0], X[idx0:idx1, 1]])
        Z_small = sm.add_constant(Z_small)
        beta_small = sm.OLS(y_train, Z_small).fit()
        z_small_t = np.array([1.0, y[t - 1], X[t, 0], X[t, 1]])
        f_with.append(float(z_small_t @ beta_small.params))

        # LARGE ("without"): const + y_{t-1} + x1..xp
        Z_large = np.column_stack([y_lag_train, X[idx0:idx1, :]])
        Z_large = sm.add_constant(Z_large)
        beta_large = sm.OLS(y_train, Z_large).fit()
        z_large_t = np.concatenate([[1.0, y[t - 1]], X[t, :]])
        f_without.append(float(z_large_t @ beta_large.params))

        y_realized.append(float(y[t]))

    return np.array(y_realized), np.array(f_with), np.array(f_without)


def plot_forecasts_3x3_panels(y_true, f_with, f_without, kde_grid=400):
    """
    One figure with 3x2 panels:
      (1) (1,1): y_true
      (2) (2,1): f_without
      (3) (3,1): f_with
      (4) (1,2): pe_without = y_true - f_without
      (5) (2,2): pe_with    = y_true - f_with
      (6) (3,2): KDEs comparing pe_without vs pe_with
      (7) (1,3): spe_without = y_true - f_without
      (8) (2,3): spe_with    = y_true - f_with
      (9) (3,3): KDEs comparing spe_without vs spe_with


    Fix: Do NOT share x-axes across columns; explicitly set KDE x-limits to error range.
    KDE computed via a simple Gaussian kernel with Silverman's bandwidth.
    """

    y_true = np.asarray(y_true).ravel()
    f_with = np.asarray(f_with).ravel()
    f_without = np.asarray(f_without).ravel()
    

    if not (len(y_true) == len(f_with) == len(f_without)):
        raise ValueError("y_true, f_with, and f_without must have the same length.")

    t = np.arange(len(y_true))
    pe_without = y_true - f_without
    pe_with = y_true - f_with
    
    spe_without = pe_without ** 2
    spe_with = pe_with ** 2

    # ----- Simple Gaussian KDE (no external deps) -----
    def _kde_1d(x, grid):
        x = np.asarray(x).ravel()
        n = len(x)
        if n < 2:
            return np.zeros_like(grid)

        std = np.std(x, ddof=1)
        if std == 0:
            out = np.zeros_like(grid)
            out[np.argmin(np.abs(grid - x[0]))] = 1.0
            return out

        h = 1.06 * std * (n ** (-1 / 5))  # Silverman ROT
        h = max(h, 1e-8)

        z = (grid[:, None] - x[None, :]) / h
        dens = np.exp(-0.5 * z**2).mean(axis=1) / (h * np.sqrt(2 * np.pi))
        return dens

    # KDE grid over prediction-error range
    lo = float(min(pe_without.min(), pe_with.min()))
    hi = float(max(pe_without.max(), pe_with.max()))
    pad = 0.10 * (hi - lo) if hi > lo else 1.0
    grid = np.linspace(lo - pad, hi + pad, kde_grid)

    dens_without = _kde_1d(pe_without, grid)
    dens_with = _kde_1d(pe_with, grid)

    
    # KDE grid over squared prediction-error range
    lo_spe = float(min(spe_without.min(), spe_with.min()))
    hi_spe = float(max(spe_without.max(), spe_with.max()))
    pad_spe = 0.10 * (hi - lo) if hi > lo else 1.0
    grid_spe = np.linspace(lo - pad, hi + pad, kde_grid)

    dens_spe_without = _kde_1d(spe_without, grid)
    dens_spe_with = _kde_1d(spe_with, grid)

    
    
    # ----- 3x2 layout (no shared x-axis) -----
    fig, axes = plt.subplots(3, 3, figsize=(7.2, 6))

    # (1) y_true
    ax = axes[0, 0]
    ax.plot(t, y_true)
    ax.set_title("y_true")
    ax.set_ylabel("Value")

    # (2) f_without
    ax = axes[1, 0]
    ax.plot(t, f_without)
    ax.set_title("f_without")
    ax.set_ylabel("Forecast")

    # (3) f_with
    ax = axes[2, 0]
    ax.plot(t, f_with)
    ax.set_title("f_with")
    ax.set_xlabel("Time")
    ax.set_ylabel("Forecast")

    # (4) pe_without
    ax = axes[0, 1]
    ax.plot(t, pe_without)
    ax.set_title("pe_without")
    ax.set_ylabel("Error")

    # (5) pe_with
    ax = axes[1, 1]
    ax.plot(t, pe_with)
    ax.set_title("pe_with")
    ax.set_ylabel("Error")

    # (6) KDE comparison
    ax = axes[2, 1]
    ax.plot(grid, dens_without, label="KDE(pe_without)")
    ax.plot(grid, dens_with, label="KDE(pe_with)")
    ax.set_title("KDE")
    ax.set_xlabel("Prediction error")
    ax.set_ylabel("Density")
    ax.set_xlim(lo - pad, hi + pad)
    
    
    # (7) spe_without
    ax = axes[0, 2]
    ax.plot(t, spe_without)
    ax.set_title("spe_without")
    ax.set_ylabel("Squared Error")

    # (8) spe_with
    ax = axes[1, 2]
    ax.plot(t, spe_with)
    ax.set_title("spe_with")
    ax.set_ylabel("Squared Error")

    # (9) KDE comparison
    ax = axes[2, 2]
    ax.plot(grid_spe, dens_spe_without, label="KDE(spe_without)")
    ax.plot(grid_spe, dens_spe_with, label="KDE(spe_with)")
    ax.set_title("KDE")
    ax.set_xlabel("Squared prediction error")
    ax.set_ylabel("Density")
    ax.set_xlim(lo_spe - pad_spe, hi_spe + pad_spe)
    
    
    #ax.legend()

    fig.tight_layout()
    plt.show()

    return pe_with, pe_without