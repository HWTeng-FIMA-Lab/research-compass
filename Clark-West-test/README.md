# Clark-West test

Created by Huei-Wen Teng on 20260210 when revising Yenting and YC's work on powershap in predicting BTC returns

## 1. EDA and simulation studies

test_Clark-West-test.ipynb

We generate a y that depends on $x_1$ and $x_2$. 

Then, we predict y by using $x_1,\ldots,x_p$ (without feature selection) adn by using $x_1$ and $x_2$ (with feature selection).

We show that both the Clark-West test and Diebold-Mariano test rejcts the null hypothesis and shows that using only the true predictors helps prediction. 

## 2. List of functions

In the file "list_fun_CWtest.py" we have the following five functions

|Index|Outputs| Function(Inputs)| Descriptions|
|---|---|---|---|
|1| dm_stat, p_value, float(d.mean()) | diebold_mariano_mspe(y_true, f_with, f_without, h=1, max_lag=None)|Implement the Diebold-Mariano test for general two models|
|2| cw_stat, p_value, float(d_cw.mean()) | clark_west_nested(y_true, f_with, f_without, h=1, max_lag=None)|Implement the Clark-West test for nested models|
|3| y, X | simulate_arx_uniform(T=500, p=10, phi=0.4, b1=1.2, b2=-0.9, b3=0.0, sigma=0.6, seed=42)|Generate artificial time series y depends on x1, x2, x3|
|4| np.array(y_realized), np.array(f_with), np.array(f_without) | rolling_forecasts(y, X, window=200)| Do a rolling window forecast, using x1,...,xp (without feature selection) and only x_1 and x_2 (with feature selection)|
|5| pe_with, pe_without | plot_forecasts_3x3_panels(y_true, f_with, f_without, kde_grid=400)| Show the time series plots, density plots for comparisons|

