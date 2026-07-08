

Study period: 20250601 to 20260630

Daily option prices and prices

Chris: BTC options 

Jimmy: SP500 index



## Estimate Q, we need the following three functions to do a simulation study 

### Focus on vanilla options

1. gen_SVCJ(???) (2)

to generate synthetic dataset for a bunch of option prices of size $n$

<img width="320" height="298" alt="image" src="https://github.com/user-attachments/assets/6e94ee63-162e-4236-920b-5de3ddf9107e" />


|Parameter | True parameter | n =100 | n = 1000| n=10000|
|---|---|---|---|---|
| $V_0$ |0.0009  |? | ?| ?|   


2a. calc_SVCJ_option_Duffie(omega, K, T) (6)
2b. calc_SVCJ_option_MC(omega, K, T) (2)

3. est_SVCJ(???) (6) to minimize the squared loss among market and model prices...


Our empirical methodology proceeds as follows. For each trading day, we estimate
a separate set of model parameters using all intraday inverse BTC option trades across
moneyness levels and maturities. Because the simulation time step in our framework
is one day, options with zero days to maturity are excluded from estimation. In
our full dataset, such contracts account for only 4.79% (35,258 contracts) of all
observations. Given their negligible proportion, their exclusion does not materially
affect the estimation results. In practice, all analyses in this study are conducted in
MATLAB (R2021a) with a Monte Carlo sample size of N = 20,000. For the estimation
of the BS model, we employ MATLAB’s fminsearch solver. For the SV, SVJ, and
SVCJ models, we use the lsqnonlin solver with the Levenberg–Marquardt algorithm,
setting the FunctionTolerance to 10−4

