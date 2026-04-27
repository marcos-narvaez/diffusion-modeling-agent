"""
Model 7: Burr XII + Lagged Exposure (1-week lag)
  - Same as Model 5 but using previous week's exposure as the covariate
"""
from _shared import *

adoptions, exposure, exposure_all, T = load_data()
exposure_mean = exposure.mean()
# Lagged exposure: week t uses week t-1 exposure; week 1 uses week 1 (no lag available)
lagged_exposure = np.concatenate([[exposure[0]], exposure[:-1]])
lagged_scaled = lagged_exposure / exposure_mean

def neg_ll(params, adoptions):
    log_r, log_alpha, log_c, beta_lag = params
    r = np.exp(log_r)
    alpha = np.exp(log_alpha)
    c = np.exp(log_c)
    N = N_TOTAL

    B = np.zeros(T + 1)
    for u in range(1, T + 1):
        tw = u ** c - (u - 1) ** c
        B[u] = B[u-1] + tw * np.exp(beta_lag * lagged_scaled[u-1])

    S = (alpha / (alpha + B)) ** r
    f = S[:-1] - S[1:]

    ll = 0.0
    for t in range(T):
        if f[t] <= 0:
            return 1e15
        ll += adoptions[t] * np.log(f[t])
    survivors = N - adoptions.sum()
    if S[T] <= 0:
        return 1e15
    ll += survivors * np.log(S[T])
    return -ll

best_params, ll = fit_model(neg_ll, 4, adoptions, ['r', 'alpha', 'c', 'beta_lagexp'])
r, alpha, c = np.exp(best_params[:3])
beta_lag = best_params[3]

B = np.zeros(T + 1)
for u in range(1, T + 1):
    B[u] = B[u-1] + (u**c - (u-1)**c) * np.exp(beta_lag * lagged_scaled[u-1])
S = (alpha / (alpha + B)) ** r
predictions = N_TOTAL * (S[:-1] - S[1:])

print_results("Burr XII + Lagged Exposure", ['r', 'alpha', 'c', 'beta_lagexp'],
              [r, alpha, c, beta_lag], ll, 4, adoptions, predictions)
