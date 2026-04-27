"""
Model 10: Burr XII + Lagged Exposure + Season Dummy
  - 1-week lagged exposure + single Season week indicator
"""
from _shared import *

adoptions, exposure, exposure_all, T = load_data()
exposure_mean = exposure.mean()
lagged_exposure = np.concatenate([[exposure[0]], exposure[:-1]])
lagged_scaled = lagged_exposure / exposure_mean
xmas_dummy = np.zeros(T)
xmas_dummy[30] = 1.0

def neg_ll(params, adoptions):
    log_r, log_alpha, log_c, beta_lag, beta_season = params
    r, alpha, c = np.exp(log_r), np.exp(log_alpha), np.exp(log_c)
    N = N_TOTAL

    B = np.zeros(T + 1)
    for u in range(1, T + 1):
        tw = u ** c - (u - 1) ** c
        B[u] = B[u-1] + tw * np.exp(beta_lag * lagged_scaled[u-1] + beta_season * xmas_dummy[u-1])

    S = (alpha / (alpha + B)) ** r
    f = S[:-1] - S[1:]

    ll = 0.0
    for t in range(T):
        if f[t] <= 0: return 1e15
        ll += adoptions[t] * np.log(f[t])
    survivors = N - adoptions.sum()
    if S[T] <= 0: return 1e15
    ll += survivors * np.log(S[T])
    return -ll

best_params, ll = fit_model(neg_ll, 5, adoptions, ['r', 'alpha', 'c', 'beta_lagexp', 'beta_season'])
r, alpha, c = np.exp(best_params[:3])
beta_lag, beta_season = best_params[3], best_params[4]

B = np.zeros(T + 1)
for u in range(1, T + 1):
    tw = u ** c - (u - 1) ** c
    B[u] = B[u-1] + tw * np.exp(beta_lag * lagged_scaled[u-1] + beta_season * xmas_dummy[u-1])
S = (alpha / (alpha + B)) ** r
predictions = N_TOTAL * (S[:-1] - S[1:])

print_results("Burr XII + Lagged Exposure + Season",
              ['r', 'alpha', 'c', 'beta_lagexp', 'beta_season'],
              [r, alpha, c, beta_lag, beta_season], ll, 5, adoptions, predictions)
