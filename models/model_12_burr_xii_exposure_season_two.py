"""
Model 12: Burr XII + Exposure + Separate Season & New Year Dummies
  - Two separate season indicators instead of a single window
"""
from _shared import *

adoptions, exposure, exposure_all, T = load_data()
exposure_mean = exposure.mean()
exposure_scaled = exposure / exposure_mean
season = np.zeros(T)
season[29:31] = 1.0  # weeks 30-31 (Season)
ny = np.zeros(T)
ny[31:33] = 1.0    # weeks 32-33 (New Year)

def neg_ll(params, adoptions):
    log_r, log_alpha, log_c, beta_exp, beta_season, beta_season2 = params
    r, alpha, c = np.exp(log_r), np.exp(log_alpha), np.exp(log_c)
    N = N_TOTAL

    B = np.zeros(T + 1)
    for u in range(1, T + 1):
        tw = u ** c - (u - 1) ** c
        cov = beta_exp * exposure_scaled[u-1] + beta_season * season[u-1] + beta_season2 * ny[u-1]
        B[u] = B[u-1] + tw * np.exp(cov)

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

best_params, ll = fit_model(neg_ll, 6, adoptions,
                            ['r', 'alpha', 'c', 'beta_exp', 'beta_season', 'beta_season2'])
r, alpha, c = np.exp(best_params[:3])
beta_exp, beta_season, beta_season2 = best_params[3], best_params[4], best_params[5]

B = np.zeros(T + 1)
for u in range(1, T + 1):
    tw = u ** c - (u - 1) ** c
    B[u] = B[u-1] + tw * np.exp(beta_exp * exposure_scaled[u-1] + beta_season * season[u-1] + beta_season2 * ny[u-1])
S = (alpha / (alpha + B)) ** r
predictions = N_TOTAL * (S[:-1] - S[1:])

print_results("Burr XII + Exposure + Season/NY Separate",
              ['r', 'alpha', 'c', 'beta_exp', 'beta_season', 'beta_season2'],
              [r, alpha, c, beta_exp, beta_season, beta_season2], ll, 6, adoptions, predictions)
