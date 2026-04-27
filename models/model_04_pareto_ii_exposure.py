"""
Model 4: Pareto II + Exposure (Exponential-Gamma with proportional hazards)
  - Heterogeneity via Gamma(r, alpha), NO duration dependence (c=1)
  - Contemporaneous exposure covariate (mean-scaled)
  - S(t) = (alpha / (alpha + B(t)))^r where B(t) = Σ exp(β_air * x_u)
"""
from _shared import *

adoptions, exposure, exposure_all, T = load_data()
exposure_mean = exposure.mean()
exposure_scaled = exposure / exposure_mean

def neg_ll(params, adoptions):
    log_r, log_alpha, beta_exp = params
    r = np.exp(log_r)
    alpha = np.exp(log_alpha)
    N = N_TOTAL

    # B(t) with c=1: increments are just 1 per week
    B = np.zeros(T + 1)
    for u in range(1, T + 1):
        B[u] = B[u-1] + np.exp(beta_exp * exposure_scaled[u-1])

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

best_params, ll = fit_model(neg_ll, 3, adoptions, ['r', 'alpha', 'beta_exp'])
r = np.exp(best_params[0])
alpha = np.exp(best_params[1])
beta_exp = best_params[2]

B = np.zeros(T + 1)
for u in range(1, T + 1):
    B[u] = B[u-1] + np.exp(beta_exp * exposure_scaled[u-1])
S = (alpha / (alpha + B)) ** r
predictions = N_TOTAL * (S[:-1] - S[1:])

print_results("Pareto II + Exposure", ['r', 'alpha', 'beta_exp'], [r, alpha, beta_exp], ll, 3, adoptions, predictions)
