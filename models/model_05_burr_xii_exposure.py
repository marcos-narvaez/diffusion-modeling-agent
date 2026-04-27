"""
Model 5: Burr XII + Contemporaneous Exposure
  - Heterogeneity, duration dependence, mean-scaled exposure covariate
  - S(t) = (alpha / (alpha + B(t)))^r
  - B(t) = Σ [u^c - (u-1)^c] * exp(β_exp * exposure_scaled_u)
"""
from _shared import *

adoptions, exposure, exposure_all, T = load_data()
exposure_mean = exposure.mean()
exposure_scaled = exposure / exposure_mean

def neg_ll(params, adoptions):
    log_r, log_alpha, log_c, beta_exp = params
    r = np.exp(log_r)
    alpha = np.exp(log_alpha)
    c = np.exp(log_c)
    N = N_TOTAL

    B = np.zeros(T + 1)
    for u in range(1, T + 1):
        tw = u ** c - (u - 1) ** c
        B[u] = B[u-1] + tw * np.exp(beta_exp * exposure_scaled[u-1])

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

best_params, ll = fit_model(neg_ll, 4, adoptions, ['r', 'alpha', 'c', 'beta_exp'])
r, alpha, c = np.exp(best_params[:3])
beta_exp = best_params[3]

B = np.zeros(T + 1)
for u in range(1, T + 1):
    B[u] = B[u-1] + (u**c - (u-1)**c) * np.exp(beta_exp * exposure_scaled[u-1])
S = (alpha / (alpha + B)) ** r
predictions = N_TOTAL * (S[:-1] - S[1:])

print_results("Burr XII + Exposure", ['r', 'alpha', 'c', 'beta_exp'],
              [r, alpha, c, beta_exp], ll, 4, adoptions, predictions)
