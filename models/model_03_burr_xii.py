"""
Model 3: Burr XII Baseline (Weibull-Gamma mixture)
  - Heterogeneity via Gamma(r, alpha), duration dependence via c
  - No covariates
  - S(t) = (alpha / (alpha + t^c))^r
"""
from _shared import *

adoptions, exposure, exposure_all, T = load_data()

def neg_ll(params, adoptions):
    log_r, log_alpha, log_c = params
    r = np.exp(log_r)
    alpha = np.exp(log_alpha)
    c = np.exp(log_c)
    N = N_TOTAL

    S = np.array([(alpha / (alpha + t ** c)) ** r for t in range(T + 1)])
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

best_params, ll = fit_model(neg_ll, 3, adoptions, ['r', 'alpha', 'c'])
r = np.exp(best_params[0])
alpha = np.exp(best_params[1])
c = np.exp(best_params[2])

S = np.array([(alpha / (alpha + t ** c)) ** r for t in range(T + 1)])
predictions = N_TOTAL * (S[:-1] - S[1:])

print_results("Burr XII Baseline", ['r', 'alpha', 'c'], [r, alpha, c], ll, 3, adoptions, predictions)
