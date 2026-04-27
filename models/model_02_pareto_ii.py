"""
Model 2: Pareto II (Exponential-Gamma mixture)
  - Heterogeneity via Gamma(r, alpha) mixing over exponential hazard
  - No duration dependence (c=1 implicitly), no covariates
  - S(t) = (alpha / (alpha + t))^r
"""
from _shared import *

adoptions, exposure, exposure_all, T = load_data()

def neg_ll(params, adoptions):
    log_r, log_alpha = params
    r = np.exp(log_r)
    alpha = np.exp(log_alpha)
    N = N_TOTAL

    S = np.array([(alpha / (alpha + t)) ** r for t in range(T + 1)])
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

best_params, ll = fit_model(neg_ll, 2, adoptions, ['r', 'alpha'])
r = np.exp(best_params[0])
alpha = np.exp(best_params[1])

S = np.array([(alpha / (alpha + t)) ** r for t in range(T + 1)])
predictions = N_TOTAL * (S[:-1] - S[1:])

print_results("Pareto II (Exponential-Gamma)", ['r', 'alpha'], [r, alpha], ll, 2, adoptions, predictions)
