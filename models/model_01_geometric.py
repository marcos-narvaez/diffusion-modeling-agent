"""
Model 1: Geometric Baseline
  - No heterogeneity, no duration dependence, no covariates
  - Single parameter: p (constant hazard probability)
  - S(t) = (1-p)^t
"""
from _shared import *

adoptions, exposure, exposure_all, T = load_data()

def neg_ll(params, adoptions):
    log_p = params[0]
    p = 1.0 / (1.0 + np.exp(-log_p))  # sigmoid to keep in (0,1)
    N = N_TOTAL
    ll = 0.0
    for t in range(1, T + 1):
        S_prev = (1 - p) ** (t - 1)
        S_curr = (1 - p) ** t
        f_t = S_prev - S_curr
        if f_t <= 0:
            return 1e15
        ll += adoptions[t-1] * np.log(f_t)
    survivors = N - adoptions.sum()
    S_T = (1 - p) ** T
    if S_T <= 0:
        return 1e15
    ll += survivors * np.log(S_T)
    return -ll

best_params, ll = fit_model(neg_ll, 1, adoptions, ['p'])
p = 1.0 / (1.0 + np.exp(-best_params[0]))

# Compute predictions
S = np.array([(1 - p) ** t for t in range(T + 1)])
predictions = N_TOTAL * (S[:-1] - S[1:])

print_results("Geometric Baseline", ['p'], [p], ll, 1, adoptions, predictions)
