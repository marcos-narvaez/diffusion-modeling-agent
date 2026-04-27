"""
Model 6: Burr XII + log(Exposure)
  - Same as Model 5 but with log-transformed exposure instead of mean-scaled
"""
from _shared import *

adoptions, exposure, exposure_all, T = load_data()
log_exposure = np.log(exposure)

def neg_ll(params, adoptions):
    log_r, log_alpha, log_c, beta_logexp = params
    r = np.exp(log_r)
    alpha = np.exp(log_alpha)
    c = np.exp(log_c)
    N = N_TOTAL

    B = np.zeros(T + 1)
    for u in range(1, T + 1):
        tw = u ** c - (u - 1) ** c
        B[u] = B[u-1] + tw * np.exp(beta_logexp * log_exposure[u-1])

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

best_params, ll = fit_model(neg_ll, 4, adoptions, ['r', 'alpha', 'c', 'beta_logexp'])
r, alpha, c = np.exp(best_params[:3])
beta_logexp = best_params[3]

B = np.zeros(T + 1)
for u in range(1, T + 1):
    B[u] = B[u-1] + (u**c - (u-1)**c) * np.exp(beta_logexp * log_exposure[u-1])
S = (alpha / (alpha + B)) ** r
predictions = N_TOTAL * (S[:-1] - S[1:])

print_results("Burr XII + log(Exposure)", ['r', 'alpha', 'c', 'beta_logexp'],
              [r, alpha, c, beta_logexp], ll, 4, adoptions, predictions)
