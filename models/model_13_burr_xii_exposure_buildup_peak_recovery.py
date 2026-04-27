"""
Model 13: Burr XII + Exposure + Buildup/Peak/Recovery Season
  - Three-part Season specification with exponential decay functions
  - Buildup: exp(-0.5 * weeks_until_peak) for 5 weeks before
  - Peak: binary for Season week (week 31)
  - Recovery: exp(-0.5 * weeks_after_peak) for 5 weeks after
  - Note: this richer seasonal decomposition is included as an example of
    a flexible spec; on well-behaved data the buildup/recovery coefficients
    can become unidentified or violate the recovery <= peak ordering, in
    which case the simpler indicator (model 9) is preferred.
"""
from _shared import *

adoptions, exposure, exposure_all, T = load_data()
exposure_mean = exposure.mean()
exposure_scaled = exposure / exposure_mean

# Season covariates
peak_week = 31  # 1-indexed
buildup = np.zeros(T)
peak = np.zeros(T)
recovery = np.zeros(T)

for w in range(1, T + 1):
    weeks_until = peak_week - w
    weeks_after = w - peak_week
    if 0 < weeks_until <= 5:
        buildup[w-1] = np.exp(-0.5 * weeks_until)
    elif weeks_until == 0:
        peak[w-1] = 1.0
    elif 0 < weeks_after <= 5:
        recovery[w-1] = np.exp(-0.5 * weeks_after)

def neg_ll(params, adoptions):
    log_r, log_alpha, log_c, beta_exp, beta_buildup, beta_peak, beta_recovery = params
    r, alpha, c = np.exp(log_r), np.exp(log_alpha), np.exp(log_c)
    N = N_TOTAL

    B = np.zeros(T + 1)
    for u in range(1, T + 1):
        tw = u ** c - (u - 1) ** c
        cov = (beta_exp * exposure_scaled[u-1] + beta_buildup * buildup[u-1] +
               beta_peak * peak[u-1] + beta_recovery * recovery[u-1])
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

best_params, ll = fit_model(neg_ll, 7, adoptions,
                            ['r', 'alpha', 'c', 'beta_exp', 'beta_buildup', 'beta_peak', 'beta_recovery'])
r, alpha, c = np.exp(best_params[:3])
beta_exp = best_params[3]
beta_buildup, beta_peak, beta_recovery = best_params[4], best_params[5], best_params[6]

B = np.zeros(T + 1)
for u in range(1, T + 1):
    tw = u ** c - (u - 1) ** c
    cov = (beta_exp * exposure_scaled[u-1] + beta_buildup * buildup[u-1] +
           beta_peak * peak[u-1] + beta_recovery * recovery[u-1])
    B[u] = B[u-1] + tw * np.exp(cov)
S = (alpha / (alpha + B)) ** r
predictions = N_TOTAL * (S[:-1] - S[1:])

print_results("Burr XII + Exposure + Buildup/Peak/Recovery (REJECTED)",
              ['r', 'alpha', 'c', 'beta_exp', 'beta_buildup', 'beta_peak', 'beta_recovery'],
              [r, alpha, c, beta_exp, beta_buildup, beta_peak, beta_recovery], ll, 7, adoptions, predictions)

# Diagnostic flags
issues = []
if beta_buildup < 0:
    issues.append(f"ISSUE: Buildup coefficient is NEGATIVE ({beta_buildup:.3f})")
if beta_recovery > beta_peak:
    issues.append(f"ISSUE: Recovery ({beta_recovery:.3f}) EXCEEDS peak ({beta_peak:.3f})")
for issue in issues:
    print(f"   {issue}")
