"""
Model 9: Burr XII + Exposure + Season Window (weeks 30-33) (reference final model)
  - Binary indicator for season window (weeks 30-33)
"""
from _shared import *

adoptions, exposure, exposure_all, T = load_data()
exposure_mean = exposure.mean()
exposure_scaled = exposure / exposure_mean
season = np.zeros(T)
season[29:33] = 1.0  # weeks 30-33 (0-indexed: 29-32)

def neg_ll(params, adoptions):
    log_r, log_alpha, log_c, beta_exp, beta_season = params
    r = np.exp(log_r)
    alpha = np.exp(log_alpha)
    c = np.exp(log_c)
    N = N_TOTAL

    B = np.zeros(T + 1)
    for u in range(1, T + 1):
        tw = u ** c - (u - 1) ** c
        B[u] = B[u-1] + tw * np.exp(beta_exp * exposure_scaled[u-1] + beta_season * season[u-1])

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

best_params, ll = fit_model(neg_ll, 5, adoptions, ['r', 'alpha', 'c', 'beta_exp', 'beta_season'])
r, alpha, c = np.exp(best_params[:3])
beta_exp, beta_season = best_params[3], best_params[4]

B = np.zeros(T + 1)
for u in range(1, T + 1):
    tw = u ** c - (u - 1) ** c
    B[u] = B[u-1] + tw * np.exp(beta_exp * exposure_scaled[u-1] + beta_season * season[u-1])
S = (alpha / (alpha + B)) ** r
predictions = N_TOTAL * (S[:-1] - S[1:])

print_results("Final: Burr XII + Exposure + Season ",
              ['r', 'alpha', 'c', 'beta_exp', 'beta_season'],
              [r, alpha, c, beta_exp, beta_season], ll, 5, adoptions, predictions)

# Also print forecasts for weeks 42-45
print("Forecasts for Withheld Weeks:")
exposure_all_scaled = exposure_all / exposure_mean
season_all = np.zeros(len(exposure_all))
season_all[29:33] = 1.0

B_all = np.zeros(len(exposure_all) + 1)
for u in range(1, len(exposure_all) + 1):
    tw = u ** c - (u - 1) ** c
    idx = u - 1
    cov = beta_exp * exposure_all_scaled[idx]
    if idx < len(season_all):
        cov += beta_season * season_all[idx]
    B_all[u] = B_all[u-1] + tw * np.exp(cov)

S_all = (alpha / (alpha + B_all)) ** r
pred_all = N_TOTAL * (S_all[:-1] - S_all[1:])

for w in range(41, 45):
    print(f"  Week {w+1}: {pred_all[w]:,.0f} (exposure: {exposure_all[w]:,.0f} )")
print(f"  Total withheld: {pred_all[41:45].sum():,.0f}")
