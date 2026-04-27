"""
Model 16: 2-Segment Latent-Class Burr XII + Exposure + Season
  - Two segments with separate (r, α, c) but pooled covariate coefficients (β_air, β_hol)
  - Mixing weight π determines segment proportions
  - 9 parameters: r1, α1, c1, r2, α2, c2, β_air, β_hol, π
"""
from _shared import *

adoptions, exposure, exposure_all, T = load_data()
exposure_mean = exposure.mean()
exposure_scaled = exposure / exposure_mean
season = np.zeros(T)
season[29:33] = 1.0

def neg_ll(params, adoptions):
    (log_r1, log_alpha1, log_c1,
     log_r2, log_alpha2, log_c2,
     beta_exp, beta_season, logit_pi) = params

    r1, alpha1, c1 = np.exp(log_r1), np.exp(log_alpha1), np.exp(log_c1)
    r2, alpha2, c2 = np.exp(log_r2), np.exp(log_alpha2), np.exp(log_c2)
    pi1 = 1.0 / (1.0 + np.exp(-logit_pi))  # sigmoid for mixing weight
    pi2 = 1.0 - pi1
    N = N_TOTAL

    # Segment 1: B(t) with c1
    B1 = np.zeros(T + 1)
    for u in range(1, T + 1):
        tw = u**c1 - (u-1)**c1
        B1[u] = B1[u-1] + tw * np.exp(beta_exp * exposure_scaled[u-1] + beta_season * season[u-1])

    # Segment 2: B(t) with c2
    B2 = np.zeros(T + 1)
    for u in range(1, T + 1):
        tw = u**c2 - (u-1)**c2
        B2[u] = B2[u-1] + tw * np.exp(beta_exp * exposure_scaled[u-1] + beta_season * season[u-1])

    S1 = (alpha1 / (alpha1 + B1))**r1
    S2 = (alpha2 / (alpha2 + B2))**r2

    # Mixture survival
    S_mix = pi1 * S1 + pi2 * S2
    f_mix = S_mix[:-1] - S_mix[1:]

    ll = 0.0
    for t in range(T):
        if f_mix[t] <= 0:
            return 1e15
        ll += adoptions[t] * np.log(f_mix[t])
    survivors = N - adoptions.sum()
    if S_mix[T] <= 0:
        return 1e15
    ll += survivors * np.log(S_mix[T])
    return -ll

# Fit with more starting points (rougher landscape)
best_ll_val = np.inf
best_params = None
rng = np.random.RandomState(42)

for i in range(24):
    x0 = rng.uniform(-2, 2, 9)
    try:
        result = minimize(neg_ll, x0, args=(adoptions,),
                          method='Nelder-Mead',
                          options={'maxiter': 30000, 'xatol': 1e-8, 'fatol': 1e-8})
        if result.fun < best_ll_val and np.isfinite(result.fun):
            best_ll_val = result.fun
            best_params = result.x
    except:
        continue

ll = -best_ll_val
r1, alpha1, c1 = np.exp(best_params[:3])
r2, alpha2, c2 = np.exp(best_params[3:6])
beta_exp, beta_season = best_params[6], best_params[7]
pi1 = 1.0 / (1.0 + np.exp(-best_params[8]))
pi2 = 1.0 - pi1

# Predictions
B1 = np.zeros(T + 1)
B2 = np.zeros(T + 1)
for u in range(1, T + 1):
    cov = np.exp(beta_exp * exposure_scaled[u-1] + beta_season * season[u-1])
    B1[u] = B1[u-1] + (u**c1 - (u-1)**c1) * cov
    B2[u] = B2[u-1] + (u**c2 - (u-1)**c2) * cov

S1 = (alpha1 / (alpha1 + B1))**r1
S2 = (alpha2 / (alpha2 + B2))**r2
S_mix = pi1 * S1 + pi2 * S2
predictions = N_TOTAL * (S_mix[:-1] - S_mix[1:])

k = 9
bic = -2 * ll + k * np.log(T)
mae = np.mean(np.abs(adoptions - predictions))
mape = np.mean(np.abs((adoptions - predictions) / adoptions)) * 100

print(f"\n{'='*60}")
print(f"  2-Segment Latent-Class Burr XII + Exposure + Season")
print(f"{'='*60}")
print(f"  Segment 1 (weight = {pi1:.3f}):")
print(f"    r1 = {r1:.6f}, alpha1 = {alpha1:.4f}, c1 = {c1:.4f}")
print(f"  Segment 2 (weight = {pi2:.3f}):")
print(f"    r2 = {r2:.6f}, alpha2 = {alpha2:.4f}, c2 = {c2:.4f}")
print(f"  Pooled covariates:")
print(f"    beta_exp = {beta_exp:.6f}")
print(f"    beta_season = {beta_season:.6f}")
print(f"\n  Log-likelihood: {ll:,.2f}")
print(f"  BIC (k={k}):     {bic:,.2f}")
print(f"  MAE:            {mae:,.0f}")
print(f"  MAPE:           {mape:.1f}%")
print(f"  Total predicted: {predictions.sum():,.0f}")
print(f"  Total actual:    {adoptions.sum():,}")

# Compare to single-segment final model
ll_single = -2006771.67
bic_single = 4013561.91
print(f"\n  --- Comparison to single-segment model ---")
print(f"  Single-segment LL: {ll_single:,.2f}, BIC: {bic_single:,.2f}")
print(f"  2-segment LL:      {ll:,.2f}, BIC: {bic:,.2f}")
print(f"  LL improvement:    {ll - ll_single:,.2f}")
print(f"  BIC improvement:   {bic_single - bic:,.2f} (positive = 2-segment better)")
lrt = 2 * (ll - ll_single)
print(f"  LRT statistic:     {lrt:,.2f} (df=4)")
from scipy.stats import chi2
p_val = 1 - chi2.cdf(lrt, 4)
print(f"  LRT p-value:       {p_val:.6f}")
print(f"{'='*60}")
