"""
Model 15: Hold-Out Validation (same spec as Model 9, trained on weeks 1-35)
  - Same Burr XII + Exposure + Season specification, but fit only on first 35 weeks
"""
from _shared import *

adoptions, exposure, exposure_all, T = load_data()

# Training split: weeks 1-35
T_train = 35
adoptions_train = adoptions[:T_train]
exposure_train = exposure[:T_train]
exposure_mean_train = exposure_train.mean()
exposure_scaled_train = exposure_train / exposure_mean_train
season_train = np.zeros(T_train)
season_train[29:33] = 1.0  # weeks 30-33 within training window

# Full exposure for prediction (use training mean for scaling)
exposure_scaled_full = exposure / exposure_mean_train
season_full = np.zeros(T)
season_full[29:33] = 1.0

def neg_ll(params, sales_t):
    log_r, log_alpha, log_c, beta_exp, beta_season = params
    r, alpha, c = np.exp(log_r), np.exp(log_alpha), np.exp(log_c)
    N = N_TOTAL
    T_t = len(sales_t)

    B = np.zeros(T_t + 1)
    for u in range(1, T_t + 1):
        tw = u ** c - (u - 1) ** c
        B[u] = B[u-1] + tw * np.exp(beta_exp * exposure_scaled_train[u-1] + beta_season * season_train[u-1])

    S = (alpha / (alpha + B)) ** r
    f = S[:-1] - S[1:]

    ll = 0.0
    for t in range(T_t):
        if f[t] <= 0: return 1e15
        ll += sales_t[t] * np.log(f[t])
    survivors = N - sales_t.sum()
    if S[T_t] <= 0: return 1e15
    ll += survivors * np.log(S[T_t])
    return -ll

# Fit on training data
best_params, ll = fit_model(neg_ll, 5, adoptions_train,
                            ['r', 'alpha', 'c', 'beta_exp', 'beta_season'])
r, alpha, c = np.exp(best_params[:3])
beta_exp, beta_season = best_params[3], best_params[4]

# Predict for ALL 41 weeks using training parameters
B_full = np.zeros(T + 1)
for u in range(1, T + 1):
    tw = u ** c - (u - 1) ** c
    B_full[u] = B_full[u-1] + tw * np.exp(beta_exp * exposure_scaled_full[u-1] + beta_season * season_full[u-1])
S_full = (alpha / (alpha + B_full)) ** r
predictions_full = N_TOTAL * (S_full[:-1] - S_full[1:])

pred_train = predictions_full[:T_train]
pred_test = predictions_full[T_train:]
adoptions_test = adoptions[T_train:]

train_mae = np.mean(np.abs(adoptions_train - pred_train))
train_mape = np.mean(np.abs((adoptions_train - pred_train) / adoptions_train)) * 100
test_mae = np.mean(np.abs(adoptions_test - pred_test))
test_mape = np.mean(np.abs((adoptions_test - pred_test) / adoptions_test)) * 100

print(f"\n{'='*60}")
print(f"  Hold-Out Validation (Train: 1-35, Test: 36-41)")
print(f"{'='*60}")
print(f"  Parameters (trained on weeks 1-35):")
print(f"    r     = {r:.6f}")
print(f"    alpha = {alpha:.6f}")
print(f"    c     = {c:.6f}")
print(f"    β_exp    = {beta_exp:.6f}")
print(f"    β_season = {beta_season:.6f}")
print(f"\n  Training LL: {ll:,.2f}")
print(f"\n  Training performance (weeks 1-35):")
print(f"    MAE:  {train_mae:,.0f}")
print(f"    MAPE: {train_mape:.1f}%")
print(f"\n  Test performance (weeks 36-41):")
print(f"    MAE:  {test_mae:,.0f}")
print(f"    MAPE: {test_mape:.1f}%")
print(f"\n  Week-by-week test comparison:")
for i, w in enumerate(range(T_train, T)):
    err_pct = abs(adoptions_test[i] - pred_test[i]) / adoptions_test[i] * 100
    print(f"    Week {w+1}: Actual={adoptions_test[i]:>6,}, Predicted={pred_test[i]:>8,.0f}, Error={err_pct:.1f}%")

is_degenerate = r > 1e6
if is_degenerate:
    print(f"\n   WARNING: r={r:.2e} — heterogeneity parameters degenerate (near-homogeneous)")
    print(f"   Without late-period data, model cannot identify population heterogeneity")
print(f"{'='*60}\n")
