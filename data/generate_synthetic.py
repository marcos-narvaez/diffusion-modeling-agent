"""
Generate synthetic adoption data for testing the agent end to end.

The data-generating process is a Burr XII timing-to-adoption model with
heterogeneity Gamma(r, alpha) on individual hazards, a Weibull-style duration
dependence parameter c, and two time-varying covariates entering on the hazard
through a proportional-hazards link:

    h(t | X_t) = h_0(t) * exp(beta_exposure * exposure_t + beta_season * season_t)

Aggregate weekly adoptions are simulated by computing the mixture survival
function and drawing weekly counts from the implied probability mass with
small multiplicative noise.

The output file is data/adoption_synthetic.csv with columns:
    week:      1..T
    adoptions: integer; "?" for the last 4 weeks (the withheld horizon)
    exposure:  float; a time-varying external covariate
    season:    0/1 indicator for a known seasonal window

The market size N is recorded in data/adoption_synthetic.meta.json so the
agent can read it without it being baked into model code.
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
OUT_CSV = HERE / "adoption_synthetic.csv"
OUT_META = HERE / "adoption_synthetic.meta.json"


def main(seed: int = 20260427) -> None:
    rng = np.random.default_rng(seed)

    # ── Ground-truth parameters ──
    T_total = 45
    T_observed = 41
    N = 2_000_000

    # Burr XII heterogeneity + duration dependence
    r_true = 0.06
    alpha_true = 190.0
    c_true = 1.34

    # Covariate effects (proportional hazards on the Burr XII A(t))
    beta_exposure = 1.48
    beta_season = 0.42

    # ── Time-varying covariates ──
    # Exposure follows a smooth ramp-up with mid-life peak and wind-down.
    weeks = np.arange(1, T_total + 1)
    exposure_raw = (
        2_500
        + 9_500 * np.exp(-0.5 * ((weeks - 26) / 6.5) ** 2)
        + 1_400 * np.exp(-0.5 * ((weeks - 12) / 4.5) ** 2)
    )
    # Inject mild week-to-week noise so the series is not perfectly smooth.
    exposure = exposure_raw * rng.normal(1.0, 0.05, T_total)
    exposure = np.clip(exposure, 200, None)

    # Seasonal window: weeks 30-33 (zero-indexed 29..32). This is a generic
    # seasonal indicator — there is no calendar attached to the data.
    season = np.zeros(T_total)
    season[29:33] = 1.0

    # ── Compute B(t) and survival under the data-generating process ──
    exposure_scaled = exposure / exposure.mean()
    B = np.zeros(T_total + 1)
    for u in range(1, T_total + 1):
        time_weight = u ** c_true - (u - 1) ** c_true
        cov = beta_exposure * exposure_scaled[u - 1] + beta_season * season[u - 1]
        B[u] = B[u - 1] + time_weight * np.exp(cov)
    S = (alpha_true / (alpha_true + B)) ** r_true
    f = S[:-1] - S[1:]

    # Expected weekly adoptions
    expected = N * f

    # Add a small amount of noise (sales counts are noisy in real life).
    adoptions = np.maximum(0, np.round(expected * rng.normal(1.0, 0.06, T_total))).astype(int)

    # ── Assemble dataframe and withhold last 4 weeks ──
    df = pd.DataFrame({
        "week": weeks,
        "adoptions": adoptions.astype(object),
        "exposure": np.round(exposure, 1),
        "season": season.astype(int),
    })
    df.loc[T_observed:, "adoptions"] = "?"

    df.to_csv(OUT_CSV, index=False)

    meta = {
        "market_size_N": N,
        "T_total": T_total,
        "T_observed": T_observed,
        "withheld_weeks": list(range(T_observed + 1, T_total + 1)),
        "ground_truth": {
            "r": r_true,
            "alpha": alpha_true,
            "c": c_true,
            "beta_exposure": beta_exposure,
            "beta_season": beta_season,
            "season_indices_one_based": [30, 31, 32, 33],
        },
        "seed": seed,
        "notes": (
            "Synthetic data drawn from a Burr XII + proportional-hazards process. "
            "The agent does not see these ground-truth parameters; they are recorded "
            "here only for downstream diagnostics."
        ),
    }
    OUT_META.write_text(json.dumps(meta, indent=2))

    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_META}")
    print(f"Total observed adoptions: {sum(int(a) for a in adoptions[:T_observed]):,}")
    print(f"Cumulative through week {T_observed}: "
          f"{sum(int(a) for a in adoptions[:T_observed]) / N * 100:.2f}% of N")


if __name__ == "__main__":
    main()
