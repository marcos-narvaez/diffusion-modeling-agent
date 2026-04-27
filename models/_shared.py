"""
Shared data loading and MLE utilities for the model files in this directory.

Each model_NN_*.py file imports from here. The contract is:

    from _shared import *
    adoptions, exposure, exposure_all, T = load_data()

`adoptions` is the observed integer series (length T), `exposure` is the
contemporaneous covariate over the observed window, `exposure_all` is the
covariate over the full horizon (including withheld weeks), and `T` is the
length of the observed window.

The market-size constant N is read from the dataset metadata file rather
than hard-coded into the model files.
"""
import json
import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize


HERE = os.path.dirname(__file__)
DATA_PATH = os.path.normpath(os.path.join(HERE, "..", "data", "adoption_synthetic.csv"))
META_PATH = os.path.normpath(os.path.join(HERE, "..", "data", "adoption_synthetic.meta.json"))


def _load_meta():
    with open(META_PATH) as f:
        return json.load(f)


_meta = _load_meta()
N_TOTAL = int(_meta["market_size_N"])


def load_data():
    df = pd.read_csv(DATA_PATH)
    observed = df[df["adoptions"] != "?"].copy()
    observed["adoptions"] = observed["adoptions"].astype(int)
    adoptions = observed["adoptions"].values
    exposure = observed["exposure"].values.astype(float)
    exposure_all = df["exposure"].values.astype(float)
    T = len(adoptions)
    return adoptions, exposure, exposure_all, T


def fit_model(neg_ll_func, n_params, adoptions, param_names,
              n_starts=None, bounds=None, extra_args=()):
    """Fit a model via Nelder-Mead with multiple random starting points.

    n_starts defaults to 48 for production runs. Set the environment
    variable AGENT_FIT_NSTARTS to override (e.g., AGENT_FIT_NSTARTS=8 for
    fast test runs).
    """
    if n_starts is None:
        n_starts = int(os.environ.get("AGENT_FIT_NSTARTS", "48"))
    maxiter = int(os.environ.get("AGENT_FIT_MAXITER", "50000"))
    best_ll = np.inf
    best_params = None
    rng = np.random.RandomState(42)
    for _ in range(n_starts):
        x0 = rng.uniform(-2, 2, n_params)
        try:
            result = minimize(
                neg_ll_func, x0,
                args=(adoptions,) + extra_args,
                method="Nelder-Mead",
                options={"maxiter": maxiter, "xatol": 1e-10, "fatol": 1e-10},
            )
            if result.fun < best_ll and np.isfinite(result.fun):
                best_ll = result.fun
                best_params = result.x
        except Exception:
            continue
    return best_params, -best_ll


def print_results(model_name, param_names, param_values, ll, k, adoptions, predictions):
    """Print a standardized model summary."""
    bic = -2 * ll + k * np.log(len(adoptions))
    mae = np.mean(np.abs(adoptions - predictions))
    nz = adoptions != 0
    mape = np.mean(np.abs((adoptions[nz] - predictions[nz]) / adoptions[nz])) * 100 if nz.any() else float("nan")

    print(f"\n{'=' * 60}")
    print(f"  {model_name}")
    print(f"{'=' * 60}")
    print(f"  Parameters (k={k}):")
    for name, val in zip(param_names, param_values):
        print(f"    {name:20s} = {val:.6f}")
    print(f"\n  Log-likelihood: {ll:,.2f}")
    print(f"  BIC:            {bic:,.2f}")
    print(f"  MAE:            {mae:,.0f}")
    print(f"  MAPE:           {mape:.1f}%")
    print(f"  Total predicted: {predictions.sum():,.0f}")
    print(f"  Total actual:    {adoptions.sum():,}")
    print(f"{'=' * 60}\n")
