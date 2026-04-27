"""
Sanity tests on the model files.

These tests do two things:
  1. Confirm the synthetic data generator runs and produces a usable file.
  2. For each model file, import-and-execute it as a script. Each model
     prints its own results; if it raises, the test fails. The reference
     final model (model_09) is checked more strictly: its fitted parameters
     must land within a tolerance of the data-generating ground truth.

Models 6 (log-exposure) and 11 (multicollinear pair) and 13 (recovery >
peak) are *expected* to fit poorly or degenerate. We only assert they run
without raising.
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"


@pytest.fixture(scope="session", autouse=True)
def _ensure_data_present():
    csv = DATA_DIR / "adoption_synthetic.csv"
    meta = DATA_DIR / "adoption_synthetic.meta.json"
    if not csv.exists() or not meta.exists():
        result = subprocess.run(
            [sys.executable, str(DATA_DIR / "generate_synthetic.py")],
            capture_output=True, text=True, check=True,
        )
        assert csv.exists() and meta.exists(), result.stderr


def _run_model(script: Path) -> str:
    """Run a model script and return its stdout. Raise on non-zero exit."""
    env = os.environ.copy()
    env.setdefault("AGENT_FIT_NSTARTS", "8")
    env.setdefault("AGENT_FIT_MAXITER", "8000")
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True, text=True, cwd=MODELS_DIR, env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"{script.name} exited {result.returncode}\nSTDERR:\n{result.stderr}"
        )
    return result.stdout


def test_data_files_exist():
    assert (DATA_DIR / "adoption_synthetic.csv").exists()
    assert (DATA_DIR / "adoption_synthetic.meta.json").exists()


def test_data_meta_contract():
    meta = json.loads((DATA_DIR / "adoption_synthetic.meta.json").read_text())
    for key in ("market_size_N", "T_total", "T_observed", "ground_truth"):
        assert key in meta


@pytest.mark.parametrize("model_path", sorted(MODELS_DIR.glob("model_*.py")))
def test_model_runs(model_path):
    """Every model file must run to completion and print numeric output."""
    out = _run_model(model_path)
    assert "Log-likelihood" in out or "Test performance" in out, (
        f"{model_path.name} produced no recognized output:\n{out[:400]}"
    )


def test_final_model_recovers_ground_truth():
    """
    Model 9 is the reference final spec. Against the synthetic data its fit
    should be close to the data-generating ground truth.
    """
    out = _run_model(MODELS_DIR / "model_09_burr_xii_exposure_season_FINAL.py")
    meta = json.loads((DATA_DIR / "adoption_synthetic.meta.json").read_text())
    truth = meta["ground_truth"]

    def _grab(name: str) -> float:
        m = re.search(rf"{re.escape(name)}\s*=\s*([\-0-9.]+)", out)
        assert m, f"could not find {name} in model_09 output"
        return float(m.group(1))

    r = _grab("r")
    alpha = _grab("alpha")
    c = _grab("c")
    beta_exp = _grab("beta_exp")
    beta_season = _grab("beta_season")

    # Tolerances are wide because the dataset has noise and the population
    # likelihood is survivor-dominated; ranges below are sanity bands.
    assert 0.02 < r < 0.30, f"r = {r}"
    assert 50 < alpha < 600, f"alpha = {alpha}"
    assert 1.0 < c < 1.8, f"c = {c}"
    assert abs(beta_exp - truth["beta_exposure"]) < 0.4, f"beta_exp = {beta_exp}"
    assert abs(beta_season - truth["beta_season"]) < 0.4, f"beta_season = {beta_season}"
