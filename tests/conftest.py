"""
Pytest config: speed up the model-fit calls during tests.

The default 48-restart × maxiter=50000 Nelder-Mead loop in
models/_shared.py is appropriate for production but too slow for CI. The
environment variables below trim that to 8 restarts × maxiter=8000, which
still recovers the ground-truth parameters within tolerance on the
synthetic dataset.
"""
import os

os.environ.setdefault("AGENT_FIT_NSTARTS", "8")
os.environ.setdefault("AGENT_FIT_MAXITER", "8000")
