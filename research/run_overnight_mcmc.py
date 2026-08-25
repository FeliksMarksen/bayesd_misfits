"""Overnight 4-Chain NumPyro NUTS MCMC Production Fitting.

Fits CausalPooledSingleTrace (CausalRWTrace) with 4 chains x 1,000 tune + 1,000 draws
using NumPyro NUTS, saving full posterior traces and convergence metrics.
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Paths
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "research"))

# Environment variables for MCMC
os.environ["FULL_RUN"] = "1"
os.environ["SAMPLER"] = "numpyro"
os.environ["N_CHAINS"] = "4"
os.environ["N_TUNE"] = "1000"
os.environ["N_DRAWS"] = "1000"
os.environ["N_NLL_DRAWS"] = "500"
os.environ["N_TRAIN"] = os.environ.get("N_TRAIN", "300")
os.environ["N_VALID"] = os.environ.get("N_VALID", "300")
os.environ["OUTPUT_PATH"] = str(REPO_ROOT / "research" / "overnight_mcmc_results.json")
os.environ["SAVE_POSTERIOR_PATH"] = str(REPO_ROOT / "research" / "overnight_rw_trace_posterior.nc")

from run_single_model import main

if __name__ == "__main__":
    print(f"[{datetime.now().isoformat()}] Starting 4-Chain NumPyro NUTS MCMC fitting for CausalPooledSingleTrace...")
    # Run MCMC on CausalPooledSingleTrace (CausalRWTrace)
    sys.argv = ["run_single_model.py", "CausalPooledSingleTrace"]
    main()
    print(f"[{datetime.now().isoformat()}] MCMC fitting complete!")
