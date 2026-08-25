#!/usr/bin/env bash
set -euo pipefail

# Production fit for the scale-free submitted Dual-Alpha + repetition model.
# The learner performs causal running reward standardization internally.
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/bayesd-misfits-uv-cache}"
export OUTPUT_PATH="research/causal_scale_production_results.json"
export SAVE_POSTERIOR_PATH="research/causal_scale_production_posterior.nc"
export N_TRAIN=100
export N_VALID=300
export N_CHAINS=4
export N_TUNE=1000
export N_DRAWS=1000
export N_NLL_DRAWS=500
export SAMPLER=numpyro

uv run python research/run_single_model.py CausalScaleSticky
