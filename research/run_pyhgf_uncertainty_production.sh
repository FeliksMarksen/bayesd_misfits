#!/usr/bin/env bash
set -euo pipefail

REPO="/home/feliks/Feliksier/Feliksier/Work/Master/02/Workshop Materials/bayesd_misfits"
cd "$REPO"

export MPLCONFIGDIR=/tmp/bayesd-pyhgf-production-mpl
export PYTENSOR_FLAGS=base_compiledir=/tmp/bayesd-pyhgf-production-pytensor
export FULL_RUN=1
export N_TRAIN=100
export N_VALID=300
export N_CHAINS=4
export N_TUNE=1000
export N_DRAWS=1000
export N_NLL_DRAWS=500
export SAMPLER=numpyro
export OUTPUT_PATH=research/pyhgf_uncertainty_production_results.json
export SAVE_POSTERIOR_PATH=research/pyhgf_uncertainty_production_posterior.nc

mkdir -p research/logs
exec .venv/bin/python research/run_single_model.py \
  'PyHGF+Uncertainty+Sticky' \
  >> research/logs/pyhgf_uncertainty_production.log 2>&1
