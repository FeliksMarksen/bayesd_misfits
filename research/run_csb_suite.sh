#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export N_TRAIN="${N_TRAIN:-50}"
export N_VALID="${N_VALID:-100}"
export N_CHAINS="${N_CHAINS:-4}"
export N_TUNE="${N_TUNE:-500}"
export N_DRAWS="${N_DRAWS:-500}"
export N_NLL_DRAWS="${N_NLL_DRAWS:-200}"

export CSB_N_SAMPLES="${CSB_N_SAMPLES:-10}"
export CSB_N_PARTICLES="${CSB_N_PARTICLES:-32}"
export CSB_OBS_SIGMA="${CSB_OBS_SIGMA:-0.1}"
export CSB_FORGETTING="${CSB_FORGETTING:-0.98}"

export OUTPUT_PATH="${OUTPUT_PATH:-research/csb_and_trace_results.json}"
export MPLCONFIGDIR="/tmp/bayesd_misfits-matplotlib"
export PYTENSOR_FLAGS="base_compiledir=/tmp/bayesd_misfits-pytensor"

echo "========================================================================"
echo "Starting CSB Fitting Suite"
echo "N_TRAIN=$N_TRAIN, N_VALID=$N_VALID, N_CHAINS=$N_CHAINS, N_TUNE=$N_TUNE, N_DRAWS=$N_DRAWS"
echo "CSB_N_SAMPLES=$CSB_N_SAMPLES, CSB_N_PARTICLES=$CSB_N_PARTICLES, CSB_OBS_SIGMA=$CSB_OBS_SIGMA, CSB_FORGETTING=$CSB_FORGETTING"
echo "OUTPUT_PATH=$OUTPUT_PATH"
echo "========================================================================"

echo "[1/2] Fitting CausalSamplingBandit..."
.venv/bin/python research/run_single_model.py CausalSamplingBandit

echo "[2/2] Fitting CausalSamplingBanditTrace..."
.venv/bin/python research/run_single_model.py CausalSamplingBanditTrace

echo "All models finished successfully!"
