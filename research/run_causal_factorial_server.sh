#!/usr/bin/env bash
# Restartable production comparison for causal-scale RW variants.

set -u

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

MODEL_SUITE="${1:-${MODEL_SUITE:-core}}"
N_TRAIN="${N_TRAIN:-300}"
N_VALID="${N_VALID:-300}"
N_CHAINS="${N_CHAINS:-4}"
N_TUNE="${N_TUNE:-1000}"
N_DRAWS="${N_DRAWS:-1000}"
N_NLL_DRAWS="${N_NLL_DRAWS:-500}"
SAMPLER="${SAMPLER:-numpyro}"
RUN_TAG="${RUN_TAG:-causal_factorial_n${N_TRAIN}}"
OUTPUT_PATH="${OUTPUT_PATH:-research/${RUN_TAG}_results.json}"
LOG_DIR="${LOG_DIR:-research/${RUN_TAG}_logs}"
POSTERIOR_DIR="${POSTERIOR_DIR:-research/${RUN_TAG}_posteriors}"
SAVE_POSTERIORS="${SAVE_POSTERIORS:-0}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"

export FULL_RUN=0
export N_TRAIN N_VALID N_CHAINS N_TUNE N_DRAWS N_NLL_DRAWS SAMPLER
export OUTPUT_PATH SKIP_COMPLETED
export XLA_FLAGS="${XLA_FLAGS:---xla_force_host_platform_device_count=${N_CHAINS}}"

pooled_models=(
  CausalPooledSingleNoHistory
  CausalPooledDualNoHistory
  CausalPooledSingleImmediate
  CausalPooledDualImmediate
  CausalPooledSingleTrace
  CausalPooledDualTrace
)

subject_models=(
  CausalSubjectSingleNoHistory
  CausalSubjectDualNoHistory
  CausalSubjectSingleImmediate
  CausalSubjectDualImmediate
  CausalSubjectSingleTrace
  CausalSubjectDualTrace
)

extended_models=(
  PyHGF
  PyHGF+Uncertainty
  PyHGF+Uncertainty+Sticky
  Resource
)

case "$MODEL_SUITE" in
  pooled)
    models=("${pooled_models[@]}")
    ;;
  subject)
    models=("${subject_models[@]}")
    ;;
  core)
    models=("${pooled_models[@]}" "${subject_models[@]}")
    ;;
  extended)
    models=(
      "${pooled_models[@]}"
      "${subject_models[@]}"
      "${extended_models[@]}"
    )
    ;;
  *)
    echo "Unknown suite '$MODEL_SUITE'. Use pooled, subject, core, or extended." >&2
    exit 2
    ;;
esac

mkdir -p "$LOG_DIR"
if [[ "$SAVE_POSTERIORS" == "1" ]]; then
  mkdir -p "$POSTERIOR_DIR"
fi

echo "Causal comparison configuration"
echo "  suite=$MODEL_SUITE"
echo "  train=$N_TRAIN validation=$N_VALID"
echo "  chains=$N_CHAINS tune=$N_TUNE draws=$N_DRAWS scoring_draws=$N_NLL_DRAWS"
echo "  sampler=$SAMPLER"
echo "  output=$OUTPUT_PATH"
echo "  XLA_FLAGS=$XLA_FLAGS"
echo "  started=$(date --iso-8601=seconds)"
echo

failures=0
for model in "${models[@]}"; do
  safe_name="${model//+/_plus_}"
  log_path="$LOG_DIR/${safe_name}.log"
  if [[ "$SAVE_POSTERIORS" == "1" ]]; then
    export SAVE_POSTERIOR_PATH="$POSTERIOR_DIR/${safe_name}.nc"
  else
    unset SAVE_POSTERIOR_PATH
  fi

  echo "========================================================================"
  echo "Starting $model at $(date --iso-8601=seconds)"
  echo "Log: $log_path"
  echo "========================================================================"

  uv run python research/run_single_model.py "$model" 2>&1 | tee -a "$log_path"
  model_status=${PIPESTATUS[0]}
  if [[ "$model_status" -ne 0 ]]; then
    failures=$((failures + 1))
    echo "$model failed with exit status $model_status; continuing."
  fi
  echo
done

echo "Completed at $(date --iso-8601=seconds) with $failures failed model(s)."
echo "Incremental results: $OUTPUT_PATH"
exit "$failures"
