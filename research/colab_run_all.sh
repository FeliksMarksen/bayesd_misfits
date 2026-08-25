#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# MindRL Model Comparison — Print remaining per-model Colab commands
# ──────────────────────────────────────────────────────────────────────────────
#
# Each model is started as a detached background job on a Colab T4 VM. You can
# close your laptop after starting a model. Later, use colab_poll_single.sh to
# download results and stop the VM.
#
# Workflow:
#   1. bash research/colab_run_single.sh RW     # returns quickly
#   2. close laptop / wait ~30-60 min
#   3. bash research/colab_poll_single.sh RW  # download + stop VM
#   4. repeat for the next model
#
# This script lists the remaining models and prints the exact commands to run.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

INCREMENTAL_FILE="./results_colab_incremental.json"

MODELS=(
    "RW"
    "RW+Decay"
    "DualAlpha"
    "DualAlpha+Decay"
    "RW+Sticky"
    "Sticky"
    "HGF"
    "HGF+Sticky"
    "PyHGF"
    "PyHGF+Sticky"
    "PyHGF+Uncertainty"
    "PyHGF+Uncertainty+Sticky"
    "Resource"
    "RW+Race"
)

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  MindRL Model Comparison — Remaining per-model commands            ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

# Figure out which models are already done
DONE_MODELS=""
if [ -f "$INCREMENTAL_FILE" ]; then
    DONE_MODELS=$(python3 -c "
import json, sys
try:
    with open('$INCREMENTAL_FILE') as f:
        data = json.load(f)
    for name, r in data.get('results', {}).items():
        if r.get('status') == 'ok':
            print(name)
except Exception as e:
    print(f'Error reading incremental file: {e}', file=sys.stderr)
" 2>&1 || true)
fi

# Build list of remaining models
REMAINING=()
for model in "${MODELS[@]}"; do
    if [ -n "$DONE_MODELS" ] && echo "$DONE_MODELS" | grep -qx "$model"; then
        echo "✓ $model already completed"
    else
        REMAINING+=("$model")
    fi
done

if [ ${#REMAINING[@]} -eq 0 ]; then
    echo ""
    echo "All models already completed! Results in $INCREMENTAL_FILE"
    exit 0
fi

echo ""
echo "Models remaining: ${#REMAINING[@]}"
echo ""
for model in "${REMAINING[@]}"; do
    echo "----------------------------------------------------------------------"
    echo "  $model"
    echo "----------------------------------------------------------------------"
    echo "  Start (returns quickly, can close laptop):"
    echo "    bash research/colab_run_single.sh $model"
    echo ""
    echo "  Later, poll / download / stop VM:"
    echo "    bash research/colab_poll_single.sh $model"
    echo ""
done

echo "Tip: run one model, poll for results, then run the next."
echo "     This keeps free Colab usage per session low and lets you close your laptop."
