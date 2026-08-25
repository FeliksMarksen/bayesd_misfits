#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# MindRL Model Comparison — Poll for a single background model on Colab
# ──────────────────────────────────────────────────────────────────────────────
#
# Use this after research/colab_run_single.sh has started a model in the
# background on the Colab VM. It checks the log, waits for the incremental
# results file, downloads it, and stops the VM.
#
# Usage (from the bayesd_misfits repo root):
#
#   bash research/colab_poll_single.sh RW
#
# You can run this from any terminal, even a different machine, as long as
# colab-cli is authenticated with the same Google account.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SESSION="mindrl"
MODEL_NAME="${1:-}"
INCREMENTAL_LOCAL="./results_colab_incremental.json"
REMOTE_RESULTS="/root/bayesd_misfits/research/incremental_model_comparison_results.json"

if [ -z "$MODEL_NAME" ]; then
    echo "Usage: bash research/colab_poll_single.sh <MODEL_NAME>"
    echo "Example: bash research/colab_poll_single.sh RW"
    exit 1
fi

LOG_PATH="/root/run_single_${MODEL_NAME//+/_}.log"
POLL_INTERVAL=120  # seconds

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  Poll for model: $MODEL_NAME"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

# Verify session exists
if ! colab sessions 2>/dev/null | grep -q "^\[$SESSION\]"; then
    echo "✗ No active session '$SESSION' found."
    echo "  If the VM stopped, the background process is gone."
    echo "  Check whether $INCREMENTAL_LOCAL already has partial results."
    exit 1
fi

echo "✓ Session '$SESSION' is active"
echo ""
echo "Press Ctrl+C to stop polling (the VM will keep running)."
echo ""

while true; do
    echo "--- $(date '+%H:%M:%S') checking progress ---"

    # Show last ~25 lines of log
    colab exec -s "$SESSION" --timeout 30 << PYEOF 2>/dev/null || true
import os
log_path = "$LOG_PATH"
if os.path.exists(log_path):
    with open(log_path, "r") as f:
        content = f.read()
    print(f"LOG ({len(content)} bytes total):")
    print(content[-2500:] if len(content) > 2500 else content)
else:
    print("Log file not found yet.")
PYEOF

    # Try to download the results file. If it exists, this succeeds.
    if colab download -s "$SESSION" "$REMOTE_RESULTS" "$INCREMENTAL_LOCAL" 2>/dev/null; then
        echo ""
        echo "✓ Results downloaded to $INCREMENTAL_LOCAL"
        echo ""
        echo "🧹 Stopping VM to save free GPU quota..."
        colab stop -s "$SESSION" 2>/dev/null && echo "✓ VM released" || \
            echo "⚠ Failed to stop VM. Run manually: colab stop -s $SESSION"

        echo ""
        echo "╔══════════════════════════════════════════════════════════════════════╗"
        echo "║  Model $MODEL_NAME done. Results in $INCREMENTAL_LOCAL"
        echo "╚══════════════════════════════════════════════════════════════════════╝"
        exit 0
    fi

    echo ""
    echo "⏳ Results not ready yet. Sleeping ${POLL_INTERVAL}s..."
    echo ""
    sleep "$POLL_INTERVAL"
done
