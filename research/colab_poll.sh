#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# MindRL Model Comparison — Colab GPU Polling Script
# ──────────────────────────────────────────────────────────────────────────────
#
# This is meant to be used after `research/colab_run.sh` has started the
# comparison as a background job on the Colab VM.
#
# It checks the remote log, waits for results, downloads them, and stops the VM.
# You can run it from another machine / after closing your laptop.
#
# Usage (from the bayesd_misfits repo root):
#
#   bash research/colab_poll.sh
#
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SESSION="mindrl"
RESULT_FILE="./results_colab.json"
REMOTE_RESULTS="/root/bayesd_misfits/research/model_comparison_results.json"
REMOTE_LOG="/root/run_comparison.log"

POLL_INTERVAL=120  # seconds

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  MindRL Model Comparison — Poll for Colab Results                  ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

# Verify session exists
if ! colab sessions 2>/dev/null | grep -q "^\[$SESSION\]"; then
    echo "✗ No active session '$SESSION' found."
    echo "  If the VM was already stopped, you cannot recover this run."
    exit 1
fi

echo "✓ Session '$SESSION' is active"
echo ""
echo "Press Ctrl+C to stop polling (the VM will keep running)."
echo ""

while true; do
    echo "--- $(date '+%H:%M:%S') checking progress ---"

    # Show last ~30 lines of log (best-effort)
    colab exec -s "$SESSION" --timeout 30 -- "tail -n 30 $REMOTE_LOG" 2>/dev/null || true

    # Check whether results file exists
    if colab exec -s "$SESSION" --timeout 30 -- "test -f $REMOTE_RESULTS" 2>/dev/null; then
        echo ""
        echo "✓ Results file detected on VM!"
        echo ""
        echo "📥 Downloading results..."
        if colab download -s "$SESSION" "$REMOTE_RESULTS" "$RESULT_FILE" 2>/dev/null; then
            echo "✓ Results saved to $RESULT_FILE"
        else
            echo "✗ Download failed. You can retry manually with:"
            echo "   colab download -s $SESSION $REMOTE_RESULTS"
            exit 1
        fi

        echo ""
        echo "🧹 Stopping VM to save free GPU quota..."
        colab stop -s "$SESSION" 2>/dev/null && echo "✓ VM released" || \
            echo "⚠ Failed to stop VM. Run manually: colab stop -s $SESSION"

        echo ""
        echo "╔══════════════════════════════════════════════════════════════════════╗"
        echo "║  Done! Results in $RESULT_FILE                                     ║"
        echo "╚══════════════════════════════════════════════════════════════════════╝"
        exit 0
    fi

    echo ""
    echo "⏳ Results not ready yet. Sleeping ${POLL_INTERVAL}s..."
    echo ""
    sleep "$POLL_INTERVAL"
done
