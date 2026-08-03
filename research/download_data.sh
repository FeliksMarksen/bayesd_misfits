#!/usr/bin/env bash
# Download the MindRL Challenge public dataset from HuggingFace.
#
# Usage:
#   ./research/download_data.sh          # default: ../hf_cache/public
#   ./research/download_data.sh ./my_dir   # custom output directory
#
# Requires the `hf` CLI (installed automatically by `uv sync`).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

OUTPUT_DIR="${1:-$PROJECT_ROOT/../hf_cache/public}"
OUTPUT_DIR="$(cd "$(dirname "$OUTPUT_DIR")" 2>/dev/null && echo "$(pwd)/$(basename "$OUTPUT_DIR")" || echo "$OUTPUT_DIR")"

FILES=(
    "public_train.jsonl"
    "public_train_reward_schedules.jsonl"
    "schema.json"
    "task_description.md"
)

echo "Downloading MindRL Challenge dataset to: $OUTPUT_DIR"
echo

# Find the hf CLI: prefer the project venv, fall back to PATH
if [ -x "$PROJECT_ROOT/.venv/bin/hf" ]; then
    HF="$PROJECT_ROOT/.venv/bin/hf"
elif command -v hf &> /dev/null; then
    HF="hf"
else
    echo "Error: 'hf' CLI not found. Run 'uv sync' first to install dependencies." >&2
    exit 1
fi

"$HF" download mindrl-hub/mindrl-challenge-public \
    "${FILES[@]}" \
    --repo-type dataset \
    --local-dir "$OUTPUT_DIR"

echo
echo "Done ✓  Files saved to: $OUTPUT_DIR"
ls -lh "$OUTPUT_DIR"