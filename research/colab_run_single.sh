#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# MindRL Model Comparison — Run one model on Colab and download results
# ──────────────────────────────────────────────────────────────────────────────
#
# Usage (from the bayesd_misfits repo root):
#
#   bash research/colab_run_single.sh RW
#   bash research/colab_run_single.sh HGF
#   bash research/colab_run_single.sh RW+Race
#
# This script is also called by research/colab_run_all.sh.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SESSION="mindrl"
MODEL_NAME="${1:-}"

if [ -z "$MODEL_NAME" ]; then
    echo "Usage: bash research/colab_run_single.sh <MODEL_NAME>"
    echo "Example: bash research/colab_run_single.sh RW"
    exit 1
fi

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  MindRL Single-Model Colab Runner: $MODEL_NAME"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

# ── Step 1: Create tarball of local code ─────────────────────────────────────
echo "📦 Creating code tarball..."
TARFILE="/tmp/bayesd_misfits_code.tar.gz"
STAGE_DIR="/tmp/bayesd_misfits_stage"

rm -rf "$STAGE_DIR"
mkdir -p "$STAGE_DIR/bayesd_misfits/research"
mkdir -p "$STAGE_DIR/bayesd_misfits/bayesd_misfits"
mkdir -p "$STAGE_DIR/bayesd_misfits/tests"

cp research/run_comparison.py research/run_single_model.py "$STAGE_DIR/bayesd_misfits/research/"
cp bayesd_misfits/__init__.py bayesd_misfits/model.py \
   bayesd_misfits/hgf.py bayesd_misfits/data.py \
   "$STAGE_DIR/bayesd_misfits/bayesd_misfits/"
[ -f "bayesd_misfits/resource_rw.py" ] && \
    cp bayesd_misfits/resource_rw.py "$STAGE_DIR/bayesd_misfits/bayesd_misfits/"
cp tests/test_hgf.py "$STAGE_DIR/bayesd_misfits/tests/"

tar czf "$TARFILE" -C "$STAGE_DIR" bayesd_misfits

echo "   ✓ $(du -h "$TARFILE" | cut -f1) of code packaged"

# ── Step 2: Provision or reuse GPU VM ────────────────────────────────────────
echo ""
echo "🖥️  Provisioning Colab T4 GPU VM..."
if colab sessions 2>/dev/null | grep -q "^\[$SESSION\]"; then
    echo "   ✓ Reusing existing session '$SESSION'"
    echo "   🔄 Restarting kernel to ensure clean state..."
    colab restart-kernel -s "$SESSION" || true
else
    colab new --gpu T4 -s "$SESSION"
    echo "   ✓ VM ready"
fi

# ── Step 3: Upload code ───────────────────────────────────────────────────────
echo ""
echo "📤 Uploading code to VM..."
colab upload -s "$SESSION" "$TARFILE" /root/code.tar.gz
echo "   ✓ Code uploaded"

# ── Step 4: Install deps + start single model in the background ──────────────
echo ""
echo "⚙️  Installing dependencies and starting model '$MODEL_NAME' in the background..."
echo "   This VM-side background job keeps running if you close your laptop."
echo "   Estimated time: ~30-60 min per model on T4 GPU."
echo ""

LOG_PATH="/root/run_single_${MODEL_NAME//+/_}.log"

colab exec -s "$SESSION" --timeout 7200 << PYTHON_EOF
import subprocess, os, sys

# Shell-injected constants (heredoc is unquoted so these expand)
MODEL_NAME = "$MODEL_NAME"
LOG_PATH = "$LOG_PATH"

print("=" * 60)
print("Setting up Colab environment...")
print(f"Python {sys.version_info.major}.{sys.version_info.minor}")
print("=" * 60)

print("\nInstalling HSSM...")
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "git+https://github.com/lnccbrown/HSSM.git@main",
    "arviz", "pyhgf", "pyarrow", "huggingface_hub",
    "scipy", "pandas",
], check=True)
print("HSSM installed")

print("\nReconciling JAX / CUDA12...")
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q", "-U", "jax[cuda12]",
], check=True)
print("JAX reconciled")

print("\nUpgrading numba for NumPy 2.x...")
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q", "numba>=0.61",
], check=True)
print("numba upgraded")

print("\nUpgrading NumPyro for JAX 0.11 compatibility...")
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q", "numpyro>=0.21",
], check=True)
print("NumPyro upgraded")

print("\nVerifying GPU...")
import jax
print(f"JAX devices: {jax.devices()}")

print("\nExtracting code...")
subprocess.run(["tar", "xzf", "/root/code.tar.gz", "-C", "/root"], check=True)
print("✓ Code extracted")

print("\nPre-downloading dataset...")
from pathlib import Path
from huggingface_hub import hf_hub_download
DATA_DIR = Path("/root/bayesd_misfits/hf_cache/public")
DATA_DIR.mkdir(parents=True, exist_ok=True)
for filename in [
    "public_train.jsonl",
    "public_train_reward_schedules.jsonl",
    "schema.json",
    "task_description.md",
]:
    try:
        hf_hub_download(
            repo_id="mindrl-hub/mindrl-challenge-public",
            filename=filename,
            repo_type="dataset",
            local_dir=str(DATA_DIR),
        )
        print(f"  ↓ {filename}")
    except Exception as e:
        print(f"  ✗ {filename}: {type(e).__name__}: {e}")
print("✓ Dataset ready")

print("\n" + "=" * 60)
print(f"Starting background model: {MODEL_NAME!r}")
print("=" * 60)

os.chdir("/root/bayesd_misfits")
run_env = os.environ.copy()
run_env["FULL_RUN"] = "1"
run_env["PYTHONUNBUFFERED"] = "1"

# Clean up stale log
for f in [LOG_PATH]:
    try:
        os.remove(f)
    except FileNotFoundError:
        pass

process = subprocess.Popen(
    [sys.executable, "-u", "research/run_single_model.py", MODEL_NAME],
    cwd="/root/bayesd_misfits",
    env=run_env,
    stdout=open(LOG_PATH, "w"),
    stderr=subprocess.STDOUT,
    start_new_session=True,  # detach from colab exec's process group
)

print(f"✓ Started VM background process (pid={process.pid})")
print(f"✓ Log file: {LOG_PATH}")
print(f"✓ Results incrementally written to: research/incremental_model_comparison_results.json")

PYTHON_EOF

# ── Step 5: Print next steps ─────────────────────────────────────────────────
echo ""
echo "🚀 Model '$MODEL_NAME' is running in the background on Colab."
echo "   You can close your laptop now."
echo ""
echo "   Check live log:  colab exec -s $SESSION --timeout 30 << 'PYEOF'"
echo "                       with open('$LOG_PATH') as f: print(f.read()[-2000:])"
echo "                       PYEOF"
echo "   Poll + download:  bash research/colab_poll_single.sh $MODEL_NAME"
echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  VM left running. Poll for results with colab_poll_single.sh      ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
