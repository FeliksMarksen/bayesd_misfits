#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# MindRL Model Comparison — Colab GPU Runner (for private repos)
# ──────────────────────────────────────────────────────────────────────────────
#
# Usage (from the bayesd_misfits repo root):
#
#   bash research/colab_run.sh
#
# What it does:
#   1. Tars the local code files (no repo access needed)
#   2. Provisions a T4 GPU VM on Colab
#   3. Uploads the code tarball
#   4. Installs HSSM + deps (Colab already has Python 3.12 + GPU JAX)
#   5. Runs the full comparison
#   6. Downloads the results JSON
#   7. Releases the VM
#
# Based on the Carney ARIA Workshop 2026 Colab setup pattern.
# The entire run takes ~30-60 min on GPU (vs 4+ hours on laptop).
# Results are saved to ./results_colab.json
#
# Prerequisites:
#   - colab-cli installed and authenticated (pip install colab-cli && colab auth)
#   - Run from the bayesd_misfits repo root
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SESSION="mindrl"
RESULT_FILE="./results_colab.json"

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  MindRL Model Comparison — Colab GPU Runner                        ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

# ── Step 1: Create tarball of local code ─────────────────────────────────────
echo "📦 Creating code tarball..."
TARFILE="/tmp/bayesd_misfits_code.tar.gz"
STAGE_DIR="/tmp/bayesd_misfits_stage"

# Stage files under a single top-level directory so extraction on the VM
# produces /root/bayesd_misfits/{research,bayesd_misfits,tests}
rm -rf "$STAGE_DIR"
mkdir -p "$STAGE_DIR/bayesd_misfits/research"
mkdir -p "$STAGE_DIR/bayesd_misfits/bayesd_misfits"
mkdir -p "$STAGE_DIR/bayesd_misfits/tests"

cp research/run_comparison.py "$STAGE_DIR/bayesd_misfits/research/"
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
    colab restart-kernel -s "$SESSION"
else
    colab new --gpu T4 -s "$SESSION"
    echo "   ✓ VM ready"
fi

# ── Step 3: Upload code ───────────────────────────────────────────────────────
echo ""
echo "📤 Uploading code to VM..."
colab upload -s "$SESSION" "$TARFILE" /root/code.tar.gz
echo "   ✓ Code uploaded"

# ── Step 4: Install deps + run comparison ────────────────────────────────────
echo ""
echo "⚙️  Installing dependencies and running comparison..."
echo "   (This takes 30-60 min on GPU. Go get a coffee.)"
echo ""

colab exec -s "$SESSION" --timeout 7200 << 'PYTHON_EOF'
import subprocess, os, sys, json

print("=" * 60)
print("Setting up Colab environment...")
print(f"Python {sys.version_info.major}.{sys.version_info.minor}")
print("=" * 60)

# Install HSSM, then pin JAX + numpyro to versions we know work.
# Our local dev env uses JAX 0.4.31 + numpyro 0.19.0 with HSSM 0.4.0.
# Colab or HSSM deps may install incompatible versions (missing xla_pmap_p).
print("
Installing HSSM...")
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "git+https://github.com/lnccbrown/HSSM.git@main",
    "arviz", "pyhgf", "pyarrow", "huggingface_hub",
    "scipy", "pandas",
], check=True)
print("Dependencies installed")

# Pin JAX + numpyro to known-good versions (matches our local env).
# Newer JAX removed xla_pmap_p which numpyro 0.19 needs.
print("Pinning JAX + numpyro to compatible versions...")
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "jax==0.4.31", "numpyro==0.19.0",
], check=True)
print("JAX + numpyro pinned")

# Verify GPU
try:
    import jax
    devices = jax.devices()
    print(f"JAX devices: {devices}")
    has_gpu = any("gpu" in str(d).lower() or "cuda" in str(d).lower() for d in devices)
    if has_gpu:
        print("✓ GPU detected — sampling will be accelerated!")
    else:
        print("⚠ No GPU — running on CPU (slower but works)")
except Exception as e:
    print(f"⚠ JAX check failed: {e}")

# Extract code
print("\nExtracting code...")
subprocess.run(["tar", "xzf", "/root/code.tar.gz", "-C", "/root"], check=True)
print("✓ Code extracted")

# The first time HSSM runs in this session, it may need to download the
# MindRL Challenge dataset from HuggingFace. Pre-download it explicitly so
# run_comparison.py does not fail with a missing-file error.
print("\nPre-downloading MindRL Challenge dataset from HuggingFace...")
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

# Run the comparison as a DETACHED background process on the VM.
# This lets you close your laptop; the job keeps running on Colab's hardware.
print("\n" + "=" * 60)
print("Starting model comparison in the background (FULL_RUN=1)...")
print("You can close your laptop now. Check progress later with:")
print("   colab exec -s mindrl --timeout 30 -- 'tail -20 /root/run_comparison.log'")
print("=" * 60)

os.chdir("/root/bayesd_misfits")
run_env = os.environ.copy()
run_env["FULL_RUN"] = "1"
run_env["PYTHONUNBUFFERED"] = "1"

log_path = "/root/run_comparison.log"
results_path = "/root/bayesd_misfits/research/model_comparison_results.json"

# Clean up any stale results/log from a previous run
for f in [log_path, results_path]:
    try:
        os.remove(f)
    except FileNotFoundError:
        pass

process = subprocess.Popen(
    [sys.executable, "-u", "research/run_comparison.py"],
    cwd="/root/bayesd_misfits",
    env=run_env,
    stdout=open(log_path, "w"),
    stderr=subprocess.STDOUT,
    start_new_session=True,  # detach from colab exec's process group
)

print(f"✓ Started background process (pid={process.pid})")
print(f"✓ Log file: {log_path}")
print(f"✓ Results will be written to: {results_path}")
print("\n👉 Use research/colab_poll.sh to check progress and download results.")

PYTHON_EOF

# ── Step 5: Print next steps ─────────────────────────────────────────────────
echo ""
echo "🚀 Comparison is running in the background on Colab."
echo "   Check live log:  colab exec -s $SESSION --timeout 30 -- 'tail -20 /root/run_comparison.log'"
echo "   Poll + download: bash research/colab_poll.sh"
echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  VM left running. Download + stop with colab_poll.sh              ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"