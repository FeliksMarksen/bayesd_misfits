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

# Include resource_rw.py only if it exists (resource-rational branch)
RESOURCE_FILE=""
if [ -f "bayesd_misfits/resource_rw.py" ]; then
    RESOURCE_FILE="bayesd_misfits/resource_rw.py"
fi

tar czf "$TARFILE" \
    research/run_comparison.py \
    bayesd_misfits/__init__.py \
    bayesd_misfits/model.py \
    bayesd_misfits/hgf.py \
    bayesd_misfits/data.py \
    $RESOURCE_FILE \
    tests/test_hgf.py 2>/dev/null || true

echo "   ✓ $(du -h "$TARFILE" | cut -f1) of code packaged"

# ── Step 2: Provision GPU VM ─────────────────────────────────────────────────
echo ""
echo "🖥️  Provisioning Colab T4 GPU VM..."
colab new --gpu T4 -s "$SESSION"
echo "   ✓ VM ready"

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

colab exec -s "$SESSION" << 'PYTHON_EOF'
import subprocess, os, sys, json

print("=" * 60)
print("Setting up Colab environment...")
print(f"Python {sys.version_info.major}.{sys.version_info.minor}")
print("=" * 60)

# Install HSSM from git — Colab already has Python 3.12 + GPU JAX.
# This follows the same pattern as the Carney ARIA Workshop 2026 notebooks.
print("\nInstalling HSSM + dependencies...")
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "git+https://github.com/lnccbrown/HSSM.git@main",
    "arviz", "pyhgf", "pyarrow", "huggingface_hub",
    "scipy", "pandas",
], check=True)
print("✓ Dependencies installed")

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

# Run the comparison
print("\n" + "=" * 60)
print("Starting model comparison (FULL_RUN=1)...")
print("=" * 60)

os.chdir("/root/bayesd_misfits")
os.environ["FULL_RUN"] = "1"

result = subprocess.run([sys.executable, "research/run_comparison.py"])

if result.returncode != 0:
    print(f"\n✗ Comparison exited with code {result.returncode}")
else:
    print("\n✓ Comparison completed!")

# Print results summary
results_path = "research/model_comparison_results.json"
if os.path.exists(results_path):
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    with open(results_path) as f:
        data = json.load(f)
    print(f"Best model: {data.get('best_model', '?')}")
    print(f"Train: {data['split']['train_trajectories']}, "
          f"Valid: {data['split']['validation_trajectories']}")
    print()
    print(f"{'Model':<28} {'NLL':>9} {'SE':>9} {'Diag':>6}")
    print("-" * 55)
    for name, r in sorted(data.get("results", {}).items(),
                          key=lambda x: x[1].get("heldout_nll_mean", 999)):
        if r.get("status") == "ok":
            diag = "PASS" if r.get("eligible") else "FAIL"
            print(f"{name:<28} {r['heldout_nll_mean']:>9.4f} "
                  f"{r['heldout_nll_se']:>9.5f} {diag:>6}")
        else:
            print(f"{name:<28} {'--':>9} {'--':>9} {r.get('status','?'):>6}")
    print(f"{'Uniform random':<28} {0.6931*2:>9.4f}")
else:
    print("✗ Results file not found")

PYTHON_EOF

# ── Step 5: Download results ─────────────────────────────────────────────────
echo ""
echo "📥 Downloading results..."
colab download -s "$SESSION" \
    /root/bayesd_misfits/research/model_comparison_results.json \
    "$RESULT_FILE" 2>/dev/null && echo "   ✓ Results saved to $RESULT_FILE" || \
    echo "   ⚠ Could not download. Try: colab download -s $SESSION /root/bayesd_misfits/research/model_comparison_results.json"

# ── Step 6: Release VM ───────────────────────────────────────────────────────
echo ""
echo "🧹 Releasing VM..."
colab stop -s "$SESSION" 2>/dev/null && echo "   ✓ VM released" || \
    echo "   ⚠ Run manually: colab stop -s $SESSION"

echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  Done! Results in $RESULT_FILE (if download succeeded)              ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"