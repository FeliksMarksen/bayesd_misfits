#!/bin/bash
set -euo pipefail
SESSION="mindrl-test"
TEST_SCRIPT="research/colab_test_env.py"

echo "Starting/reusing session '$SESSION'..."
if colab sessions 2>/dev/null | grep -q "^\[$SESSION\]"; then
    echo "Reusing existing session"
else
    colab new --gpu T4 -s "$SESSION"
fi

echo "Creating code tarball..."
TARFILE="/tmp/bayesd_misfits_code.tar.gz"
STAGE_DIR="/tmp/bayesd_misfits_stage"
rm -rf "$STAGE_DIR"
mkdir -p "$STAGE_DIR/bayesd_misfits/research"
mkdir -p "$STAGE_DIR/bayesd_misfits/bayesd_misfits"
mkdir -p "$STAGE_DIR/bayesd_misfits/tests"
cp research/run_comparison.py research/colab_test_env.py "$STAGE_DIR/bayesd_misfits/research/"
cp bayesd_misfits/__init__.py bayesd_misfits/model.py \
   bayesd_misfits/hgf.py bayesd_misfits/data.py \
   "$STAGE_DIR/bayesd_misfits/bayesd_misfits/"
[ -f "bayesd_misfits/resource_rw.py" ] && \
    cp bayesd_misfits/resource_rw.py "$STAGE_DIR/bayesd_misfits/bayesd_misfits/"
cp tests/test_hgf.py "$STAGE_DIR/bayesd_misfits/tests/"
tar czf "$TARFILE" -C "$STAGE_DIR" bayesd_misfits

echo "Uploading test script + code..."
colab upload -s "$SESSION" "$TEST_SCRIPT" /root/colab_test_env.py
colab upload -s "$SESSION" "$TARFILE" /root/code.tar.gz

echo "Installing deps + running smoke test..."
colab exec -s "$SESSION" --timeout 600 << 'PYTHON_EOF'
import subprocess, sys
print("\n=== Installing HSSM + deps ===")
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "git+https://github.com/lnccbrown/HSSM.git@main",
    "arviz", "pyhgf", "pyarrow", "huggingface_hub",
    "scipy", "pandas",
], check=True)
print("✓ HSSM installed")

print("\n=== Reconciling JAX[cuda12] ===")
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q", "-U", "jax[cuda12]",
], check=True)
print("✓ JAX reconciled")

print("\n=== Upgrading numba for NumPy 2.x compatibility ===")
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q", "numba>=0.61",
], check=True)
print("✓ numba upgraded")

print("\n=== Extracting code ===")
subprocess.run(["tar", "xzf", "/root/code.tar.gz", "-C", "/root"], check=True)
print("✓ Code extracted")
process = subprocess.Popen(
    [sys.executable, "/root/colab_test_env.py"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
)
for line in process.stdout:
    print(line, end="")
process.wait()
if process.returncode != 0:
    raise SystemExit(f"Smoke test failed with exit code {process.returncode}")
PYTHON_EOF

echo "Stopping session..."
colab stop -s "$SESSION" 2>/dev/null || true
