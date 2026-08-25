#!/bin/bash
set -euo pipefail
SESSION="mindrl-test"

echo "Reusing session '$SESSION' (keeps install + data)..."
if ! colab sessions 2>/dev/null | grep -q "^\[$SESSION\]"; then
    echo "Session not active, creating..."
    colab new --gpu T4 -s "$SESSION"
fi

echo "Uploading updated test script + code..."
# We only need to re-upload the changed script and the code tarball
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

colab upload -s "$SESSION" research/colab_test_env.py /root/colab_test_env.py
colab upload -s "$SESSION" "$TARFILE" /root/code.tar.gz

echo "Running smoke test..."
colab exec -s "$SESSION" --timeout 1800 << 'PYTHON_EOF'
import subprocess, sys, os, json

# Marker file to avoid reinstalling deps every iteration
MARKER = "/tmp/colab_deps_installed.marker"

if not os.path.exists(MARKER):
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

    open(MARKER, "w").close()
else:
    print("\n=== Dependencies already installed (skipping install) ===")

print("\n=== Extracting code ===")
subprocess.run(["tar", "xzf", "/root/code.tar.gz", "-C", "/root"], check=True)
print("✓ Code extracted")

process = subprocess.Popen(
    [sys.executable, "/root/colab_test_env.py"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
)
for line in process.stdout:
    print(line, end="")
process.wait()
if process.returncode != 0:
    raise SystemExit(f"Smoke test failed with exit code {process.returncode}")
PYTHON_EOF

echo "Done. (Session '$SESSION' is still running.)"
