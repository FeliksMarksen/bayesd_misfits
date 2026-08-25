#!/bin/bash
set -euo pipefail
SESSION="mindrl-test"

echo "Uploading probe script..."
colab upload -s "$SESSION" research/colab_numpyro_test.py /root/colab_numpyro_test.py

echo "Testing numpyro versions against current JAX..."
for npv in 0.16.1 0.17.0 0.18.0 0.19.0 0.20.0 0.21.0; do
    echo ""
    echo "=== Trying numpyro==$npv ==="
    colab exec -s "$SESSION" --timeout 600 << EOF
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "numpyro==$npv"], check=True)
print("numpyro $npv installed")
process = subprocess.Popen(
    [sys.executable, "/root/colab_numpyro_test.py"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
)
for line in process.stdout:
    print(line, end="")
process.wait()
EOF
done
