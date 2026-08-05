#!/usr/bin/env python3
# ──────────────────────────────────────────────────────────────────────────────
# MindRL Model Comparison — Colab Notebook Setup
# ──────────────────────────────────────────────────────────────────────────────
#
# HOW TO USE:
#
# 1. Open https://colab.research.google.com in your browser
# 2. Runtime → Change runtime type → T4 GPU → Save
# 3. Click the folder icon (left sidebar) → drag research/bayesd_code.tar.gz
#    into Colab (or use the upload button)
# 4. Create a new code cell, paste ALL of this file into it, and run
# 5. Wait 30-60 min — results print at the bottom and save to /content/results.json
# 6. Download results.json from the file browser
#
# To create the tarball locally:
#   cd bayesd_misfits
#   tar czf research/bayesd_code.tar.gz \
#       research/run_comparison.py \
#       bayesd_misfits/__init__.py \
#       bayesd_misfits/model.py \
#       bayesd_misfits/hgf.py \
#       bayesd_misfits/data.py \
#       bayesd_misfits/resource_rw.py \
#       tests/test_hgf.py
# ──────────────────────────────────────────────────────────────────────────────

import subprocess, os, sys, json

# ── 1. Install HSSM ──────────────────────────────────────────────────────────
print("Installing HSSM + dependencies...")
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "git+https://github.com/lnccbrown/HSSM.git@main",
    "arviz", "pyhgf", "pyarrow", "huggingface_hub", "scipy", "pandas",
], check=True)
print("✓ Dependencies installed")

# ── 2. Check GPU ──────────────────────────────────────────────────────────────
try:
    import jax
    devices = jax.devices()
    print(f"JAX devices: {devices}")
    has_gpu = any("gpu" in str(d).lower() or "cuda" in str(d).lower() for d in devices)
    print(f"{'✓ GPU detected!' if has_gpu else '⚠ No GPU — running on CPU'}")
except Exception as e:
    print(f"⚠ JAX check: {e}")

# ── 3. Extract code ───────────────────────────────────────────────────────────
print("\nExtracting code...")
TARFILE = "/content/bayesd_code.tar.gz"
if not os.path.exists(TARFILE):
    # Try alternative names
    for alt in ["/content/bayesd_misfits_code.tar.gz", "/content/code.tar.gz"]:
        if os.path.exists(alt):
            TARFILE = alt
            break
    else:
        raise FileNotFoundError(
            f"Upload the tarball first! Expected one of: "
            f"bayesd_code.tar.gz, bayesd_misfits_code.tar.gz, code.tar.gz "
            f"in /content/"
        )

subprocess.run(["tar", "xzf", TARFILE, "-C", "/content"], check=True)
os.chdir("/content/bayesd_misfits")
print(f"✓ Code extracted to /content/bayesd_misfits")

# ── 4. Run the comparison ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Starting model comparison (FULL_RUN=1)...")
print("Settings: 8 chains × 1500 tune × 1500 draws")
print("Estimated time: 30-60 min on GPU, 2-4 hours on CPU")
print("=" * 60)

os.environ["FULL_RUN"] = "1"
result = subprocess.run([sys.executable, "research/run_comparison.py"])

# ── 5. Print results ──────────────────────────────────────────────────────────
results_path = "research/model_comparison_results.json"
if os.path.exists(results_path):
    # Copy to /content for easy download
    import shutil
    shutil.copy(results_path, "/content/results.json")

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
    print(f"\n✓ Results saved to /content/results.json — download it from the file browser!")
else:
    print(f"\n✗ Results not found at {results_path}")
    print(f"  Exit code: {result.returncode}")