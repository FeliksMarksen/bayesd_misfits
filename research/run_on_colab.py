#!/usr/bin/env python3
"""Run the MindRL model comparison on Google Colab GPU via colab-cli.

Usage (from your local machine):

    # One-shot: provision GPU VM, run, print results, release VM
    colab run --gpu T4 research/run_on_colab.py

    # Or step by step for more control:
    colab new --gpu T4
    colab exec -f research/run_on_colab.py
    colab download /root/bayesd_misfits/research/model_comparison_results.json ./results_colab.json
    colab stop

    # Override branch or settings via env vars:
    colab run --gpu T4 research/run_on_colab.py  # uses BRANCH=resource-rational by default

Environment variables:
    BRANCH        Git branch to clone (default: resource-rational)
    FULL_RUN      Set to "1" for production settings (default: 1)
    N_TRAIN       Override training trajectories (default: 100)
    N_VALID       Override validation trajectories (default: 300)
    N_CHAINS      Override MCMC chains (default: 4)
    N_DRAWS       Override MCMC draws (default: 1000)

The results are saved to /root/bayesd_misfits/research/model_comparison_results.json
on the Colab VM. Use `colab download` to retrieve them.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys


BRANCH = os.environ.get("BRANCH", "resource-rational")
REPO_URL = "https://github.com/FeliksMarksen/bayesd_misfits.git"
REPO_DIR = "/root/bayesd_misfits"


def run(cmd: str, check: bool = True, env: dict | None = None) -> subprocess.CompletedProcess:
    """Run a shell command, printing it first."""
    print(f"\n$ {cmd}", flush=True)
    result = subprocess.run(cmd, shell=True, check=False, env=env)
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed (exit {result.returncode}): {cmd}")
    return result


def main():
    print("=" * 72)
    print("MindRL Model Comparison — Colab GPU Runner")
    print(f"Branch: {BRANCH}")
    print(f"Python: {sys.version}")
    print("=" * 72)

    # ── Step 1: Check Python version ─────────────────────────────────
    # HSSM requires Python >=3.12. Colab VMs may have 3.10 or 3.11.
    if sys.version_info < (3, 12):
        print("\n⚠ HSSM requires Python 3.12+. Attempting to install via conda...")
        run("conda install -y -c conda-forge python=3.12", check=False)
        # Re-exec with the new Python if conda updated it
        if os.path.exists("/opt/conda/bin/python3.12"):
            os.execv("/opt/conda/bin/python3.12", ["/opt/conda/bin/python3.12", __file__])
        print("⚠ Could not install Python 3.12. HSSM may fail to install.")
        print("  Manual fix: conda install python=3.12, then re-run.")
    else:
        print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} — good")

    # ── Step 2: Install dependencies ────────────────────────────────
    print("\n" + "=" * 72)
    print("Installing dependencies (this takes a few minutes)...")
    print("=" * 72)

    # HSSM from git (includes jax, numpyro, bambi, pymc under the hood)
    run("pip install -q 'hssm @ git+https://github.com/lnccbrown/HSSM.git@main'")

    # Additional deps for the comparison script
    run("pip install -q arviz pyhgf pyarrow huggingface_hub scipy pandas")

    # Verify JAX sees the GPU
    try:
        import jax
        devices = jax.devices()
        print(f"\n✓ JAX devices: {devices}")
        if any("gpu" in str(d).lower() for d in devices):
            print("✓ GPU detected — sampling will be accelerated")
        else:
            print("⚠ No GPU detected — will run on CPU (still works, just slower)")
    except ImportError:
        print("⚠ Could not import JAX — something went wrong with installation")

    # ── Step 3: Clone the repository ────────────────────────────────
    print("\n" + "=" * 72)
    print(f"Cloning {REPO_URL} (branch: {BRANCH})...")
    print("=" * 72)

    if os.path.exists(REPO_DIR):
        run(f"rm -rf {REPO_DIR}")
    run(f"git clone --depth 1 --branch {BRANCH} {REPO_URL} {REPO_DIR}")
    os.chdir(REPO_DIR)
    print(f"✓ Repository cloned to {REPO_DIR}")

    # ── Step 4: Run the comparison ───────────────────────────────────
    print("\n" + "=" * 72)
    print("Starting model comparison (FULL_RUN=1)...")
    print("This will take 30-60 min on GPU, 2-4 hours on CPU.")
    print("=" * 72)

    env = os.environ.copy()
    env["FULL_RUN"] = env.get("FULL_RUN", "1")
    # Don't override these if the user set them
    for var in ["N_TRAIN", "N_VALID", "N_CHAINS", "N_DRAWS", "N_TUNE", "N_NLL_DRAWS", "SAMPLER"]:
        if var in os.environ:
            env[var] = os.environ[var]

    result = run("python research/run_comparison.py", check=False, env=env)

    # ── Step 5: Print results summary ────────────────────────────────
    results_path = "research/model_comparison_results.json"
    if os.path.exists(results_path):
        print("\n" + "=" * 72)
        print("RESULTS")
        print("=" * 72)
        with open(results_path) as f:
            data = json.load(f)

        print(f"Run type: {data.get('run_type', '?')}")
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
                print(f"{name:<28} {'--':>9} {'--':>9} "
                      f"{r.get('status', '?'):>6}")
        print(f"{'Uniform random':<28} {0.6931*2:>9.4f}")

        print(f"\n✓ Results saved to {results_path}")
        print(f"  Retrieve with: colab download {REPO_DIR}/{results_path} ./results_colab.json")
    else:
        print("\n✗ Results file not found — the comparison may have failed.")
        print(f"  Exit code: {result.returncode}")

    print("\nDone. Run 'colab stop' to release the VM.")


if __name__ == "__main__":
    main()