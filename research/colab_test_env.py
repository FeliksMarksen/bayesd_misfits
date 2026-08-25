"""Minimal Colab environment smoke test for the bayesd_misfits pipeline."""
from __future__ import annotations

import os
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/bayesd_misfits-matplotlib")
os.environ.setdefault("PYTENSOR_FLAGS", "base_compiledir=/tmp/bayesd_misfits-pytensor")

print("Python:", sys.version)

import subprocess
print("\n--- pip list (relevant packages) ---")
result = subprocess.run(
    [sys.executable, "-m", "pip", "list"],
    capture_output=True, text=True, check=True,
)
for line in result.stdout.splitlines():
    if any(pkg in line.lower() for pkg in ["numpy", "numba", "jax", "hssm", "bambi", "pymc", "pytensor"]):
        print(line)

print("\n--- NumPy version ---")
import numpy as np
print("numpy:", np.__version__)

print("\n--- JAX devices ---")
import jax
print("jax:", jax.__version__)
print("devices:", jax.devices())

print("\n--- HSSM import ---")
import hssm
print("hssm:", hssm.__version__)

print("\n--- HuggingFace dataset download ---")
from pathlib import Path
from huggingface_hub import hf_hub_download, HfApi
api = HfApi()
try:
    # Check if repo exists and is accessible
    info = api.repo_info(repo_id="mindrl-hub/mindrl-challenge-public", repo_type="dataset")
    print(f"✓ Dataset repo found: {info.id}")
except Exception as e:
    print(f"✗ Cannot access dataset repo: {type(e).__name__}: {e}")

DATA_DIR = Path("/root/bayesd_misfits/hf_cache/public")
DATA_DIR.mkdir(parents=True, exist_ok=True)
for filename in ["public_train.jsonl", "public_train_reward_schedules.jsonl", "schema.json", "task_description.md"]:
    try:
        downloaded = hf_hub_download(
            repo_id="mindrl-hub/mindrl-challenge-public",
            filename=filename,
            repo_type="dataset",
            local_dir=str(DATA_DIR),
        )
        print(f"  ↓ {filename} -> {downloaded}")
    except Exception as e:
        print(f"  ✗ {filename}: {type(e).__name__}: {e}")

print("\nFiles in data dir:", sorted(DATA_DIR.glob("*")))

print("\n--- Local module import + data load ---")
sys.path.insert(0, "/root/bayesd_misfits")
from bayesd_misfits.data import load_challenge_data

df = load_challenge_data(
    data_dir=DATA_DIR,
    feedback_transform="normalize",
    rt_placeholder=-1.0,
    group_by="subject",
)
print("Data shape:", df.shape)
print("Participants:", df["participant_id"].nunique())
print("Columns:", list(df.columns))

print("\n✓ Environment smoke test passed.")
