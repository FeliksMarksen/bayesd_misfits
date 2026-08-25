"""Fit and score a single named model for the MindRL comparison.

This is a companion to run_comparison.py that runs one model at a time so
that long Colab free-tier runs can make incremental progress. Each invocation
loads existing incremental results, fits one model, and writes the updated
results back.

Usage
-----
Inside the bayesd_misfits repo root on the VM::

    python research/run_single_model.py RW
    python research/run_single_model.py HGF
    python research/run_single_model.py PyHGF
    python research/run_single_model.py RW+Race

The incremental output file is fixed at::

    research/incremental_model_comparison_results.json

Environment variables (same as run_comparison.py):
    FULL_RUN, N_TRAIN, N_VALID, N_CHAINS, N_TUNE, N_DRAWS, N_NLL_DRAWS, SAMPLER
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Import the comparison machinery from run_comparison.py.
# run_comparison.py only executes main() under __name__ == "__main__", so
# importing it is safe.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_comparison import (
    FULL_RUN,
    N_CHAINS,
    N_DRAWS,
    N_NLL_DRAWS,
    N_TRAIN,
    N_TUNE,
    N_VALID,
    SAMPLER,
    build_models,
    fit_model,
    heldout_population_nll,
    heldout_race_nll,
    load_split_data,
    population_parameter_summary,
    sampler_diagnostics,
)

OUTPUT_PATH = Path(
    os.environ.get(
        "OUTPUT_PATH",
        "research/incremental_model_comparison_results.json",
    )
)


def load_or_init_results() -> dict:
    """Load existing incremental results or create a fresh template."""
    if OUTPUT_PATH.exists():
        with OUTPUT_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"Loaded existing incremental results ({len(data.get('results', {}))} models).")
        return data

    return {
        "run_type": "incremental",
        "note": "Results built one model at a time via run_single_model.py",
        "seed": 20260719,
        "sampler": SAMPLER,
        "chains": N_CHAINS,
        "tune": N_TUNE,
        "draws": N_DRAWS,
        "posterior_draws_for_scoring": N_NLL_DRAWS,
        "n_train": N_TRAIN,
        "n_valid": N_VALID,
        "full_run": FULL_RUN,
        "results": {},
    }


def save_results(data: dict) -> None:
    """Atomically write results to disk."""
    tmp_path = OUTPUT_PATH.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, allow_nan=False)
    tmp_path.replace(OUTPUT_PATH)
    print(f"Saved incremental results to {OUTPUT_PATH}")


def run_single_model(name: str) -> None:
    """Fit and score one model, updating the incremental results file."""
    print("=" * 72)
    print(f"Incremental single-model run: {name}")
    print(f"FULL_RUN={FULL_RUN}, sampler={SAMPLER}, chains={N_CHAINS}, "
          f"tune={N_TUNE}, draws={N_DRAWS}")
    print("=" * 72)

    data = load_or_init_results()

    expected_run = {
        "sampler": SAMPLER,
        "chains": N_CHAINS,
        "tune": N_TUNE,
        "draws": N_DRAWS,
        "posterior_draws_for_scoring": N_NLL_DRAWS,
        "n_train": N_TRAIN,
        "n_valid": N_VALID,
    }
    mismatches = {
        key: (data.get(key), expected)
        for key, expected in expected_run.items()
        if key in data and data.get(key) != expected
    }
    if mismatches:
        details = ", ".join(
            f"{key}: file={old!r}, run={new!r}"
            for key, (old, new) in mismatches.items()
        )
        raise ValueError(
            f"OUTPUT_PATH mixes incompatible run configurations ({details})"
        )

    # Load data. The original run_comparison.py always includes RT because the
    # full comparison set always contains the race model (RW+Race). To keep
    # identical data splits, we do the same here.
    include_rt = True
    train_data, valid_data, split_info = load_split_data(include_rt=include_rt)
    data["split"] = split_info

    models = build_models()
    if name not in models:
        raise ValueError(
            f"Unknown model '{name}'. Available models: {', '.join(models.keys())}"
        )

    spec = models[name]

    previous = data.get("results", {}).get(name)
    if (
        os.environ.get("SKIP_COMPLETED", "0") == "1"
        and previous is not None
        and previous.get("status") == "ok"
    ):
        print(f"Skipping {name}: a completed result already exists.")
        return

    data["grouping"] = {
        "rl_sequence": "participant_id column containing one trajectory ID",
        "hierarchical_random_effect": "subject_id containing the real human subject",
        "validation_split": "disjoint real human subjects",
    }

    print(f"\n{'=' * 72}\nFitting {name}...")
    try:
        model, idata = fit_model(name, spec, train_data)
        diagnostics = sampler_diagnostics(idata)
        status = "PASS" if diagnostics["pass"] else "FAIL"
        rhat_text = (
            f"{diagnostics['max_rhat']:.3f}"
            if diagnostics["max_rhat"] is not None
            else "unavailable"
        )
        ess_text = (
            f"{diagnostics['min_bulk_ess']:.0f}"
            if diagnostics["min_bulk_ess"] is not None
            else "unavailable"
        )
        bfmi_text = (
            f"{diagnostics['min_bfmi']:.3f}"
            if diagnostics["min_bfmi"] is not None
            else "unavailable"
        )
        print(
            f"Diagnostics {status}: divergences={diagnostics['divergences']}, "
            f"max R-hat={rhat_text}, min ESS={ess_text}, min BFMI={bfmi_text}"
        )
        if diagnostics["failures"]:
            print("  " + "; ".join(diagnostics["failures"]))

        print("Scoring held-out trajectories with population parameters...")
        if spec.get("is_race", False):
            print("  (using HSSM log_likelihood for neural-network race model)")
            nll = heldout_race_nll(model, idata, valid_data, spec)
        else:
            nll = heldout_population_nll(idata, valid_data, spec)

        import numpy as np

        data["results"][name] = {
            "status": "ok",
            "eligible": diagnostics["pass"],
            "hierarchy": spec.get("hierarchy", "subject"),
            "heldout_nll_mean": float(np.mean(nll)),
            "heldout_nll_sd": float(np.std(nll, ddof=1)),
            "heldout_nll_se": float(np.std(nll, ddof=1) / np.sqrt(len(nll))),
            "heldout_nll_q03": float(np.quantile(nll, 0.03)),
            "heldout_nll_q97": float(np.quantile(nll, 0.97)),
            "heldout_total_nll": float(np.sum(nll)),
            "diagnostics": diagnostics,
            "population_parameters": population_parameter_summary(
                idata, spec["params"], spec["bounds"]
            ),
            "parameter_bounds": {
                param: list(bounds) for param, bounds in spec["bounds"].items()
            },
        }
        posterior_path_value = os.environ.get("SAVE_POSTERIOR_PATH")
        if posterior_path_value:
            posterior_path = Path(posterior_path_value)
            posterior_path.parent.mkdir(parents=True, exist_ok=True)
            idata.to_netcdf(posterior_path)
            print(f"Saved complete posterior to {posterior_path}")
    except Exception as error:
        print(f"FAILED: {type(error).__name__}: {error}")
        data["results"][name] = {
            "status": "fit_failed",
            "eligible": False,
            "error": f"{type(error).__name__}: {error}",
        }
        raise  # re-raise so the caller (and log) sees the failure
    finally:
        save_results(data)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python research/run_single_model.py <MODEL_NAME>")
        print("Available models:")
        for name in build_models():
            print(f"  {name}")
        sys.exit(1)

    model_name = sys.argv[1]
    try:
        run_single_model(model_name)
    except Exception as e:
        print(f"\nModel {model_name} failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
