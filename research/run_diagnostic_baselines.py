"""Fit frozen diagnostic baselines and score disjoint held-out subjects.

Unlike the main HSSM comparison, this script uses deterministic maximum
likelihood for a small set of pooled models.  Its purpose is mechanism
decomposition, not posterior inference.  Every held-out prediction uses fixed
training estimates plus trajectory-local state updated only from already
revealed choices and rewards.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bayesd_misfits.causal_rw import (
    NArmCausalScaleSingleAlphaChoiceTrace,
    NArmCausalScaleSingleAlphaNoHistory,
    NArmCausalScaleSingleAlphaSticky,
)
from bayesd_misfits.choice_baselines import (
    NArmChoiceTrace,
    NArmPreviousChoice,
    NArmRunningChoiceFrequency,
)
from run_comparison import N_TRAIN, N_TRIALS, N_VALID, SEED, load_split_data

# SciPy's convergence and line-search tolerances assume double precision.  The
# HSSM comparison selects float32 at import time for NUTS performance, but this
# small deterministic optimizer is more reliable in float64.
jax.config.update("jax_enable_x64", True)


OUTPUT_PATH = Path(
    os.environ.get(
        "DIAGNOSTIC_OUTPUT_PATH",
        f"research/diagnostic_baselines_n{N_TRAIN}_results.json",
    )
)
N_ACTIONS = 4
N_RESTARTS = int(os.environ.get("DIAGNOSTIC_RESTARTS", "5"))
BOOTSTRAP_DRAWS = int(os.environ.get("DIAGNOSTIC_BOOTSTRAP_DRAWS", "5000"))


def trajectory_arrays(data) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return identically ordered choice, feedback, and human-subject arrays."""
    trajectories = [
        trajectory.sort_values("trial_id")
        for _, trajectory in data.groupby("participant_id", sort=True)
    ]
    choices = np.stack(
        [trajectory["response"].to_numpy(dtype=np.int32) for trajectory in trajectories]
    )
    feedback = np.stack(
        [trajectory["feedback"].to_numpy(dtype=np.float32) for trajectory in trajectories]
    )
    subjects = np.asarray(
        [str(trajectory["subject_id"].iloc[0]) for trajectory in trajectories]
    )
    return choices, feedback, subjects


def build_nll_function(learner_factory: Callable[[], Any], param_names: list[str]):
    """Build a JIT-compiled function returning per-trial NLL."""
    learner = learner_factory()

    def nll(vector, choices, feedback):
        params = {name: vector[index] for index, name in enumerate(param_names)}

        def score_trajectory(trajectory_choices, trajectory_feedback):
            def score_trial(state, observation):
                choice, reward = observation
                computed = learner.compute_jax(state, params, context={})
                logits = jnp.stack(
                    [computed[f"logit{action}"] for action in range(N_ACTIONS)]
                )
                trial_nll = -jax.nn.log_softmax(logits)[choice]
                updated = learner.update_jax(
                    state,
                    params,
                    context={"choice": choice, "feedback": reward},
                )
                return updated, trial_nll

            _, trial_nll = jax.lax.scan(
                score_trial,
                learner.init_jax_state(),
                (trajectory_choices, trajectory_feedback),
            )
            return trial_nll

        return jax.vmap(score_trajectory)(choices, feedback)

    return jax.jit(nll)


def fit_frozen_parameters(
    learner_factory: Callable[[], Any],
    param_names: list[str],
    bounds: list[tuple[float, float]],
    starts: list[list[float]],
    train_choices: np.ndarray,
    train_feedback: np.ndarray,
) -> tuple[dict[str, float], dict[str, Any], Callable[..., Any]]:
    """Fit pooled natural-scale parameters with deterministic multistart MLE."""
    nll_function = build_nll_function(learner_factory, param_names)
    choices_jax = jnp.asarray(train_choices)
    feedback_jax = jnp.asarray(train_feedback)

    def mean_nll(vector):
        return jnp.mean(nll_function(vector, choices_jax, feedback_jax))

    value_and_grad = jax.jit(jax.value_and_grad(mean_nll))

    def objective(vector):
        value, gradient = value_and_grad(jnp.asarray(vector))
        return float(value), np.asarray(gradient, dtype=np.float64)

    rng = np.random.default_rng(SEED + 40)
    candidate_starts = [np.asarray(start, dtype=float) for start in starts]
    while len(candidate_starts) < N_RESTARTS:
        candidate_starts.append(
            np.asarray([rng.uniform(lower, upper) for lower, upper in bounds])
        )

    attempts = []
    best = None
    for start in candidate_starts[:N_RESTARTS]:
        result = minimize(
            objective,
            start,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={
                "maxiter": 500,
                "maxls": 100,
                "ftol": 1e-10,
                "gtol": 1e-7,
            },
        )
        method = "L-BFGS-B"
        if not result.success:
            # A bounded derivative-free finish is inexpensive for these
            # one-to-four-parameter controls and avoids accepting a failed
            # line-search flag merely because its objective looks plausible.
            fallback = minimize(
                lambda vector: objective(vector)[0],
                result.x,
                method="Powell",
                bounds=bounds,
                options={"maxiter": 500, "ftol": 1e-10, "xtol": 1e-7},
            )
            if np.isfinite(fallback.fun) and (
                fallback.success or fallback.fun <= result.fun
            ):
                result = fallback
                method = "Powell fallback"
        attempts.append(
            {
                "method": method,
                "success": bool(result.success),
                "message": str(result.message),
                "mean_train_nll": float(result.fun),
                "iterations": int(result.nit),
                "parameters": {
                    name: float(value)
                    for name, value in zip(param_names, result.x, strict=True)
                },
            }
        )
        if np.isfinite(result.fun) and (best is None or result.fun < best.fun):
            best = result

    if best is None:
        raise RuntimeError("All optimization restarts produced non-finite objectives")

    parameters = {
        name: float(value)
        for name, value in zip(param_names, best.x, strict=True)
    }
    _, selected_gradient = objective(best.x)
    optimizer = {
        "method": "multistart L-BFGS-B with bounded Powell fallback",
        "selected_success": bool(best.success),
        "selected_message": str(best.message),
        "selected_iterations": int(best.nit),
        "selected_gradient_norm": float(np.linalg.norm(selected_gradient)),
        "restarts": attempts,
    }
    return parameters, optimizer, nll_function


def score_summary(nll: np.ndarray) -> dict[str, Any]:
    flat = np.asarray(nll, dtype=float).reshape(-1)
    per_trajectory = np.asarray(nll, dtype=float).mean(axis=1)
    return {
        "mean_nll": float(flat.mean()),
        "sd_nll": float(flat.std(ddof=1)),
        "se_nll": float(flat.std(ddof=1) / np.sqrt(flat.size)),
        "total_nll": float(flat.sum()),
        "trajectory_mean_nll": per_trajectory.tolist(),
    }


def constant_probability_nll(
    probabilities: np.ndarray, choices: np.ndarray
) -> np.ndarray:
    log_probabilities = np.log(np.asarray(probabilities, dtype=float))
    return -log_probabilities[choices]


def paired_improvement(
    candidate: dict[str, Any],
    reference: dict[str, Any],
    trajectory_subjects: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Summarize improvement with a paired human-subject cluster bootstrap."""
    candidate_nll = np.asarray(candidate["heldout"]["trajectory_mean_nll"])
    reference_nll = np.asarray(reference["heldout"]["trajectory_mean_nll"])
    improvement = reference_nll - candidate_nll
    if trajectory_subjects.shape != improvement.shape:
        raise ValueError("Each held-out trajectory must have one subject label")
    unique_subjects = np.unique(trajectory_subjects)
    subject_indices = [
        np.flatnonzero(trajectory_subjects == subject) for subject in unique_subjects
    ]
    subject_sums = np.asarray(
        [improvement[indices].sum() for indices in subject_indices]
    )
    subject_counts = np.asarray([len(indices) for indices in subject_indices])
    subject_means = subject_sums / subject_counts
    sampled = rng.integers(
        0,
        len(subject_indices),
        size=(BOOTSTRAP_DRAWS, len(subject_indices)),
    )
    # Resample whole people, retaining all of each selected person's
    # trajectories.  Sums/counts make this exactly equivalent to repeatedly
    # concatenating the selected clusters, but fully vectorized.
    bootstrap_means = (
        subject_sums[sampled].sum(axis=1)
        / subject_counts[sampled].sum(axis=1)
    )
    return {
        "mean_nll_improvement": float(improvement.mean()),
        "cluster_bootstrap_se": float(bootstrap_means.std(ddof=1)),
        "bootstrap_q025": float(np.quantile(bootstrap_means, 0.025)),
        "bootstrap_q975": float(np.quantile(bootstrap_means, 0.975)),
        "trajectory_win_fraction": float(np.mean(improvement > 0.0)),
        "subject_win_fraction": float(np.mean(subject_means > 0.0)),
        "bootstrap_unit": "held-out human subject",
        "n_subjects": int(len(unique_subjects)),
    }


def main() -> None:
    if N_RESTARTS < 1:
        raise ValueError("DIAGNOSTIC_RESTARTS must be positive")
    if BOOTSTRAP_DRAWS < 1:
        raise ValueError("DIAGNOSTIC_BOOTSTRAP_DRAWS must be positive")

    print("=" * 72)
    print("Causal diagnostic baseline decomposition")
    print(f"train={N_TRAIN}, validation={N_VALID}, trials={N_TRIALS}")
    print(f"optimizer restarts={N_RESTARTS}, bootstrap draws={BOOTSTRAP_DRAWS}")
    print("=" * 72)

    train_data, valid_data, split_info = load_split_data(include_rt=False)
    train_choices, train_feedback, _ = trajectory_arrays(train_data)
    valid_choices, valid_feedback, valid_subjects = trajectory_arrays(valid_data)

    results: dict[str, dict[str, Any]] = {}
    uniform = np.full(N_ACTIONS, 1.0 / N_ACTIONS)
    train_counts = np.bincount(train_choices.reshape(-1), minlength=N_ACTIONS) + 1.0
    global_probabilities = train_counts / train_counts.sum()

    for name, probabilities, rationale in [
        ("Uniform", uniform, "chance-level four-action control"),
        (
            "GlobalActionFrequency",
            global_probabilities,
            "fixed action-label frequencies estimated from training subjects only",
        ),
    ]:
        results[name] = {
            "fit_method": "closed_form",
            "rationale": rationale,
            "parameters": {"action_probabilities": probabilities.tolist()},
            "train": score_summary(
                constant_probability_nll(probabilities, train_choices)
            ),
            "heldout": score_summary(
                constant_probability_nll(probabilities, valid_choices)
            ),
        }

    running_factory = lambda: NArmRunningChoiceFrequency(N_ACTIONS)
    running_nll_function = build_nll_function(running_factory, [])
    empty = jnp.asarray([], dtype=jnp.float32)
    results["RunningChoiceFrequency"] = {
        "fit_method": "none",
        "rationale": (
            "causal within-trajectory action counts with one pseudo-count per action"
        ),
        "parameters": {"prior_count": 1.0},
        "train": score_summary(
            np.asarray(
                running_nll_function(
                    empty, jnp.asarray(train_choices), jnp.asarray(train_feedback)
                )
            )
        ),
        "heldout": score_summary(
            np.asarray(
                running_nll_function(
                    empty, jnp.asarray(valid_choices), jnp.asarray(valid_feedback)
                )
            )
        ),
    }

    specifications = [
        {
            "name": "PreviousChoiceOnly",
            "rationale": "previous choice without reward learning",
            "factory": lambda: NArmPreviousChoice(N_ACTIONS),
            "params": ["repetition_weight"],
            "bounds": [(-5.0, 5.0)],
            "starts": [[0.0], [1.0], [2.0]],
        },
        {
            "name": "ChoiceTraceOnly",
            "rationale": "gradual choice history without reward learning",
            "factory": lambda: NArmChoiceTrace(N_ACTIONS),
            "params": ["repetition_weight", "choice_trace_rate"],
            "bounds": [(-5.0, 5.0), (0.001, 1.0)],
            "starts": [[1.0, 0.5], [2.5, 0.33], [1.0, 1.0]],
        },
        {
            "name": "CausalRW",
            "rationale": "single-alpha causal-scale reward learning without choice history",
            "factory": lambda: NArmCausalScaleSingleAlphaNoHistory(N_ACTIONS),
            "params": ["rl_alpha", "beta"],
            "bounds": [(0.001, 0.999), (0.0, 15.0)],
            "starts": [[0.3, 2.0], [0.7, 1.5], [0.9, 0.75]],
        },
        {
            "name": "CausalRWImmediate",
            "rationale": "single-alpha causal-scale reward learning plus previous choice",
            "factory": lambda: NArmCausalScaleSingleAlphaSticky(N_ACTIONS),
            "params": ["rl_alpha", "beta", "repetition_weight"],
            "bounds": [(0.001, 0.999), (0.0, 15.0), (-5.0, 5.0)],
            "starts": [[0.7, 1.0, 1.5], [0.3, 2.0, 1.0], [0.9, 0.75, 2.0]],
        },
        {
            "name": "CausalRWTrace",
            "rationale": "single-alpha causal-scale reward learning plus gradual choice trace",
            "factory": lambda: NArmCausalScaleSingleAlphaChoiceTrace(N_ACTIONS),
            "params": [
                "rl_alpha",
                "beta",
                "repetition_weight",
                "choice_trace_rate",
            ],
            "bounds": [
                (0.001, 0.999),
                (0.0, 15.0),
                (-5.0, 5.0),
                (0.001, 1.0),
            ],
            "starts": [
                [0.9, 0.75, 2.5, 0.33],
                [0.7, 1.0, 1.5, 0.5],
                [0.3, 2.0, 1.0, 0.8],
            ],
        },
    ]

    for spec in specifications:
        print(f"\nFitting {spec['name']}...")
        parameters, optimizer, nll_function = fit_frozen_parameters(
            spec["factory"],
            spec["params"],
            spec["bounds"],
            spec["starts"],
            train_choices,
            train_feedback,
        )
        vector = jnp.asarray([parameters[name] for name in spec["params"]])
        train_nll = np.asarray(
            nll_function(vector, jnp.asarray(train_choices), jnp.asarray(train_feedback))
        )
        valid_nll = np.asarray(
            nll_function(vector, jnp.asarray(valid_choices), jnp.asarray(valid_feedback))
        )
        results[spec["name"]] = {
            "fit_method": "multistart_mle",
            "rationale": spec["rationale"],
            "parameters": parameters,
            "parameter_bounds": {
                name: list(bounds)
                for name, bounds in zip(spec["params"], spec["bounds"], strict=True)
            },
            "optimizer": optimizer,
            "train": score_summary(train_nll),
            "heldout": score_summary(valid_nll),
        }
        print(
            f"  train NLL={results[spec['name']]['train']['mean_nll']:.6f}; "
            f"held-out NLL={results[spec['name']]['heldout']['mean_nll']:.6f}; "
            f"parameters={parameters}"
        )

    comparisons = {
        "global_action_bias_vs_uniform": ("GlobalActionFrequency", "Uniform"),
        "running_frequency_vs_uniform": ("RunningChoiceFrequency", "Uniform"),
        "previous_choice_vs_uniform": ("PreviousChoiceOnly", "Uniform"),
        "choice_trace_vs_previous_choice": ("ChoiceTraceOnly", "PreviousChoiceOnly"),
        "reward_learning_vs_uniform": ("CausalRW", "Uniform"),
        "rw_immediate_vs_rw": ("CausalRWImmediate", "CausalRW"),
        "rw_trace_vs_rw_immediate": ("CausalRWTrace", "CausalRWImmediate"),
        "reward_added_to_choice_trace": ("CausalRWTrace", "ChoiceTraceOnly"),
        "choice_trace_vs_reward_only": ("ChoiceTraceOnly", "CausalRW"),
        "choice_trace_vs_rw_immediate": ("ChoiceTraceOnly", "CausalRWImmediate"),
        "rw_trace_vs_rw": ("CausalRWTrace", "CausalRW"),
    }
    rng = np.random.default_rng(SEED + 41)
    paired = {
        label: {
            "candidate": candidate,
            "reference": reference,
            **paired_improvement(
                results[candidate], results[reference], valid_subjects, rng
            ),
        }
        for label, (candidate, reference) in comparisons.items()
    }

    output = {
        "run_type": "causal_diagnostic_baselines",
        "seed": SEED,
        "n_train": N_TRAIN,
        "n_valid": N_VALID,
        "n_trials": N_TRIALS,
        "fit_scope": "pooled fixed parameters estimated from training subjects only",
        "evaluation_rule": (
            "fixed parameters plus trajectory-local state updated from already revealed history"
        ),
        "optimizer_restarts": N_RESTARTS,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "split": split_info,
        "results": results,
        "paired_heldout_improvements": paired,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(OUTPUT_PATH)

    print("\nHeld-out mean NLL")
    for name, result in sorted(
        results.items(), key=lambda item: item[1]["heldout"]["mean_nll"]
    ):
        print(f"  {name:<26} {result['heldout']['mean_nll']:.6f}")
    print(f"\nSaved results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
