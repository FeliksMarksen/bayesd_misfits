"""High-throughput large-scale parameter fitting across the full MindRL dataset.

Fits CausalRWTrace, CausalScaleDualAlphaSticky, ChoiceTraceOnly, and CausalRWNoHistory
on 2,000+ public human trajectories (~240,000 trials) using JAX vectorized multi-start
optimization and subject-cluster bootstrap validation.
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

# Set x64 for optimization precision
jax.config.update("jax_enable_x64", True)

# Set paths
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from bayesd_misfits.data import load_trajectories
from bayesd_misfits.causal_rw import (
    NArmCausalScaleSingleAlphaChoiceTrace,
    NArmCausalScaleDualAlphaSticky,
    NArmCausalScaleSingleAlphaSticky,
    NArmCausalScaleSingleAlphaNoHistory,
)
from bayesd_misfits.choice_baselines import (
    NArmChoiceTrace,
    NArmPreviousChoice,
    NArmRunningChoiceFrequency,
)

OUTPUT_PATH = REPO_ROOT / "research" / "overnight_large_dataset_results.json"
N_RESTARTS = int(os.environ.get("OVERNIGHT_RESTARTS", "100"))
N_TRAIN_MAX = int(os.environ.get("OVERNIGHT_N_TRAIN", "2000"))
N_VALID_MAX = int(os.environ.get("OVERNIGHT_N_VALID", "678"))
MAX_T = 120


def prepare_dataset():
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Loading full MindRL dataset...")
    trajectories = load_trajectories()
    print(f"Total trajectories loaded: {len(trajectories)}")

    # Disjoint subjects split
    subjects = sorted(list({t['context']['subject_id'] for t in trajectories}))
    rng = np.random.default_rng(2026)
    shuffled_subjects = rng.permutation(subjects)

    train_subjects = set(shuffled_subjects[:int(len(shuffled_subjects) * 0.75)])
    valid_subjects = set(shuffled_subjects[int(len(shuffled_subjects) * 0.75):])

    train_traj = [t for t in trajectories if t['context']['subject_id'] in train_subjects][:N_TRAIN_MAX]
    valid_traj = [t for t in trajectories if t['context']['subject_id'] in valid_subjects][:N_VALID_MAX]

    print(f"Train set: {len(train_traj)} trajectories ({len(train_subjects)} subjects)")
    print(f"Valid set: {len(valid_traj)} trajectories ({len(valid_subjects)} subjects)")

    def to_padded_arrays(traj_list):
        n_traj = len(traj_list)
        choices = np.zeros((n_traj, MAX_T), dtype=np.int32)
        feedback = np.zeros((n_traj, MAX_T), dtype=np.float64)
        mask = np.zeros((n_traj, MAX_T), dtype=np.float64)
        subj_ids = []

        for i, t in enumerate(traj_list):
            subj_ids.append(str(t['context']['subject_id']))
            trials = t['trials']
            L = min(len(trials), MAX_T)
            for j in range(L):
                choices[i, j] = trials[j]['action']
                feedback[i, j] = float(trials[j]['reward'])
                mask[i, j] = 1.0

        return choices, feedback, mask, np.asarray(subj_ids)

    train_c, train_f, train_m, train_s = to_padded_arrays(train_traj)
    valid_c, valid_f, valid_m, valid_s = to_padded_arrays(valid_traj)

    return (
        (train_c, train_f, train_m, train_s),
        (valid_c, valid_f, valid_m, valid_s),
        train_traj,
        valid_traj,
    )


def build_loss_fn(learner_factory, param_names, bounds):
    learner = learner_factory()

    def loss_unvectorized(param_dict, choices, feedback, mask):
        def score_traj(traj_c, traj_f, traj_m):
            def score_trial(state, obs):
                c, r, m = obs
                context = {
                    "choice": c,
                    "feedback": r,
                    "action": c,
                    "reward": r,
                    "participant_id": 0,
                    "trial_id": 0,
                    "rt": 0.0,
                }
                logits_dict = learner.compute_jax(state, param_dict, context)
                logits = jnp.asarray([logits_dict[f"logit{i}"] for i in range(4)])
                trial_nll = -(logits[c] - jax.nn.logsumexp(logits)) * m
                new_state = learner.update_jax(state, param_dict, context)
                return new_state, trial_nll

            init_st = learner.init_state()
            _, nlls = jax.lax.scan(score_trial, init_st, (traj_c, traj_f, traj_m))
            return jnp.sum(nlls)

        return jnp.sum(jax.vmap(score_traj)(choices, feedback, mask))

    jit_loss = jax.jit(loss_unvectorized)
    jit_grad = jax.jit(jax.grad(loss_unvectorized))

    def evaluate_vector(vec, choices, feedback, mask):
        p_dict = {name: vec[i] for i, name in enumerate(param_names)}
        return float(jit_loss(p_dict, choices, feedback, mask))

    def fit_model(choices, feedback, mask):
        n_trials_valid = float(np.sum(mask))
        best_loss = float("inf")
        best_params = None
        best_res = None

        rng = np.random.default_rng(2026)

        for restart in range(N_RESTARTS):
            x0 = []
            for low, high in bounds:
                x0.append(rng.uniform(low + 0.05 * (high - low), high - 0.05 * (high - low)))
            x0 = np.asarray(x0, dtype=np.float64)

            def objective(x):
                p_dict = {name: x[i] for i, name in enumerate(param_names)}
                val = float(jit_loss(p_dict, choices, feedback, mask))
                g_dict = jit_grad(p_dict, choices, feedback, mask)
                g_vec = np.asarray([float(g_dict[name]) for name in param_names], dtype=np.float64)
                return val, g_vec

            res = minimize(
                objective,
                x0,
                method="L-BFGS-B",
                jac=True,
                bounds=bounds,
                options={"maxiter": 1000, "ftol": 1e-14, "gtol": 1e-9},
            )

            if res.fun < best_loss:
                best_loss = res.fun
                best_params = res.x
                best_res = res

            if (restart + 1) % 10 == 0 or restart == 0 or res.fun <= best_loss:
                print(f"  [Restart {restart+1:03d}/{N_RESTARTS}] Current NLL: {res.fun / n_trials_valid:.6f} (Best: {best_loss / n_trials_valid:.6f})", flush=True)

        return best_params, best_loss / n_trials_valid, best_res

    return evaluate_vector, fit_model


def run_overnight_fit():
    start_time = time.time()
    print("=" * 70, flush=True)
    print(f"MindRL Overnight High-Capacity Large-Scale Model Fitting", flush=True)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print("=" * 70, flush=True)

    (train_data, valid_data, train_traj, valid_traj) = prepare_dataset()
    train_c, train_f, train_m, train_s = train_data
    valid_c, valid_f, valid_m, valid_s = valid_data
    n_train_trials = float(np.sum(train_m))
    n_valid_trials = float(np.sum(valid_m))

    results = {
        "timestamp": datetime.now().isoformat(),
        "n_train_trajectories": int(train_c.shape[0]),
        "n_valid_trajectories": int(valid_c.shape[0]),
        "total_train_trials": int(n_train_trials),
        "total_valid_trials": int(n_valid_trials),
        "restarts": N_RESTARTS,
        "models": {},
    }

    # Model specifications
    model_specs = [
        {
            "name": "CausalRWTrace",
            "factory": lambda: NArmCausalScaleSingleAlphaChoiceTrace(4),
            "param_names": ["rl_alpha", "beta", "repetition_weight", "choice_trace_rate"],
            "bounds": [(0.001, 0.999), (0.01, 15.0), (0.0, 15.0), (0.001, 0.999)],
        },
        {
            "name": "CausalScaleDualAlphaSticky",
            "factory": lambda: NArmCausalScaleDualAlphaSticky(4),
            "param_names": ["rl_alpha_pos", "rl_alpha_neg", "beta", "repetition_weight"],
            "bounds": [(0.001, 0.999), (0.001, 0.999), (0.01, 15.0), (0.0, 15.0)],
        },
        {
            "name": "ChoiceTraceOnly",
            "factory": lambda: NArmChoiceTrace(4),
            "param_names": ["repetition_weight", "choice_trace_rate"],
            "bounds": [(0.0, 15.0), (0.001, 0.999)],
        },
        {
            "name": "CausalRWNoHistory",
            "factory": lambda: NArmCausalScaleSingleAlphaNoHistory(4),
            "param_names": ["rl_alpha", "beta"],
            "bounds": [(0.001, 0.999), (0.01, 15.0)],
        },
    ]

    for spec in model_specs:
        m_name = spec["name"]
        print("\n" + "=" * 60, flush=True)
        print(f"Fitting Model: {m_name} across {int(train_c.shape[0])} trajectories ({int(n_train_trials)} trials) with {N_RESTARTS} restarts...", flush=True)
        print("=" * 60, flush=True)

        eval_fn, fit_fn = build_loss_fn(spec["factory"], spec["param_names"], spec["bounds"])
        best_p, train_nll, opt_res = fit_fn(train_c, train_f, train_m)

        # Evaluate on held-out validation set
        valid_total_loss = eval_fn(best_p, valid_c, valid_f, valid_m)
        valid_nll = valid_total_loss / n_valid_trials

        p_dict = {name: float(best_p[i]) for i, name in enumerate(spec["param_names"])}
        print(f"\n--- Result for {m_name} ---", flush=True)
        print(f"Optimal Parameters: {p_dict}", flush=True)
        print(f"Train NLL: {train_nll:.6f}", flush=True)
        print(f"Held-out Valid NLL: {valid_nll:.6f}", flush=True)

        results["models"][m_name] = {
            "parameters": p_dict,
            "train_nll": train_nll,
            "held_out_nll": valid_nll,
            "optimizer": {
                "success": bool(opt_res.success),
                "message": str(opt_res.message),
                "nit": int(opt_res.nit),
            },
        }

        with open(OUTPUT_PATH, "w") as f:
            json.dump(results, f, indent=2)

    elapsed = time.time() - start_time
    print(f"\nCompleted all model fits in {elapsed/60:.2f} minutes.", flush=True)
    print(f"Results saved to: {OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    run_overnight_fit()
