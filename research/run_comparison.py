"""Fit and compare nine N-arm bandit learning models.

The comparison is deliberately deployment-oriented:

* bounded parameters use a generalized-logit link, including random effects;
* the training and validation sets contain disjoint human subjects;
* validation uses population-level parameters, matching the fixed submitted agent;
* the same learner implementations are used for fitting and one-step prediction;
* a model is eligible only when population/hyperparameter convergence and
  global HMC diagnostics pass.

Usage
-----
Quick diagnostic run::

    .venv/bin/python scripts/run_comparison.py

Production run::

    FULL_RUN=1 .venv/bin/python scripts/run_comparison.py

The sample sizes and sampler can also be overridden with ``N_TRAIN``,
``N_VALID``, ``N_TUNE``, ``N_DRAWS``, ``N_CHAINS``, ``N_NLL_DRAWS``, and
``SAMPLER`` environment variables.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/bayesd_misfits-matplotlib")
os.environ.setdefault(
    "PYTENSOR_FLAGS", "base_compiledir=/tmp/bayesd_misfits-pytensor"
)

import arviz as az
import hssm
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy.special import expit, logsumexp, logit
from ssms.rl import ModelConfig
from ssms.rl.env import Bandit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bayesd_misfits.data import ensure_data_downloaded, load_challenge_data
from bayesd_misfits.hgf import NArmHGF, NArmHGFSticky
from bayesd_misfits.model import (
    NArmDualAlphaRW,
    NArmRescorlaWagner,
    NArmRWDualAlphaSticky,
    NArmRWDriftLearner,
    NArmRWSticky,
)
from bayesd_misfits.resource_rw import NArmRWDualAlphaStickyResource

warnings.filterwarnings("ignore")
logging.getLogger("jax._src.xla_bridge").setLevel("ERROR")

FLOATX = "float32"
hssm.set_floatX(FLOATX, update_jax=True)

SEED = 20260719
FULL_RUN = os.environ.get("FULL_RUN", "0") == "1"
N_TRIALS = 120
N_TRAIN = int(os.environ.get("N_TRAIN", 100 if FULL_RUN else 20))
N_VALID = int(os.environ.get("N_VALID", 300 if FULL_RUN else 60))
N_CHAINS = int(os.environ.get("N_CHAINS", 4 if FULL_RUN else 2))
N_TUNE = int(os.environ.get("N_TUNE", 1000 if FULL_RUN else 500))
N_DRAWS = int(os.environ.get("N_DRAWS", 1000 if FULL_RUN else 500))
N_NLL_DRAWS = int(os.environ.get("N_NLL_DRAWS", 500 if FULL_RUN else 200))
SAMPLER = os.environ.get("SAMPLER", "numpyro")

MAX_RHAT = 1.01
MIN_BFMI = 0.30
MIN_BULK_ESS = max(100, N_CHAINS * N_DRAWS // 10)

# HGF constants shared by fitting and evaluation.
OBS_PRECISION = 20.0
INITIAL_SIGMA1 = 0.25
INITIAL_SIGMA2 = 1.0
INITIAL_MU1 = 0.5
INITIAL_MU2 = -1.0
THETA_VAR = 0.01

# Natural-scale prior targets: lower, upper, mean, SD, participant RE SD.
PRIOR_SPECS: dict[str, tuple[float, float, float, float, float]] = {
    "rl_alpha": (0.0, 1.0, 0.30, 0.15, 0.05),
    "rl_alpha_pos": (0.0, 1.0, 0.30, 0.15, 0.05),
    "rl_alpha_neg": (0.0, 1.0, 0.30, 0.15, 0.05),
    "rl_decay": (0.0, 0.15, 0.03, 0.03, 0.01),
    "sticky": (-3.0, 3.0, 0.0, 0.50, 0.10),
    "omega": (-8.0, 2.0, -2.0, 1.00, 0.50),
    "kappa": (0.0, 4.0, 1.0, 0.50, 0.10),
    "beta": (0.0, 15.0, 5.0, 2.00, 0.50),
    # Race-model decision parameters (for race_no_bias_angle_4)
    "scaler": (0.001, 10.0, 2.0, 1.00, 0.20),
    "a": (1.0, 3.0, 2.0, 0.30, 0.10),
    "z": (0.0, 0.9, 0.5, 0.10, 0.05),
    "t": (0.0, 2.0, 0.3, 0.10, 0.05),
    "theta": (-0.1, 1.45, 0.0, 0.30, 0.10),
    # Resource-rational extension parameters (Bruckner et al. 2025)
    "sticky_gain": (0.0, 5.0, 0.0, 0.50, 0.10),
    "fatigue_rate": (0.0, 0.1, 0.0, 0.02, 0.005),
    "surprise_gain": (0.0, 5.0, 0.0, 0.50, 0.10),
}


def _env_int(name: str, value: int) -> int:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


for _name, _value in {
    "N_TRAIN": N_TRAIN,
    "N_VALID": N_VALID,
    "N_CHAINS": N_CHAINS,
    "N_TUNE": N_TUNE,
    "N_DRAWS": N_DRAWS,
    "N_NLL_DRAWS": N_NLL_DRAWS,
}.items():
    _env_int(_name, _value)


def inverse_gen_logit(value: Any, bounds: tuple[float, float]) -> Any:
    """Map an unconstrained regression coefficient to its natural bounds."""
    lower, upper = bounds
    return lower + (upper - lower) * expit(value)


def hierarchical_param(name: str) -> hssm.Param:
    """Create a bounded hierarchy with priors expressed on link scale.

    The prior targets in ``PRIOR_SPECS`` are intuitive natural-scale values.
    A delta-method conversion puts both the intercept and random-effect scale
    on the generalized-logit linear-predictor scale used by Bambi/HSSM.
    """
    lower, upper, mean, sd, re_sd = PRIOR_SPECS[name]
    proportion = np.clip((mean - lower) / (upper - lower), 1e-6, 1.0 - 1e-6)
    eta_mean = float(logit(proportion))
    derivative = (upper - lower) * proportion * (1.0 - proportion)
    eta_sd = float(sd / derivative)
    eta_re_sd = float(re_sd / derivative)

    return hssm.Param(
        name,
        formula=f"{name} ~ 1 + (1|participant_id)",
        bounds=(lower, upper),
        link="log_logit",
        prior={
            "Intercept": hssm.Prior("Normal", mu=eta_mean, sigma=eta_sd),
            "1|participant_id": {
                "name": "Normal",
                "mu": 0.0,
                "sigma": {"name": "HalfNormal", "sigma": eta_re_sd},
            },
        },
    )


def load_split_data(include_rt: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load complete trajectories and split them by human subject."""
    ensure_data_downloaded()
    df = load_challenge_data(
        feedback_transform="normalize",
        rt_placeholder=-1.0,
        group_by="trajectory",
    )
    trial_counts = df.groupby("participant_id").size()
    complete_ids = trial_counts[trial_counts == N_TRIALS].index
    df = df[df["participant_id"].isin(complete_ids)].copy()

    rng = np.random.default_rng(SEED)
    subjects = np.asarray(sorted(df["subject_id"].unique()))
    rng.shuffle(subjects)
    split_at = max(1, int(0.70 * len(subjects)))
    train_subjects = set(subjects[:split_at])
    valid_subjects = set(subjects[split_at:])

    train_pool = np.asarray(sorted(
        df.loc[df["subject_id"].isin(train_subjects), "participant_id"].unique()
    ))
    valid_pool = np.asarray(sorted(
        df.loc[df["subject_id"].isin(valid_subjects), "participant_id"].unique()
    ))
    if len(train_pool) < N_TRAIN or len(valid_pool) < N_VALID:
        raise ValueError(
            "Requested split is larger than the available complete trajectories: "
            f"train {N_TRAIN}/{len(train_pool)}, validation {N_VALID}/{len(valid_pool)}"
        )

    train_ids = rng.choice(train_pool, size=N_TRAIN, replace=False)
    valid_ids = rng.choice(valid_pool, size=N_VALID, replace=False)

    def select(ids: np.ndarray) -> pd.DataFrame:
        selected = df[df["participant_id"].isin(ids)].copy()
        id_map = {
            old: new
            for new, old in enumerate(sorted(selected["participant_id"].unique()))
        }
        selected["participant_id"] = selected["participant_id"].map(id_map)
        selected = selected.sort_values(["participant_id", "trial_id"])
        cols = ["participant_id", "trial_id", "response", "feedback"]
        if include_rt:
            cols.append("rt")
        return selected[cols].reset_index(drop=True)

    train_data = select(train_ids)
    valid_data = select(valid_ids)
    split_info = {
        "available_complete_trajectories": int(len(complete_ids)),
        "available_subjects": int(len(subjects)),
        "train_trajectories": int(train_data["participant_id"].nunique()),
        "validation_trajectories": int(valid_data["participant_id"].nunique()),
        "train_rows": int(len(train_data)),
        "validation_rows": int(len(valid_data)),
        "subject_overlap": 0,
    }
    print(
        f"Training: {split_info['train_trajectories']} trajectories × {N_TRIALS} trials; "
        f"validation: {split_info['validation_trajectories']} × {N_TRIALS}; "
        "human subjects are disjoint."
    )
    return train_data, valid_data, split_info


def _make_config(
    description: str,
    learner: Any,
    env: Bandit,
    params: list[str],
    decision: str = "inv_temp_softmax_4",
    response: list[str] | None = None,
) -> ModelConfig:
    if response is None:
        response = ["response"]
    config = ModelConfig(
        decision,
        description,
        decision,
        learner,
        env,
        response=response,
    )
    config.bounds.update({name: PRIOR_SPECS[name][:2] for name in params})
    config.params_default = [PRIOR_SPECS[name][2] for name in config.list_params]
    config.validate()
    return config


def build_models() -> dict[str, dict[str, Any]]:
    """Build matched model specifications with explicit, verified bounds."""
    env = Bandit.bernoulli(
        probabilities=[0.25] * 4,
        response_labels=[0, 1, 2, 3],
    )

    definitions: list[tuple[str, str, Callable[[], Any], list[str]]] = [
        (
            "RW",
            "RW",
            lambda: NArmRescorlaWagner(4),
            ["rl_alpha", "beta"],
        ),
        (
            "RW+Decay",
            "RW+decay",
            lambda: NArmRescorlaWagner(4, use_decay=True),
            ["rl_alpha", "rl_decay", "beta"],
        ),
        (
            "DualAlpha",
            "DualAlpha",
            lambda: NArmDualAlphaRW(4),
            ["rl_alpha_pos", "rl_alpha_neg", "beta"],
        ),
        (
            "DualAlpha+Decay",
            "DualAlpha+decay",
            lambda: NArmDualAlphaRW(4, use_decay=True),
            ["rl_alpha_pos", "rl_alpha_neg", "rl_decay", "beta"],
        ),
        (
            "RW+Sticky",
            "RW+sticky",
            lambda: NArmRWSticky(4),
            ["rl_alpha", "sticky", "beta"],
        ),
        (
            "Sticky",
            "DualAlpha+sticky",
            lambda: NArmRWDualAlphaSticky(4),
            ["rl_alpha_pos", "rl_alpha_neg", "sticky", "beta"],
        ),
        (
            "HGF",
            "uHGF",
            lambda: NArmHGF(
                4,
                initial_mu1=INITIAL_MU1,
                initial_mu2=INITIAL_MU2,
                initial_sigma1=INITIAL_SIGMA1,
                initial_sigma2=INITIAL_SIGMA2,
                theta_var=THETA_VAR,
                obs_precision=OBS_PRECISION,
            ),
            ["omega", "kappa", "beta"],
        ),
        (
            "HGF+Sticky",
            "uHGF+sticky",
            lambda: NArmHGFSticky(
                4,
                initial_mu1=INITIAL_MU1,
                initial_mu2=INITIAL_MU2,
                initial_sigma1=INITIAL_SIGMA1,
                initial_sigma2=INITIAL_SIGMA2,
                theta_var=THETA_VAR,
                obs_precision=OBS_PRECISION,
            ),
            ["omega", "kappa", "sticky", "beta"],
        ),
        # ── Resource-rational extension (Bruckner et al. 2025) ──
        (
            "Resource",
            "DualAlpha+sticky+resource",
            lambda: NArmRWDualAlphaStickyResource(4),
            ["rl_alpha_pos", "rl_alpha_neg", "sticky", "sticky_gain",
             "fatigue_rate", "surprise_gain", "beta"],
        ),
        # ── Race-model variant (decision module: race_no_bias_angle_4) ──
        (
            "RW+Race",
            "RW+race",
            lambda: NArmRWDriftLearner(4),
            ["rl_alpha", "scaler", "a", "z", "t", "theta"],
            "race_no_bias_angle_4",
            ["rt", "response"],
        ),
    ]

    models: dict[str, dict[str, Any]] = {}
    for entry in definitions:
        name, description, factory, params = entry[:4]
        decision = entry[4] if len(entry) > 4 else "inv_temp_softmax_4"
        response = entry[5] if len(entry) > 5 else None
        learner = factory()
        config = _make_config(description, learner, env, params, decision=decision, response=response)
        actual_bounds = {param: tuple(config.bounds[param]) for param in params}
        requested_bounds = {param: PRIOR_SPECS[param][:2] for param in params}
        if actual_bounds != requested_bounds:
            raise RuntimeError(
                f"Bound mismatch for {name}: {actual_bounds} != {requested_bounds}"
            )
        models[name] = {
            "config": config,
            "params": params,
            "bounds": actual_bounds,
            "learner_factory": factory,
            "include": [hierarchical_param(param) for param in params],
            "is_race": decision != "inv_temp_softmax_4",
        }
    return models


def _stack_posterior(idata: Any) -> Any:
    posterior = idata.posterior
    if hasattr(posterior, "to_dataset"):
        posterior = posterior.to_dataset()
    return posterior.stack(sample=("chain", "draw"))


def population_draw(
    posterior: Any,
    params: list[str],
    bounds: dict[str, tuple[float, float]],
    draw_index: int,
) -> dict[str, float]:
    """Extract one fixed/population draw on each parameter's natural scale."""
    result: dict[str, float] = {}
    for name in params:
        eta = float(
            np.asarray(
                posterior[f"{name}_Intercept"].isel(sample=draw_index).values
            ).squeeze()
        )
        result[name] = float(inverse_gen_logit(eta, bounds[name]))
    return result


def heldout_population_nll(
    idata: Any,
    data: pd.DataFrame,
    spec: dict[str, Any],
) -> np.ndarray:
    """Compute held-out one-step NLL using fixed population parameters only."""
    posterior = _stack_posterior(idata)
    rng = np.random.default_rng(SEED + 1)
    draw_indices = rng.choice(
        posterior.sizes["sample"],
        size=min(N_NLL_DRAWS, posterior.sizes["sample"]),
        replace=False,
    )
    trajectories = [
        trajectory.sort_values("trial_id")
        for _, trajectory in data.groupby("participant_id", sort=True)
    ]
    choices = jnp.asarray(
        np.stack([trajectory["response"].to_numpy() for trajectory in trajectories]),
        dtype=jnp.int32,
    )
    feedback = jnp.asarray(
        np.stack([trajectory["feedback"].to_numpy() for trajectory in trajectories])
    )
    theta_draws = [
        population_draw(posterior, spec["params"], spec["bounds"], int(index))
        for index in draw_indices
    ]
    batched_theta = {
        name: jnp.asarray([theta[name] for theta in theta_draws])
        for name in spec["params"]
    }
    learner = spec["learner_factory"]()

    def score_draw(theta):
        def score_trajectory(trajectory_choices, trajectory_feedback):
            def score_trial(state, observation):
                choice, reward = observation
                computed = learner.compute_jax(state, theta, context={})
                values = jnp.stack([computed[f"q{i}"] for i in range(4)])
                log_probability = jax.nn.log_softmax(theta["beta"] * values)[choice]
                updated_state = learner.update_jax(
                    state,
                    theta,
                    context={"choice": choice, "feedback": reward},
                )
                return updated_state, log_probability

            _, log_probabilities = jax.lax.scan(
                score_trial,
                learner.init_jax_state(),
                (trajectory_choices, trajectory_feedback),
            )
            return log_probabilities

        return jax.vmap(score_trajectory)(choices, feedback)

    log_probs = np.asarray(jax.jit(jax.vmap(score_draw))(batched_theta))
    log_probs = log_probs.reshape(log_probs.shape[0], -1)
    return -logsumexp(log_probs, axis=0) + np.log(log_probs.shape[0])


def heldout_race_nll(
    model: Any,
    idata: Any,
    data: pd.DataFrame,
    spec: dict[str, Any],
) -> np.ndarray:
    """Compute held-out NLL for race models via HSSM's log_likelihood.

    Race models use an approx_differentiable (neural-network) likelihood that
    cannot be computed manually like softmax.  This function delegates to
    HSSM's ``model.log_likelihood`` which has access to the trained network.

    The posterior is used as-is (full hierarchical draws).  For held-out
    participants whose random effects are not in the posterior, HSSM falls
    back to population-level intercepts — matching the population-only scoring
    used for softmax models.
    """
    # Offset participant IDs so held-out subjects don't collide with training
    # subjects in the posterior's random-effects arrays.
    offset_data = data.copy()
    offset = 100_000
    offset_data["participant_id"] = offset_data["participant_id"] + offset

    dt = model.log_likelihood(dt=idata, data=offset_data, inplace=False)

    # Extract per-draw, per-trial log-likelihood from the DataTree.
    ll_group = dt["log_likelihood"]
    if hasattr(ll_group, "to_dataset"):
        ll_group = ll_group.to_dataset()
    # The response variable name is typically "response" in the log_likelihood group.
    ll_var = list(ll_group.data_vars)[0]
    ll = np.asarray(ll_group[ll_var].values)  # shape: (chain, draw, obs)
    ll = ll.reshape(-1, ll.shape[-1])          # (n_draws, n_trials)

    # Log-mean-exp NLL: -log(mean_d exp(ll_d)) per trial
    return -logsumexp(ll, axis=0) + np.log(ll.shape[0])


def population_parameter_summary(
    idata: Any,
    params: list[str],
    bounds: dict[str, tuple[float, float]],
) -> dict[str, dict[str, float]]:
    """Summarize group intercepts after converting them to natural scale."""
    posterior = _stack_posterior(idata)
    result: dict[str, dict[str, float]] = {}
    for name in params:
        eta = np.asarray(posterior[f"{name}_Intercept"].values, dtype=float)
        natural = np.asarray(inverse_gen_logit(eta, bounds[name]), dtype=float)
        result[name] = {
            "mean": float(np.mean(natural)),
            "sd": float(np.std(natural, ddof=1)),
            "q03": float(np.quantile(natural, 0.03)),
            "q97": float(np.quantile(natural, 0.97)),
        }
    return result


def sampler_diagnostics(idata: Any) -> dict[str, Any]:
    """Return population-level convergence checks and global HMC checks."""
    posterior = idata.posterior
    if hasattr(posterior, "to_dataset"):
        posterior = posterior.to_dataset()
    variables = [
        name
        for name in posterior.data_vars
        if name.endswith("_Intercept") or name.endswith("_sigma")
    ]
    if not variables:
        variables = list(posterior.data_vars)

    rhat_values = np.asarray(
        az.rhat(posterior, var_names=variables, method="rank").to_array(),
        dtype=float,
    )
    ess_values = np.asarray(
        az.ess(posterior, var_names=variables, method="bulk").to_array(),
        dtype=float,
    )
    finite_rhat = rhat_values[np.isfinite(rhat_values)]
    finite_ess = ess_values[np.isfinite(ess_values)]
    max_rhat = float(np.max(finite_rhat)) if finite_rhat.size else None
    min_bulk_ess = float(np.min(finite_ess)) if finite_ess.size else None

    sample_stats = idata.sample_stats
    if hasattr(sample_stats, "to_dataset"):
        sample_stats = sample_stats.to_dataset()
    divergences = (
        int(np.asarray(sample_stats["diverging"]).sum())
        if "diverging" in sample_stats
        else 0
    )
    try:
        energy = np.atleast_2d(np.asarray(sample_stats["energy"], dtype=float))
        bfmi_values = np.mean(np.diff(energy, axis=1) ** 2, axis=1) / np.var(
            energy, axis=1, ddof=1
        )
        finite_bfmi = bfmi_values[np.isfinite(bfmi_values)]
        min_bfmi = float(np.min(finite_bfmi)) if finite_bfmi.size else None
    except (KeyError, TypeError, ValueError):
        min_bfmi = None

    failures: list[str] = []
    if divergences:
        failures.append(f"{divergences} divergences")
    if max_rhat is None:
        failures.append("R-hat unavailable")
    elif max_rhat > MAX_RHAT:
        failures.append(f"max R-hat {max_rhat:.3f} > {MAX_RHAT:.2f}")
    if min_bulk_ess is None:
        failures.append("bulk ESS unavailable")
    elif min_bulk_ess < MIN_BULK_ESS:
        failures.append(f"min bulk ESS {min_bulk_ess:.0f} < {MIN_BULK_ESS}")
    if min_bfmi is not None and min_bfmi < MIN_BFMI:
        failures.append(f"min BFMI {min_bfmi:.3f} < {MIN_BFMI:.2f}")

    return {
        "pass": not failures,
        "failures": failures,
        "divergences": divergences,
        "max_rhat": max_rhat,
        "min_bulk_ess": min_bulk_ess,
        "min_bfmi": min_bfmi,
        "checked_variables": variables,
    }


def fit_model(name: str, spec: dict[str, Any], data: pd.DataFrame) -> tuple[Any, Any]:
    """Fit a model and return (model, idata).

    The model object is retained so race-model NLL scoring can use
    HSSM's ``log_likelihood`` method with the neural-network likelihood.
    """
    model_config = hssm.rl.RLSSMConfig.from_ssms_model(spec["config"])
    model = hssm.RLSSM(
        data=data,
        model_config=model_config,
        p_outlier=0,
        lapse=None,
        process_initvals=True,
        include=spec["include"],
    )
    target_accept = 0.99 if name.startswith("HGF") else 0.97
    idata = model.sample(
        sampler=SAMPLER,
        draws=N_DRAWS,
        tune=N_TUNE,
        chains=N_CHAINS,
        cores=4,
        target_accept=target_accept,
        random_seed=SEED,
        progressbar=False,
        idata_kwargs={"log_likelihood": False},
    )
    return model, idata


def main() -> None:
    print("=" * 72)
    print(f"Corrected 7-model comparison (FULL_RUN={FULL_RUN}, sampler={SAMPLER})")
    print(
        f"train={N_TRAIN}, validation={N_VALID}, chains={N_CHAINS}, "
        f"tune={N_TUNE}, draws={N_DRAWS}, scoring draws={N_NLL_DRAWS}"
    )
    print(
        "Validation uses disjoint subjects and population-only parameters; "
        "all bounded hierarchies use generalized-logit links."
    )
    print("=" * 72)

    train_data, valid_data, split_info = load_split_data(
        include_rt=any(spec.get("is_race", False) for spec in build_models().values())
    )
    models = build_models()
    results: dict[str, dict[str, Any]] = {}

    for name, spec in models.items():
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
            results[name] = {
                "status": "ok",
                "eligible": diagnostics["pass"],
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
        except Exception as error:  # continue so one failed model does not hide others
            print(f"FAILED: {type(error).__name__}: {error}")
            results[name] = {
                "status": "fit_failed",
                "eligible": False,
                "error": f"{type(error).__name__}: {error}",
            }

    print(f"\n{'=' * 72}\nHeld-out population-level NLL (lower is better)")
    print(f"{'Model':<20} {'Mean':>9} {'SE':>9} {'Diagnostics':>14}")
    print(f"{'-' * 20} {'-' * 9} {'-' * 9} {'-' * 14}")
    for name, result in results.items():
        if result["status"] != "ok":
            print(f"{name:<20} {'--':>9} {'--':>9} {'FIT FAILED':>14}")
            continue
        diag_status = "PASS" if result["eligible"] else "FAIL"
        print(
            f"{name:<20} {result['heldout_nll_mean']:>9.4f} "
            f"{result['heldout_nll_se']:>9.5f} {diag_status:>14}"
        )
    print(f"{'Uniform random':<20} {np.log(4):>9.4f}")

    eligible = {
        name: result
        for name, result in results.items()
        if result.get("eligible", False)
    }
    best_model = (
        min(eligible, key=lambda name: eligible[name]["heldout_nll_mean"])
        if eligible
        else None
    )
    if best_model is None:
        print("\nNo model passed all diagnostics; no winner is declared.")
    else:
        best = results[best_model]
        print(
            f"\nBest eligible model: {best_model} "
            f"(held-out NLL {best['heldout_nll_mean']:.4f} ± "
            f"{best['heldout_nll_se']:.5f} SE)"
        )
        print("Population parameters:")
        for param, summary in best["population_parameters"].items():
            print(
                f"  {param}: {summary['mean']:.4f} "
                f"[{summary['q03']:.4f}, {summary['q97']:.4f}]"
            )

    output = {
        "run_type": "production" if FULL_RUN else "diagnostic",
        "seed": SEED,
        "floatX": FLOATX,
        "sampler": SAMPLER,
        "chains": N_CHAINS,
        "tune": N_TUNE,
        "draws": N_DRAWS,
        "posterior_draws_for_scoring": N_NLL_DRAWS,
        "n_trials_per_trajectory": N_TRIALS,
        "split": split_info,
        "evaluation": {
            "data": "held-out trajectories from disjoint human subjects",
            "parameters": "population intercepts only (no participant random effects)",
            "metric": "one-step-ahead posterior predictive log-mean-exp NLL",
        },
        "diagnostic_thresholds": {
            "scope": (
                "R-hat/ESS on population intercepts and random-effect scales; "
                "divergences/BFMI global"
            ),
            "max_rhat": MAX_RHAT,
            "min_bulk_ess": MIN_BULK_ESS,
            "min_bfmi": MIN_BFMI,
            "max_divergences": 0,
        },
        "hgf_constants": {
            "obs_precision": OBS_PRECISION,
            "initial_mu1": INITIAL_MU1,
            "initial_mu2": INITIAL_MU2,
            "initial_sigma1": INITIAL_SIGMA1,
            "initial_sigma2": INITIAL_SIGMA2,
            "theta_var": THETA_VAR,
            "volatility_update": "unbounded HGF two-expansion update",
        },
        "best_model": best_model,
        "results": results,
    }
    output_path = (
        Path(__file__).resolve().parent
        / "model_comparison_results.json"
    )
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, allow_nan=False)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
