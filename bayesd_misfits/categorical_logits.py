"""Direct categorical-logit decision process for uncertainty-aware RL models."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp


MODEL_NAME = "categorical_logits_4"
_REGISTERED = False


def categorical_logits_simulator(
    *,
    logit0,
    logit1,
    logit2,
    logit3,
    n_samples: int = 1000,
    n_trials: int = 1,
    max_t: float = 20.0,
    random_state: int | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Sample choices directly from four supplied categorical logits."""
    del kwargs

    def broadcast(value: Any, name: str) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32).squeeze()
        if array.ndim == 0:
            return np.full(n_trials, float(array), dtype=np.float32)
        if array.ndim == 1 and len(array) == n_trials:
            return array
        raise ValueError(
            f"{name} must be scalar or have n_trials={n_trials} values; "
            f"got shape {array.shape}"
        )

    logits = np.column_stack([
        broadcast(logit0, "logit0"),
        broadcast(logit1, "logit1"),
        broadcast(logit2, "logit2"),
        broadcast(logit3, "logit3"),
    ])
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)

    rng = np.random.default_rng(random_state)
    choices = np.empty((n_samples, n_trials, 1), dtype=np.int64)
    labels = np.arange(4, dtype=np.int64)
    for trial in range(n_trials):
        choices[:, trial, 0] = rng.choice(
            labels,
            size=n_samples,
            p=probabilities[trial],
        )
    return {
        "rts": np.full((n_samples, n_trials, 1), -1.0, dtype=np.float32),
        "choices": choices,
        "metadata": {
            "possible_choices": labels.tolist(),
            "n_choices": 4,
            "n_samples": n_samples,
            "max_t": max_t,
            "placeholder_rt": -1.0,
        },
    }


def register_categorical_logits_4() -> str:
    """Register the custom direct-logit process with ssms and HSSM once."""
    global _REGISTERED
    if _REGISTERED:
        return MODEL_NAME

    from hssm.rl import register_ssm
    from hssm.utils import annotate_function
    from ssms.config import (
        ModelConfigBuilder,
        get_model_registry,
        register_model_config,
    )

    logit_names = [f"logit{i}" for i in range(4)]
    if not get_model_registry().has_model(MODEL_NAME):
        config = ModelConfigBuilder.from_scratch(
            name=MODEL_NAME,
            params=logit_names,
            simulator_function=categorical_logits_simulator,
            nchoices=4,
            param_bounds=[[-100.0] * 4, [100.0] * 4],
            default_params=[0.0] * 4,
            choices=[0, 1, 2, 3],
        )
        config["param_bounds_dict"] = {
            name: (-100.0, 100.0) for name in logit_names
        }
        config["tags"] = ["choice_only_rl"]
        register_model_config(MODEL_NAME, config)

    @annotate_function(inputs=[*logit_names, "response"], outputs=["logp"])
    def categorical_logits_logp(lan_matrix):
        matrix = jnp.asarray(lan_matrix)
        logits = matrix[:, :4]
        response = matrix[:, 4]
        response_is_finite = jnp.isfinite(response)
        response_is_integral = response == jnp.floor(response)
        response_int = jnp.where(response_is_finite, response, 0.0).astype(jnp.int32)
        valid_response = (
            response_is_finite
            & response_is_integral
            & (response_int >= 0)
            & (response_int < 4)
        )
        safe_response = jnp.clip(response_int, 0, 3)
        chosen_logits = jnp.take_along_axis(
            logits, safe_response[:, None], axis=1
        ).squeeze(axis=1)
        logp = chosen_logits - logsumexp(logits, axis=1)
        return jnp.where(valid_response, logp, -jnp.inf)

    register_ssm(
        name=MODEL_NAME,
        ssm_base_logp_func=categorical_logits_logp,
        list_params_ssm=logit_names,
        bounds_ssm={},
        params_default_ssm=[0.0] * 4,
        response=["response"],
    )
    _REGISTERED = True
    return MODEL_NAME
