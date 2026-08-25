"""Causally reward-standardized dual-alpha reinforcement learning.

This learner is designed for evaluation tasks whose reward units and offset are
unknown.  It keeps Rescorla-Wagner action values in the raw reward units and
uses only rewards already revealed in the current trajectory to express those
values in running standard-deviation units for choice.

The resulting choice logits are invariant to every positive affine reward
transformation (``reward -> scale * reward + offset``).  Running statistics are
part of the trajectory-local state; they are reset between trajectories and are
not fitted during evaluation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import jax.numpy as jnp
except ImportError:  # pragma: no cover
    jnp = None


class NArmCausalScaleDualAlphaSticky:
    """Dual-alpha RW with causal reward scaling and independent repetition.

    Raw Q-values follow the usual delta rule.  Before a decision, they are
    centered by the running reward mean and divided by the running sample
    standard deviation.  An arm that has not yet been selected uses the current
    reward mean as its neutral prior.  Until two non-identical rewards have been
    revealed, the value contribution is zero because no reward scale can yet be
    inferred causally.

    The learner emits categorical logits directly:

    ``beta * standardized_q + repetition_weight * previous_choice_indicator``

    Keeping the repetition coefficient outside the value scaling prevents
    reward units from changing the strength of perseveration.
    """

    def __init__(
        self,
        n_actions: int = 4,
        feedback_field: str = "feedback",
    ) -> None:
        if n_actions < 2:
            raise ValueError("n_actions must be at least 2")
        self._n_actions = n_actions
        self._feedback_field = feedback_field
        self._state: dict[str, Any] | None = None

    @property
    def n_actions(self) -> int:
        return self._n_actions

    @property
    def computed_params(self) -> list[str]:
        return [f"logit{i}" for i in range(self._n_actions)]

    @property
    def free_params(self) -> list[str]:
        return [
            "rl_alpha_pos",
            "rl_alpha_neg",
            "beta",
            "repetition_weight",
        ]

    @property
    def param_bounds(self) -> dict[str, tuple[float, float]]:
        return {
            "rl_alpha_pos": (0.0, 1.0),
            "rl_alpha_neg": (0.0, 1.0),
            "beta": (0.0, 15.0),
            "repetition_weight": (-5.0, 5.0),
        }

    @property
    def default_params(self) -> dict[str, float]:
        return {
            "rl_alpha_pos": 0.3,
            "rl_alpha_neg": 0.3,
            "beta": 5.0,
            "repetition_weight": 0.0,
        }

    @property
    def available_backends(self) -> tuple[str, ...]:
        return ("python", "jax")

    @property
    def supports_gradient(self) -> bool:
        return True

    @property
    def required_context_fields(self) -> list[str]:
        return ["choice", self._feedback_field]

    def init_state(self) -> dict[str, Any]:
        return {
            "q_values": np.zeros(self._n_actions, dtype=np.float64),
            "action_seen": np.zeros(self._n_actions, dtype=bool),
            "reward_count": 0,
            "reward_mean": 0.0,
            "reward_m2": 0.0,
            "last_choice": -1,
        }

    def init_jax_state(self) -> dict[str, Any]:
        if jnp is None:  # pragma: no cover
            raise RuntimeError("JAX is required for init_jax_state()")
        return {
            "q_values": jnp.zeros((self._n_actions,)),
            "action_seen": jnp.zeros((self._n_actions,), dtype=bool),
            "reward_count": jnp.asarray(0, dtype=jnp.int32),
            "reward_mean": jnp.asarray(0.0),
            "reward_m2": jnp.asarray(0.0),
            "last_choice": jnp.asarray(-1, dtype=jnp.int32),
        }

    def reset(self, **kwargs: Any) -> None:
        del kwargs
        self._state = self.init_state()

    def _standardized_python(self, state: dict[str, Any]) -> np.ndarray:
        q = np.asarray(state["q_values"], dtype=np.float64)
        seen = np.asarray(state["action_seen"], dtype=bool)
        mean = float(state["reward_mean"])
        count = int(state["reward_count"])
        m2 = max(float(state["reward_m2"]), 0.0)
        effective_q = np.where(seen, q, mean)
        if count < 2 or m2 <= 0.0:
            return np.zeros(self._n_actions, dtype=np.float64)
        sample_sd = np.sqrt(m2 / (count - 1))
        if not np.isfinite(sample_sd) or sample_sd <= 0.0:
            return np.zeros(self._n_actions, dtype=np.float64)
        return (effective_q - mean) / sample_sd

    def _standardized_jax(self, state: dict[str, Any]) -> Any:
        q = state["q_values"]
        seen = state["action_seen"]
        mean = state["reward_mean"]
        count = state["reward_count"]
        m2 = jnp.maximum(state["reward_m2"], 0.0)
        effective_q = jnp.where(seen, q, mean)
        has_scale = (count >= 2) & (m2 > 0.0)
        denominator = jnp.maximum(count - 1, 1)
        sample_sd = jnp.sqrt(m2 / denominator)
        safe_sd = jnp.where(has_scale, sample_sd, 1.0)
        return jnp.where(has_scale, (effective_q - mean) / safe_sd, 0.0)

    def compute_python(self, state, params, context):
        del context
        standardized = self._standardized_python(state)
        logits = float(params["beta"]) * standardized
        last_choice = int(state["last_choice"])
        if last_choice >= 0:
            logits[last_choice] += float(params["repetition_weight"])
        return {
            f"logit{action}": float(logits[action])
            for action in range(self._n_actions)
        }

    def compute_jax(self, state, params, context):
        del context
        standardized = self._standardized_jax(state)
        actions = jnp.arange(self._n_actions)
        logits = (
            jnp.asarray(params["beta"]) * standardized
            + jnp.where(
                actions == state["last_choice"],
                jnp.asarray(params["repetition_weight"]),
                0.0,
            )
        )
        return {
            f"logit{action}": logits[action]
            for action in range(self._n_actions)
        }

    def update_python(self, state, params, context):
        choice = int(context["choice"])
        reward = float(context[self._feedback_field])
        q = np.asarray(state["q_values"], dtype=np.float64).copy()
        seen = np.asarray(state["action_seen"], dtype=bool).copy()
        count = int(state["reward_count"])
        mean = float(state["reward_mean"])
        m2 = float(state["reward_m2"])

        if seen[choice]:
            prior = q[choice]
        elif count > 0:
            prior = mean
        else:
            # One observation cannot identify location or scale.  Treat the
            # first outcome as the neutral anchor for every action.
            prior = reward

        prediction_error = reward - prior
        alpha = (
            float(params["rl_alpha_pos"])
            if prediction_error >= 0.0
            else float(params["rl_alpha_neg"])
        )
        q[choice] = prior + alpha * prediction_error
        seen[choice] = True

        new_count = count + 1
        delta = reward - mean
        new_mean = mean + delta / new_count
        new_m2 = m2 + delta * (reward - new_mean)
        return {
            "q_values": q,
            "action_seen": seen,
            "reward_count": new_count,
            "reward_mean": new_mean,
            "reward_m2": max(new_m2, 0.0),
            "last_choice": choice,
        }

    def update_jax(self, state, params, context):
        choice = jnp.asarray(context["choice"], dtype=jnp.int32)
        reward = jnp.asarray(context[self._feedback_field])
        q = state["q_values"]
        seen = state["action_seen"]
        count = state["reward_count"]
        mean = state["reward_mean"]
        m2 = state["reward_m2"]

        prior = jnp.where(
            seen[choice],
            q[choice],
            jnp.where(count > 0, mean, reward),
        )
        prediction_error = reward - prior
        alpha = jnp.where(
            prediction_error >= 0.0,
            jnp.asarray(params["rl_alpha_pos"]),
            jnp.asarray(params["rl_alpha_neg"]),
        )
        new_q = q.at[choice].set(prior + alpha * prediction_error)
        new_seen = seen.at[choice].set(True)

        new_count = count + 1
        delta = reward - mean
        new_mean = mean + delta / new_count
        new_m2 = jnp.maximum(m2 + delta * (reward - new_mean), 0.0)
        return {
            "q_values": new_q,
            "action_seen": new_seen,
            "reward_count": new_count,
            "reward_mean": new_mean,
            "reward_m2": new_m2,
            "last_choice": choice,
        }

    def compute_ssm_params(self, trial_params):
        if self._state is None:
            raise RuntimeError("Call reset() before compute_ssm_params()")
        return self.compute_python(self._state, trial_params, context={})

    def update(self, action, reward, trial_params):
        if self._state is None:
            raise RuntimeError("Call reset() before update()")
        self._state = self.update_python(
            self._state,
            trial_params,
            context={"choice": action, self._feedback_field: reward},
        )


class NArmCausalScaleSingleAlphaSticky(NArmCausalScaleDualAlphaSticky):
    """Single-alpha counterpart of the immediate-repetition learner."""

    @property
    def free_params(self) -> list[str]:
        return ["rl_alpha", "beta", "repetition_weight"]

    @property
    def param_bounds(self) -> dict[str, tuple[float, float]]:
        return {
            "rl_alpha": (0.0, 1.0),
            "beta": (0.0, 15.0),
            "repetition_weight": (-5.0, 5.0),
        }

    @property
    def default_params(self) -> dict[str, float]:
        return {
            "rl_alpha": 0.3,
            "beta": 5.0,
            "repetition_weight": 0.0,
        }

    @staticmethod
    def _dual_params(params: dict[str, Any]) -> dict[str, Any]:
        return {
            **params,
            "rl_alpha_pos": params["rl_alpha"],
            "rl_alpha_neg": params["rl_alpha"],
        }

    def update_python(self, state, params, context):
        return super().update_python(state, self._dual_params(params), context)

    def update_jax(self, state, params, context):
        return super().update_jax(state, self._dual_params(params), context)


class NArmCausalScaleDualAlphaNoHistory(NArmCausalScaleDualAlphaSticky):
    """Dual-alpha learner without an outcome-independent choice-history term."""

    @property
    def free_params(self) -> list[str]:
        return ["rl_alpha_pos", "rl_alpha_neg", "beta"]

    @property
    def param_bounds(self) -> dict[str, tuple[float, float]]:
        return {
            "rl_alpha_pos": (0.0, 1.0),
            "rl_alpha_neg": (0.0, 1.0),
            "beta": (0.0, 15.0),
        }

    @property
    def default_params(self) -> dict[str, float]:
        return {
            "rl_alpha_pos": 0.3,
            "rl_alpha_neg": 0.3,
            "beta": 5.0,
        }

    @staticmethod
    def _no_history_params(params: dict[str, Any]) -> dict[str, Any]:
        return {**params, "repetition_weight": 0.0}

    def compute_python(self, state, params, context):
        return super().compute_python(
            state, self._no_history_params(params), context
        )

    def compute_jax(self, state, params, context):
        return super().compute_jax(state, self._no_history_params(params), context)


class NArmCausalScaleSingleAlphaNoHistory(NArmCausalScaleSingleAlphaSticky):
    """Single-alpha learner without an outcome-independent history term."""

    @property
    def free_params(self) -> list[str]:
        return ["rl_alpha", "beta"]

    @property
    def param_bounds(self) -> dict[str, tuple[float, float]]:
        return {"rl_alpha": (0.0, 1.0), "beta": (0.0, 15.0)}

    @property
    def default_params(self) -> dict[str, float]:
        return {"rl_alpha": 0.3, "beta": 5.0}

    @staticmethod
    def _no_history_params(params: dict[str, Any]) -> dict[str, Any]:
        return {**params, "repetition_weight": 0.0}

    def compute_python(self, state, params, context):
        return super().compute_python(
            state, self._no_history_params(params), context
        )

    def compute_jax(self, state, params, context):
        return super().compute_jax(state, self._no_history_params(params), context)


class NArmCausalScaleDualAlphaChoiceTrace(NArmCausalScaleDualAlphaSticky):
    """Dual-alpha learner with a gradual, outcome-independent choice trace.

    A trace is maintained for every action.  After a choice it moves toward the
    one-hot vector for that choice at ``choice_trace_rate``.  The next decision
    receives ``repetition_weight * trace`` as a separate logit contribution.
    Thus ``choice_trace_rate=1`` recovers the immediate previous-choice model,
    while smaller values allow several recent choices to influence behavior.

    This is the gradual-perseveration construction studied by Sugawara and
    Katahira (2021, Scientific Reports, doi:10.1038/s41598-020-80593-7).
    """

    @property
    def free_params(self) -> list[str]:
        return [
            "rl_alpha_pos",
            "rl_alpha_neg",
            "beta",
            "repetition_weight",
            "choice_trace_rate",
        ]

    @property
    def param_bounds(self) -> dict[str, tuple[float, float]]:
        return {
            **super().param_bounds,
            "choice_trace_rate": (0.0, 1.0),
        }

    @property
    def default_params(self) -> dict[str, float]:
        return {**super().default_params, "choice_trace_rate": 0.5}

    def init_state(self) -> dict[str, Any]:
        return {
            **super().init_state(),
            "choice_trace": np.zeros(self._n_actions, dtype=np.float64),
        }

    def init_jax_state(self) -> dict[str, Any]:
        return {
            **super().init_jax_state(),
            "choice_trace": jnp.zeros((self._n_actions,)),
        }

    def compute_python(self, state, params, context):
        del context
        standardized = self._standardized_python(state)
        logits = (
            float(params["beta"]) * standardized
            + float(params["repetition_weight"])
            * np.asarray(state["choice_trace"], dtype=np.float64)
        )
        return {
            f"logit{action}": float(logits[action])
            for action in range(self._n_actions)
        }

    def compute_jax(self, state, params, context):
        del context
        logits = (
            jnp.asarray(params["beta"]) * self._standardized_jax(state)
            + jnp.asarray(params["repetition_weight"]) * state["choice_trace"]
        )
        return {
            f"logit{action}": logits[action]
            for action in range(self._n_actions)
        }

    def update_python(self, state, params, context):
        updated = super().update_python(state, params, context)
        choice = int(context["choice"])
        rate = float(params["choice_trace_rate"])
        target = np.zeros(self._n_actions, dtype=np.float64)
        target[choice] = 1.0
        trace = np.asarray(state["choice_trace"], dtype=np.float64)
        updated["choice_trace"] = trace + rate * (target - trace)
        return updated

    def update_jax(self, state, params, context):
        updated = super().update_jax(state, params, context)
        choice = jnp.asarray(context["choice"], dtype=jnp.int32)
        rate = jnp.asarray(params["choice_trace_rate"])
        target = jax_one_hot(choice, self._n_actions, state["choice_trace"].dtype)
        updated["choice_trace"] = (
            state["choice_trace"] + rate * (target - state["choice_trace"])
        )
        return updated


class NArmCausalScaleSingleAlphaChoiceTrace(
    NArmCausalScaleDualAlphaChoiceTrace
):
    """Single-alpha counterpart of the gradual choice-trace learner."""

    @property
    def free_params(self) -> list[str]:
        return [
            "rl_alpha",
            "beta",
            "repetition_weight",
            "choice_trace_rate",
        ]

    @property
    def param_bounds(self) -> dict[str, tuple[float, float]]:
        return {
            "rl_alpha": (0.0, 1.0),
            "beta": (0.0, 15.0),
            "repetition_weight": (-5.0, 5.0),
            "choice_trace_rate": (0.0, 1.0),
        }

    @property
    def default_params(self) -> dict[str, float]:
        return {
            "rl_alpha": 0.3,
            "beta": 5.0,
            "repetition_weight": 0.0,
            "choice_trace_rate": 0.5,
        }

    @staticmethod
    def _dual_params(params: dict[str, Any]) -> dict[str, Any]:
        return {
            **params,
            "rl_alpha_pos": params["rl_alpha"],
            "rl_alpha_neg": params["rl_alpha"],
        }

    def update_python(self, state, params, context):
        return super().update_python(state, self._dual_params(params), context)

    def update_jax(self, state, params, context):
        return super().update_jax(state, self._dual_params(params), context)


def jax_one_hot(index: Any, size: int, dtype: Any) -> Any:
    """Small one-hot helper that keeps this module independent of jax.nn."""
    return (jnp.arange(size) == index).astype(dtype)
