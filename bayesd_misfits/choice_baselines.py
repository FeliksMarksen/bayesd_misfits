"""Reward-free causal choice-history baselines for the MindRL bandit task.

These learners deliberately ignore outcomes.  They are diagnostic controls for
testing whether reinforcement-learning models improve prediction beyond simple
choice autocorrelation.  Every mutable state is trajectory-local and depends
only on choices that have already been observed.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import jax.numpy as jnp
except ImportError:  # pragma: no cover - JAX is available in the fit environment
    jnp = None


class NArmChoiceTrace:
    """Choice-only softmax model with a gradually decaying action trace.

    After each choice, the trace moves toward the corresponding one-hot vector
    at ``choice_trace_rate``.  The next-choice logits are the trace multiplied
    by ``repetition_weight``.  A rate of one recovers a previous-choice-only
    model; smaller rates retain a longer history.
    """

    def __init__(self, n_actions: int = 4) -> None:
        if n_actions < 2:
            raise ValueError("n_actions must be at least 2")
        self._n_actions = n_actions
        self._state: dict[str, Any] | None = None

    @property
    def n_actions(self) -> int:
        return self._n_actions

    @property
    def computed_params(self) -> list[str]:
        return [f"logit{i}" for i in range(self._n_actions)]

    @property
    def free_params(self) -> list[str]:
        return ["repetition_weight", "choice_trace_rate"]

    @property
    def param_bounds(self) -> dict[str, tuple[float, float]]:
        return {
            "repetition_weight": (-5.0, 5.0),
            "choice_trace_rate": (0.0, 1.0),
        }

    @property
    def default_params(self) -> dict[str, float]:
        return {"repetition_weight": 0.0, "choice_trace_rate": 0.5}

    @property
    def available_backends(self) -> tuple[str, ...]:
        return ("python", "jax")

    @property
    def supports_gradient(self) -> bool:
        return True

    @property
    def required_context_fields(self) -> list[str]:
        return ["choice"]

    def init_state(self) -> dict[str, np.ndarray]:
        return {"choice_trace": np.zeros(self._n_actions, dtype=np.float64)}

    def init_jax_state(self) -> dict[str, Any]:
        if jnp is None:  # pragma: no cover
            raise RuntimeError("JAX is required for init_jax_state()")
        return {"choice_trace": jnp.zeros((self._n_actions,))}

    def reset(self, **kwargs: Any) -> None:
        del kwargs
        self._state = self.init_state()

    def compute_python(self, state, params, context):
        del context
        logits = (
            float(params["repetition_weight"])
            * np.asarray(state["choice_trace"], dtype=np.float64)
        )
        return {
            f"logit{action}": float(logits[action])
            for action in range(self._n_actions)
        }

    def compute_jax(self, state, params, context):
        del context
        logits = (
            jnp.asarray(params["repetition_weight"]) * state["choice_trace"]
        )
        return {
            f"logit{action}": logits[action]
            for action in range(self._n_actions)
        }

    def update_python(self, state, params, context):
        choice = int(context["choice"])
        rate = float(params["choice_trace_rate"])
        target = np.zeros(self._n_actions, dtype=np.float64)
        target[choice] = 1.0
        trace = np.asarray(state["choice_trace"], dtype=np.float64)
        return {"choice_trace": trace + rate * (target - trace)}

    def update_jax(self, state, params, context):
        choice = jnp.asarray(context["choice"], dtype=jnp.int32)
        rate = jnp.asarray(params["choice_trace_rate"])
        target = (jnp.arange(self._n_actions) == choice).astype(
            state["choice_trace"].dtype
        )
        return {
            "choice_trace": state["choice_trace"]
            + rate * (target - state["choice_trace"])
        }

    def compute_ssm_params(self, trial_params):
        if self._state is None:
            raise RuntimeError("Call reset() before compute_ssm_params()")
        return self.compute_python(self._state, trial_params, context={})

    def update(self, action, reward, trial_params):
        del reward
        if self._state is None:
            raise RuntimeError("Call reset() before update()")
        self._state = self.update_python(
            self._state, trial_params, context={"choice": action}
        )


class NArmPreviousChoice(NArmChoiceTrace):
    """Choice-only model whose only predictor is the immediately prior choice."""

    @property
    def free_params(self) -> list[str]:
        return ["repetition_weight"]

    @property
    def param_bounds(self) -> dict[str, tuple[float, float]]:
        return {"repetition_weight": (-5.0, 5.0)}

    @property
    def default_params(self) -> dict[str, float]:
        return {"repetition_weight": 0.0}

    @staticmethod
    def _with_rate(params: dict[str, Any]) -> dict[str, Any]:
        return {**params, "choice_trace_rate": 1.0}

    def update_python(self, state, params, context):
        return super().update_python(state, self._with_rate(params), context)

    def update_jax(self, state, params, context):
        return super().update_jax(state, self._with_rate(params), context)


class NArmRunningChoiceFrequency:
    """Causal within-trajectory empirical choice-frequency predictor.

    Each action starts with ``prior_count`` pseudo-observations.  Prediction is
    the normalized count vector, and only the revealed choice increments its
    count.  Rewards and future choices are never used.
    """

    def __init__(self, n_actions: int = 4, prior_count: float = 1.0) -> None:
        if n_actions < 2:
            raise ValueError("n_actions must be at least 2")
        if prior_count <= 0.0:
            raise ValueError("prior_count must be positive")
        self._n_actions = n_actions
        self._prior_count = float(prior_count)
        self._state: dict[str, Any] | None = None

    @property
    def free_params(self) -> list[str]:
        return []

    def init_state(self) -> dict[str, np.ndarray]:
        return {
            "choice_counts": np.full(
                self._n_actions, self._prior_count, dtype=np.float64
            )
        }

    def init_jax_state(self) -> dict[str, Any]:
        if jnp is None:  # pragma: no cover
            raise RuntimeError("JAX is required for init_jax_state()")
        return {
            "choice_counts": jnp.full((self._n_actions,), self._prior_count)
        }

    def compute_python(self, state, params, context):
        del params, context
        logits = np.log(np.asarray(state["choice_counts"], dtype=np.float64))
        return {
            f"logit{action}": float(logits[action])
            for action in range(self._n_actions)
        }

    def compute_jax(self, state, params, context):
        del params, context
        logits = jnp.log(state["choice_counts"])
        return {
            f"logit{action}": logits[action]
            for action in range(self._n_actions)
        }

    def update_python(self, state, params, context):
        del params
        counts = np.asarray(state["choice_counts"], dtype=np.float64).copy()
        counts[int(context["choice"])] += 1.0
        return {"choice_counts": counts}

    def update_jax(self, state, params, context):
        del params
        choice = jnp.asarray(context["choice"], dtype=jnp.int32)
        return {
            "choice_counts": state["choice_counts"].at[choice].add(1.0)
        }

