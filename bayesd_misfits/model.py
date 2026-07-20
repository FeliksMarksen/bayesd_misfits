"""N-arm Rescorla-Wagner learners for HSSM RLSSM.

Implements the ``LearningProcess`` protocol from ``ssms.rl``.
Variants: single-alpha, dual-alpha (pos/neg PE), optional decay, stickiness.
Drift-rate variant outputs v = scaler * Q for race models.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import jax.numpy as jnp
except ImportError:  # pragma: no cover
    jnp = None


class NArmRescorlaWagner:
    """N-arm RW learner with optional forgetting (decay toward initial_q)."""

    def __init__(
        self,
        n_actions: int = 4,
        initial_q: float = 0.5,
        feedback_field: str = "feedback",
        use_decay: bool = False,
    ):
        self._n_actions = n_actions
        self._initial_q = initial_q
        self._feedback_field = feedback_field
        self._use_decay = use_decay
        self._state: dict[str, Any] | None = None

    @property
    def n_actions(self) -> int:
        return self._n_actions

    @property
    def computed_params(self) -> list[str]:
        return [f"q{i}" for i in range(self._n_actions)]

    @property
    def free_params(self) -> list[str]:
        params = ["rl_alpha"]
        if self._use_decay:
            params.append("rl_decay")
        return params

    @property
    def param_bounds(self) -> dict[str, tuple[float, float]]:
        bounds = {"rl_alpha": (0.0, 1.0)}
        if self._use_decay:
            bounds["rl_decay"] = (0.0, 1.0)
        return bounds

    @property
    def default_params(self) -> dict[str, float]:
        defaults = {"rl_alpha": 0.2}
        if self._use_decay:
            defaults["rl_decay"] = 0.0
        return defaults

    @property
    def available_backends(self) -> tuple[str, ...]:
        return ("python", "jax")

    @property
    def supports_gradient(self) -> bool:
        return True

    @property
    def required_context_fields(self) -> list[str]:
        return ["choice", self._feedback_field]

    def init_state(self) -> dict[str, np.ndarray]:
        return {
            "q_values": np.full(self._n_actions, self._initial_q, dtype=np.float64)
        }

    def init_jax_state(self) -> dict[str, Any]:
        return {
            "q_values": jnp.full((self._n_actions,), self._initial_q)
        }

    def reset(self, **kwargs) -> None:
        self._state = self.init_state()

    def compute_python(self, state, params, context):
        q = state["q_values"]
        return {f"q{i}": float(q[i]) for i in range(self._n_actions)}

    def update_python(self, state, params, context):
        choice = int(context["choice"])
        feedback = float(context[self._feedback_field])
        alpha = params["rl_alpha"]
        q = np.asarray(state["q_values"], dtype=np.float64).copy()

        if self._use_decay:
            decay = params.get("rl_decay", 0.0)
            q = (1.0 - decay) * q + decay * self._initial_q

        q[choice] += alpha * (feedback - q[choice])
        return {"q_values": q}

    def compute_jax(self, state, params, context):
        q = state["q_values"]
        return {f"q{i}": q[i] for i in range(self._n_actions)}

    def update_jax(self, state, params, context):
        choice = context["choice"]
        feedback = context[self._feedback_field]
        alpha = params["rl_alpha"]
        q = state["q_values"]

        if self._use_decay:
            decay = params.get("rl_decay", 0.0)
            q = (1.0 - decay) * q + decay * self._initial_q

        delta = feedback - q[choice]
        return {"q_values": q.at[choice].add(alpha * delta)}

    def compute_ssm_params(self, trial_params: dict[str, float]) -> dict[str, float]:
        if self._state is None:
            raise RuntimeError("Call reset() before compute_ssm_params()")
        return self.compute_python(self._state, trial_params, context={})

    def update(
        self, action: int, reward: float, trial_params: dict[str, float]
    ) -> None:
        if self._state is None:
            raise RuntimeError("Call reset() before update()")
        self._state = self.update_python(
            self._state,
            trial_params,
            context={"choice": action, self._feedback_field: reward},
        )


class NArmRWDriftLearner:
    """N-arm RW outputting scaled drift rates (v = scaler * Q) for race models."""

    def __init__(
        self,
        n_actions: int = 4,
        initial_q: float = 0.5,
        feedback_field: str = "feedback",
        use_decay: bool = False,
    ):
        self._n_actions = n_actions
        self._initial_q = initial_q
        self._feedback_field = feedback_field
        self._use_decay = use_decay
        self._state: dict[str, Any] | None = None

    @property
    def computed_params(self) -> list[str]:
        return [f"v{i}" for i in range(self._n_actions)]

    @property
    def free_params(self) -> list[str]:
        params = ["rl_alpha", "scaler"]
        if self._use_decay:
            params.append("rl_decay")
        return params

    @property
    def param_bounds(self) -> dict[str, tuple[float, float]]:
        bounds = {"rl_alpha": (0.0, 1.0), "scaler": (0.001, 10.0)}
        if self._use_decay:
            bounds["rl_decay"] = (0.0, 1.0)
        return bounds

    @property
    def default_params(self) -> dict[str, float]:
        defaults = {"rl_alpha": 0.2, "scaler": 2.0}
        if self._use_decay:
            defaults["rl_decay"] = 0.0
        return defaults

    @property
    def available_backends(self) -> tuple[str, ...]:
        return ("python", "jax")

    @property
    def supports_gradient(self) -> bool:
        return True

    @property
    def required_context_fields(self) -> list[str]:
        return ["choice", self._feedback_field]

    def init_state(self) -> dict[str, np.ndarray]:
        return {"q_values": np.full(self._n_actions, self._initial_q, dtype=np.float64)}

    def init_jax_state(self) -> dict[str, Any]:
        return {"q_values": jnp.full((self._n_actions,), self._initial_q)}

    def reset(self, **kwargs) -> None:
        self._state = self.init_state()

    def compute_python(self, state, params, context):
        q = state["q_values"]
        scaler = params["scaler"]
        return {f"v{i}": float(q[i] * scaler) for i in range(self._n_actions)}

    def compute_jax(self, state, params, context):
        q = state["q_values"]
        scaler = params["scaler"]
        return {f"v{i}": q[i] * scaler for i in range(self._n_actions)}

    def update_python(self, state, params, context):
        choice = int(context["choice"])
        feedback = float(context[self._feedback_field])
        alpha = params["rl_alpha"]
        q = np.asarray(state["q_values"], dtype=np.float64).copy()
        if self._use_decay:
            decay = params.get("rl_decay", 0.0)
            q = (1.0 - decay) * q + decay * self._initial_q
        q[choice] += alpha * (feedback - q[choice])
        return {"q_values": q}

    def update_jax(self, state, params, context):
        choice = context["choice"]
        feedback = context[self._feedback_field]
        alpha = params["rl_alpha"]
        q = state["q_values"]
        if self._use_decay:
            decay = params.get("rl_decay", 0.0)
            q = (1.0 - decay) * q + decay * self._initial_q
        delta = feedback - q[choice]
        return {"q_values": q.at[choice].add(alpha * delta)}

    def compute_ssm_params(self, trial_params: dict[str, float]) -> dict[str, float]:
        if self._state is None:
            raise RuntimeError("Call reset() before compute_ssm_params()")
        return self.compute_python(self._state, trial_params, context={})

    def update(self, action: int, reward: float, trial_params: dict[str, float]) -> None:
        if self._state is None:
            raise RuntimeError("Call reset() before update()")
        self._state = self.update_python(
            self._state, trial_params,
            context={"choice": action, self._feedback_field: reward},
        )


class NArmDualAlphaRW:
    """N-arm RW with separate learning rates for positive/negative PEs."""

    def __init__(
        self,
        n_actions: int = 4,
        initial_q: float = 0.5,
        feedback_field: str = "feedback",
        use_decay: bool = False,
    ):
        self._n_actions = n_actions
        self._initial_q = initial_q
        self._feedback_field = feedback_field
        self._use_decay = use_decay
        self._state: dict[str, Any] | None = None

    @property
    def computed_params(self) -> list[str]:
        return [f"q{i}" for i in range(self._n_actions)]

    @property
    def free_params(self) -> list[str]:
        params = ["rl_alpha_pos", "rl_alpha_neg"]
        if self._use_decay:
            params.append("rl_decay")
        return params

    @property
    def param_bounds(self) -> dict[str, tuple[float, float]]:
        bounds = {"rl_alpha_pos": (0.0, 1.0), "rl_alpha_neg": (0.0, 1.0)}
        if self._use_decay:
            bounds["rl_decay"] = (0.0, 1.0)
        return bounds

    @property
    def default_params(self) -> dict[str, float]:
        defaults = {"rl_alpha_pos": 0.3, "rl_alpha_neg": 0.1}
        if self._use_decay:
            defaults["rl_decay"] = 0.0
        return defaults

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
        return {"q_values": np.full(self._n_actions, self._initial_q, dtype=np.float64)}

    def init_jax_state(self) -> dict[str, Any]:
        return {"q_values": jnp.full((self._n_actions,), self._initial_q)}

    def reset(self, **kwargs) -> None:
        self._state = self.init_state()

    def compute_python(self, state, params, context):
        q = state["q_values"]
        return {f"q{i}": float(q[i]) for i in range(self._n_actions)}

    def compute_jax(self, state, params, context):
        q = state["q_values"]
        return {f"q{i}": q[i] for i in range(self._n_actions)}

    def update_python(self, state, params, context):
        choice = int(context["choice"])
        feedback = float(context[self._feedback_field])
        a_pos = params["rl_alpha_pos"]
        a_neg = params["rl_alpha_neg"]
        q = np.asarray(state["q_values"], dtype=np.float64).copy()

        if self._use_decay:
            decay = params.get("rl_decay", 0.0)
            q = (1.0 - decay) * q + decay * self._initial_q

        pe = feedback - q[choice]
        alpha = a_pos if pe >= 0 else a_neg
        q[choice] += alpha * pe
        return {"q_values": q}

    def update_jax(self, state, params, context):
        choice = context["choice"]
        feedback = context[self._feedback_field]
        a_pos = params["rl_alpha_pos"]
        a_neg = params["rl_alpha_neg"]
        q = state["q_values"]

        if self._use_decay:
            decay = params.get("rl_decay", 0.0)
            q = (1.0 - decay) * q + decay * self._initial_q

        pe = feedback - q[choice]
        alpha = jnp.where(pe >= 0, a_pos, a_neg)
        return {"q_values": q.at[choice].add(alpha * pe)}

    def compute_ssm_params(self, trial_params):
        if self._state is None:
            raise RuntimeError("Call reset() before compute_ssm_params()")
        return self.compute_python(self._state, trial_params, context={})

    def update(self, action, reward, trial_params):
        if self._state is None:
            raise RuntimeError("Call reset() before update()")
        self._state = self.update_python(
            self._state, trial_params,
            context={"choice": action, self._feedback_field: reward},
        )


class NArmRWDualAlphaSticky:
    """N-arm dual-alpha RW with choice stickiness (perseveration).

    Adds a ``sticky`` bonus to the last-chosen arm's Q-value before softmax.
    The sticky parameter is applied inside the learner so the decision
    process sees biased Q-values directly.
    """

    def __init__(
        self,
        n_actions: int = 4,
        initial_q: float = 0.5,
        feedback_field: str = "feedback",
    ):
        self._n_actions = n_actions
        self._initial_q = initial_q
        self._feedback_field = feedback_field
        self._state: dict[str, Any] | None = None

    @property
    def computed_params(self) -> list[str]:
        return [f"q{i}" for i in range(self._n_actions)]

    @property
    def free_params(self) -> list[str]:
        return ["rl_alpha_pos", "rl_alpha_neg", "sticky"]

    @property
    def param_bounds(self) -> dict[str, tuple[float, float]]:
        return {
            "rl_alpha_pos": (0.0, 1.0),
            "rl_alpha_neg": (0.0, 1.0),
            "sticky": (-5.0, 5.0),
        }

    @property
    def default_params(self) -> dict[str, float]:
        return {"rl_alpha_pos": 0.3, "rl_alpha_neg": 0.1, "sticky": 0.0}

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
            "q_values": np.full(self._n_actions, self._initial_q, dtype=np.float64),
            "last_choice": -1,
        }

    def init_jax_state(self) -> dict[str, Any]:
        return {
            "q_values": jnp.full((self._n_actions,), self._initial_q),
            "last_choice": -1,
        }

    def reset(self, **kwargs) -> None:
        self._state = self.init_state()

    def compute_python(self, state, params, context):
        q = state["q_values"].copy()
        sticky = params["sticky"]
        last = state["last_choice"]
        if last >= 0:
            q[last] += sticky
        return {f"q{i}": float(q[i]) for i in range(self._n_actions)}

    def compute_jax(self, state, params, context):
        q = state["q_values"]
        sticky = params["sticky"]
        last = state["last_choice"]
        q_biased = jnp.where(
            jnp.arange(self._n_actions) == last, q + sticky, q,
        )
        return {f"q{i}": q_biased[i] for i in range(self._n_actions)}

    def update_python(self, state, params, context):
        choice = int(context["choice"])
        feedback = float(context[self._feedback_field])
        a_pos = params["rl_alpha_pos"]
        a_neg = params["rl_alpha_neg"]
        q = np.asarray(state["q_values"], dtype=np.float64).copy()

        pe = feedback - q[choice]
        alpha = a_pos if pe >= 0 else a_neg
        q[choice] += alpha * pe
        return {"q_values": q, "last_choice": choice}

    def update_jax(self, state, params, context):
        choice = context["choice"]
        feedback = context[self._feedback_field]
        a_pos = params["rl_alpha_pos"]
        a_neg = params["rl_alpha_neg"]
        q = state["q_values"]

        pe = feedback - q[choice]
        alpha = jnp.where(pe >= 0, a_pos, a_neg)
        new_q = q.at[choice].add(alpha * pe)
        return {"q_values": new_q, "last_choice": choice}

    def compute_ssm_params(self, trial_params):
        if self._state is None:
            raise RuntimeError("Call reset() before compute_ssm_params()")
        return self.compute_python(self._state, trial_params, context={})

    def update(self, action, reward, trial_params):
        if self._state is None:
            raise RuntimeError("Call reset() before update()")
        self._state = self.update_python(
            self._state, trial_params,
            context={"choice": action, self._feedback_field: reward},
        )
