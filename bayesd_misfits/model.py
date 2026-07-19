"""Custom N-arm Rescorla-Wagner learner for HSSM RLSSM.

Implements the ``LearningProcess`` protocol from ``ssms.rl`` for an arbitrary
number of arms.  Designed for the MindRL Challenge 4-arm drifting bandit but
generalises to any N-arm bandit by changing ``n_actions``.

The learner maintains N Q-values and exposes them as computed parameters
(``q0, q1, ..., qN-1``) that feed into the ``inv_temp_softmax_N`` decision
process registered in HSSM.  The only free parameters are:

- ``rl_alpha`` — Rescorla-Wagner learning rate.
- ``beta``     — inverse temperature (owned by the decision process, not the
                 learner, but listed in the model's free params).

For a *restless* (drifting) bandit we also support an optional forgetting
rate ``rl_decay`` that pulls Q-values toward ``initial_q`` each trial:

    Q[a] ← (1 - decay) * Q[a] + decay * initial_q      (all arms, every trial)
    Q[c] ← Q[c] + alpha * (feedback - Q[c])             (chosen arm only)

Setting ``rl_decay = 0`` recovers the standard RW rule.  When the model uses
``decay``, declare ``rl_decay`` in ``free_params`` and add it to the HSSM
``include`` list.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# JAX is imported lazily so the module can be inspected without a JAX install.
try:
    import jax.numpy as jnp
except ImportError:  # pragma: no cover
    jnp = None


class NArmRescorlaWagner:
    """N-arm Rescorla-Wagner learner with optional forgetting.

    Parameters
    ----------
    n_actions : int
        Number of choice alternatives (e.g. 4 for the MindRL Challenge).
    initial_q : float
        Initial Q-value for all arms (default 0.5).
    feedback_field : str
        Column name in the trial context that carries the reward (default
        ``"feedback"``).
    use_decay : bool
        If ``True``, add ``rl_decay`` to ``free_params`` and apply exponential
        forgetting toward ``initial_q`` each trial.  Recommended for restless
        (drifting) bandits.
    """

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

    # ------------------------------------------------------------------
    # Protocol properties
    # ------------------------------------------------------------------

    @property
    def n_actions(self) -> int:
        return self._n_actions

    @property
    def computed_params(self) -> list[str]:
        """Q-values exposed to the decision process (e.g. ['q0','q1','q2','q3'])."""
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

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Python / NumPy backend
    # ------------------------------------------------------------------

    def compute_python(
        self,
        state: dict[str, Any],
        params: dict[str, float],
        context: dict[str, Any],
    ) -> dict[str, float]:
        q = state["q_values"]
        return {f"q{i}": float(q[i]) for i in range(self._n_actions)}

    def update_python(
        self,
        state: dict[str, Any],
        params: dict[str, float],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        choice = int(context["choice"])
        feedback = float(context[self._feedback_field])
        alpha = params["rl_alpha"]
        q = np.asarray(state["q_values"], dtype=np.float64).copy()

        # Optional forgetting toward initial_q
        if self._use_decay:
            decay = params.get("rl_decay", 0.0)
            q = (1.0 - decay) * q + decay * self._initial_q

        # RW update on chosen arm only
        q[choice] += alpha * (feedback - q[choice])
        return {"q_values": q}

    # ------------------------------------------------------------------
    # JAX backend (differentiable — used by HSSM NUTS)
    # ------------------------------------------------------------------

    def compute_jax(
        self,
        state: dict[str, Any],
        params: dict[str, float],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        q = state["q_values"]
        return {f"q{i}": q[i] for i in range(self._n_actions)}

    def update_jax(
        self,
        state: dict[str, Any],
        params: dict[str, float],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        choice = context["choice"]
        feedback = context[self._feedback_field]
        alpha = params["rl_alpha"]
        q = state["q_values"]

        if self._use_decay:
            decay = params.get("rl_decay", 0.0)
            q = (1.0 - decay) * q + decay * self._initial_q

        delta = feedback - q[choice]
        return {"q_values": q.at[choice].add(alpha * delta)}

    # ------------------------------------------------------------------
    # Convenience wrappers (used by the ssms.rl.Simulator)
    # ------------------------------------------------------------------

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