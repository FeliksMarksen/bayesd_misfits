"""Exploratory resource-depletion RW heuristic inspired by Bruckner et al. (2025).

Bruckner, Nassar, Li & Eppinger (2025, Psychological Review) propose that
age-related learning differences emerge from a *resource-rational sampling*
process: instead of computing optimal Bayesian updates, the brain draws a
limited number of samples from the posterior. When cognitive resources are
limited, fewer samples are drawn, producing three systematic biases:

1. **Slower learning** — fewer samples → lower effective learning rate
2. **Perseveration** — fewer samples → noisier beliefs → stick to previous choice
3. **Anchoring** — fewer samples → beliefs stay close to the prior

The published model derives these effects from a within-update sampling
process.  This module does **not** implement that sampling process.  It tests a
simpler, additional hypothesis that resources decline across a task trajectory
and directly modulate learning and stickiness.  It must therefore be described
as an exploratory fatigue heuristic rather than a faithful implementation of
the paper.  The trial-varying resource level is:

    rho_t = 1 / (1 + fatigue_rate * t)
    alpha_eff = alpha_base * rho_t                        # less resource → slower learning
    sticky_eff = sticky_base + sticky_gain * (1 - rho_t) # less resource → more perseveration

The optional ``surprise_gain`` parameter adds a PE-dependent modulation of
the learning rate, consistent with the paper's "criterion level of accuracy"
stopping rule: large prediction errors indicate the current belief is
inaccurate, so more sampling is warranted:

    alpha_eff = alpha_base * rho_t * (1 + surprise_gain * |PE|),  capped at 1.0

When ``fatigue_rate = 0`` and ``surprise_gain = 0``, the model reduces
exactly to :class:`~bayesd_misfits.model.NArmRWDualAlphaSticky`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import jax.numpy as jnp
except ImportError:  # pragma: no cover
    jnp = None


class NArmRWDualAlphaStickyResource:
    """N-arm dual-alpha RW with trial-varying resource level and stickiness.

    Free parameters
    ---------------
    rl_alpha_pos, rl_alpha_neg : float in [0, 1]
        Base learning rates for positive / negative prediction errors.
    sticky : float in [-5, 5]
        Baseline perseveration bonus added to the last-chosen arm's Q-value.
    sticky_gain : float in [0, 5]
        How much stickiness *increases* as resources deplete
        (``sticky_eff = sticky + sticky_gain * (1 - rho_t)``).
    fatigue_rate : float in [0, 0.1]
        Rate of within-session resource depletion
        (``rho_t = 1 / (1 + fatigue_rate * t)``).
    surprise_gain : float in [0, 5]
        PE-magnitude gating on the learning rate
        (``alpha_eff *= (1 + surprise_gain * |PE|)``, capped at 1).
    beta : float in [0, 15]
        Softmax inverse temperature (handled by the decision model, not here).
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

    # ------------------------------------------------------------------ #
    # Protocol properties
    # ------------------------------------------------------------------ #
    @property
    def n_actions(self) -> int:
        return self._n_actions

    @property
    def computed_params(self) -> list[str]:
        return [f"q{i}" for i in range(self._n_actions)]

    @property
    def free_params(self) -> list[str]:
        return [
            "rl_alpha_pos",
            "rl_alpha_neg",
            "sticky",
            "sticky_gain",
            "fatigue_rate",
            "surprise_gain",
        ]

    @property
    def param_bounds(self) -> dict[str, tuple[float, float]]:
        return {
            "rl_alpha_pos": (0.0, 1.0),
            "rl_alpha_neg": (0.0, 1.0),
            "sticky": (-5.0, 5.0),
            "sticky_gain": (0.0, 5.0),
            "fatigue_rate": (0.0001, 0.05),
            "surprise_gain": (0.0, 5.0),
        }

    @property
    def default_params(self) -> dict[str, float]:
        return {
            "rl_alpha_pos": 0.3,
            "rl_alpha_neg": 0.3,
            "sticky": 0.0,
            "sticky_gain": 0.0,
            "fatigue_rate": 0.0,
            "surprise_gain": 0.0,
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

    # ------------------------------------------------------------------ #
    # State
    # ------------------------------------------------------------------ #
    def init_state(self) -> dict[str, Any]:
        return {
            "q_values": np.full(self._n_actions, self._initial_q, dtype=np.float64),
            "last_choice": -1,
            "trial": 0,
        }

    def init_jax_state(self) -> dict[str, Any]:
        return {
            "q_values": jnp.full((self._n_actions,), self._initial_q),
            "last_choice": jnp.asarray(-1, dtype=jnp.int32),
            "trial": jnp.asarray(0, dtype=jnp.int32),
        }

    def reset(self, **kwargs) -> None:
        self._state = self.init_state()

    # ------------------------------------------------------------------ #
    # Resource level
    # ------------------------------------------------------------------ #
    @staticmethod
    def _resource_level(trial: Any, fatigue_rate: Any) -> Any:
        """Trial-varying resource level rho_t in (0, 1].

        rho_t = 1 / (1 + fatigue_rate * t)
        At t=0, rho=1 (full resources).  As t → ∞, rho → 0 (depleted).
        """
        if fatigue_rate <= 0:
            return 1.0 if isinstance(trial, (int, float)) else jnp.ones_like(trial, dtype=jnp.float32)
        return 1.0 / (1.0 + fatigue_rate * trial)

    # ------------------------------------------------------------------ #
    # Python backend
    # ------------------------------------------------------------------ #
    def compute_python(self, state, params, context):
        """Return Q-values with resource-adjusted stickiness for the decision model."""
        q = np.asarray(state["q_values"], dtype=np.float64).copy()
        last = int(state["last_choice"])
        trial = int(state.get("trial", 0))

        sticky = float(params["sticky"])
        sticky_gain = float(params["sticky_gain"])
        fatigue_rate = float(params["fatigue_rate"])

        rho = self._resource_level(trial, fatigue_rate)
        sticky_eff = sticky + sticky_gain * (1.0 - rho)

        if 0 <= last < self._n_actions:
            q[last] += sticky_eff
        return {f"q{i}": float(q[i]) for i in range(self._n_actions)}

    def update_python(self, state, params, context):
        choice = int(context["choice"])
        feedback = float(context[self._feedback_field])
        a_pos = float(params["rl_alpha_pos"])
        a_neg = float(params["rl_alpha_neg"])
        sticky = float(params["sticky"])
        sticky_gain = float(params["sticky_gain"])
        fatigue_rate = float(params["fatigue_rate"])
        surprise_gain = float(params["surprise_gain"])

        q = np.asarray(state["q_values"], dtype=np.float64).copy()
        trial = int(state.get("trial", 0))

        # Resource-adjusted learning rate
        rho = self._resource_level(trial, fatigue_rate)
        base_alpha = a_pos if a_pos >= 0 else a_neg  # placeholder; PE sign decides below

        pe = feedback - q[choice]
        base_alpha = a_pos if pe >= 0 else a_neg
        alpha = base_alpha * rho

        # Surprise gating: large |PE| → more effective sampling
        if surprise_gain > 0:
            alpha = alpha * (1.0 + surprise_gain * abs(pe))
        alpha = min(alpha, 0.9999)  # avoid exactly 1.0 for numerical stability

        q[choice] += alpha * pe

        return {
            "q_values": q,
            "last_choice": choice,
            "trial": trial + 1,
        }

    # ------------------------------------------------------------------ #
    # JAX backend (needed for HSSM fitting + vectorised held-out scoring)
    # ------------------------------------------------------------------ #
    def compute_jax(self, state, params, context):
        q = state["q_values"]
        last = state["last_choice"]
        trial = state["trial"]

        sticky = params["sticky"]
        sticky_gain = params["sticky_gain"]
        fatigue_rate = params["fatigue_rate"]

        rho = jnp.where(
            fatigue_rate > 0,
            1.0 / (1.0 + fatigue_rate * trial),
            jnp.ones_like(trial, dtype=q.dtype),
        )
        sticky_eff = sticky + sticky_gain * (1.0 - rho)

        q_biased = jnp.where(
            jnp.arange(self._n_actions) == last,
            q + sticky_eff,
            q,
        )
        return {f"q{i}": q_biased[i] for i in range(self._n_actions)}

    def update_jax(self, state, params, context):
        choice = context["choice"]
        feedback = context[self._feedback_field]
        a_pos = params["rl_alpha_pos"]
        a_neg = params["rl_alpha_neg"]
        fatigue_rate = params["fatigue_rate"]
        surprise_gain = params["surprise_gain"]

        q = state["q_values"]
        trial = state["trial"]

        rho = jnp.where(
            fatigue_rate > 0,
            1.0 / (1.0 + fatigue_rate * trial),
            jnp.ones_like(trial, dtype=q.dtype),
        )

        pe = feedback - q[choice]
        base_alpha = jnp.where(pe >= 0, a_pos, a_neg)
        alpha = base_alpha * rho

        # Surprise gating
        alpha = alpha * (1.0 + surprise_gain * jnp.abs(pe))
        alpha = jnp.clip(alpha, 0.0, 0.9999)

        new_q = q.at[choice].add(alpha * pe)
        return {
            "q_values": new_q,
            "last_choice": choice,
            "trial": trial + 1,
        }

    # ------------------------------------------------------------------ #
    # Convenience wrapper (not used by HSSM, but handy for testing)
    # ------------------------------------------------------------------ #
    def compute_ssm_params(self, trial_params: dict[str, float]) -> dict[str, float]:
        if self._state is None:
            raise RuntimeError("Call reset() before compute_ssm_params()")
        return self.compute_python(self._state, trial_params, context={})

    def update(self, action: int, reward: float, trial_params: dict[str, float]) -> None:
        if self._state is None:
            raise RuntimeError("Call reset() before update()")
        self._state = self.update_python(
            self._state,
            trial_params,
            context={"choice": action, self._feedback_field: reward},
        )
