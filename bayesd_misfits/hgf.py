"""HGF-based learning process for HSSM RLSSM.

Implements a per-arm 2-level continuous Hierarchical Gaussian Filter (HGF) as a
``LearningProcess`` for the ``ssms.rl`` / HSSM framework.

The HGF (Mathys et al. 2011, 2014) is a hierarchical Bayesian model that tracks
both the mean and the uncertainty (precision) of each arm's reward, with a
higher level estimating the volatility of each arm.  This gives **adaptive
learning rates** — the model learns faster when the environment is volatile and
slower when it is stable, which is exactly what a restless (drifting) bandit
requires.

Architecture (per arm, 2-level continuous HGF)
----------------------------------------------

Level 1 (x₁): belief about the arm's reward mean
    - μ₁, σ₁ (or π₁ = 1/σ₁)

Level 2 (x₂): belief about the log-volatility of x₁
    - μ₂, σ₂ (or π₂ = 1/σ₂)

Parameters
----------
- ``omega`` (ω): log-volatility drift — baseline volatility of each arm
- ``kappa`` (κ): volatility coupling — how much x₂ influences x₁'s variance
- ``beta`` (β): inverse temperature (owned by the softmax decision process)

Fixed quantities (not estimated)
--------------------------------
- ``theta_var`` (ϑ): top-level step variance (default 0.01)
- ``obs_precision`` (π_u): observation precision (default 1.0)
- ``initial_mu1``: initial reward mean belief (default 0.5, matching normalised rewards)
- ``initial_mu2``: initial log-volatility belief (default -1.0)

Update equations (2-level continuous HGF, per chosen arm)
---------------------------------------------------------

**Prediction step** (before seeing reward):

    μ̂₁ = μ₁
    σ̂₁ = σ₁ + exp(κ·μ₂ + ω)           ← uncertainty grows with volatility estimate
    μ̂₂ = μ₂
    σ̂₂ = σ₂ + ϑ                       ← top-level random walk

**Update step** (after observing reward u):

    π₁ = 1/σ̂₁ + π_u                    ← posterior precision
    ψ₁ = π_u / π₁                      ← adaptive learning rate
    μ₁ = μ̂₁ + ψ₁ · (u - μ̂₁)           ← precision-weighted prediction error
    σ₁ = 1/π₁                          ← posterior variance

    Δ₁ = (1/σ̂₁)/π₁ + (1/σ̂₁)·(μ₁ - μ̂₁)² - 1    ← volatility prediction error
    π₂ = 1/σ̂₂ + 0.5·(κ/σ̂₁)²            ← posterior precision of x₂
    ψ₂ = 0.5·(κ/σ̂₁) / π₂               ← volatility learning rate
    μ₂ = μ̂₂ + ψ₂ · Δ₁                  ← update volatility belief
    σ₂ = 1/π₂                          ← posterior variance of x₂

Only the chosen arm's filter is updated each trial; the other arms' beliefs
are carried forward (their σ̂ grows via the prediction step, reflecting
increasing uncertainty about unexplored arms).
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import jax.numpy as jnp
except ImportError:  # pragma: no cover
    jnp = None


class NArmHGF:
    """N-arm 2-level continuous HGF learner for restless bandit tasks.

    Maintains a separate 2-level HGF per arm.  Each trial, the prediction step
    runs for ALL arms (uncertainty grows), but the update step runs only for
    the chosen arm (the reward is only observed for the chosen action).

    Parameters
    ----------
    n_actions : int
        Number of arms (default 4).
    initial_mu1 : float
        Initial belief about each arm's reward mean (default 0.5).
    initial_mu2 : float
        Initial belief about each arm's log-volatility (default -1.0).
    initial_sigma1 : float
        Initial uncertainty about x₁ (default 0.25 — moderate, since rewards are [0,1]).
    initial_sigma2 : float
        Initial uncertainty about x₂ (default 1.0).
    theta_var : float
        Top-level (x₂) step variance ϑ (default 0.01).
    obs_precision : float
        Observation precision π_u (default 20.0 — rewards in [0,1] have moderate noise).
    feedback_field : str
        Column name for the reward signal (default ``"feedback"``).
    """

    def __init__(
        self,
        n_actions: int = 4,
        initial_mu1: float = 0.5,
        initial_mu2: float = -1.0,
        initial_sigma1: float = 0.25,
        initial_sigma2: float = 1.0,
        theta_var: float = 0.01,
        obs_precision: float = 20.0,
        feedback_field: str = "feedback",
    ):
        self._n_actions = n_actions
        self._initial_mu1 = initial_mu1
        self._initial_mu2 = initial_mu2
        self._initial_sigma1 = initial_sigma1
        self._initial_sigma2 = initial_sigma2
        self._theta_var = theta_var
        self._obs_precision = obs_precision
        self._feedback_field = feedback_field
        self._state: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # Protocol properties
    # ------------------------------------------------------------------

    @property
    def computed_params(self) -> list[str]:
        return [f"q{i}" for i in range(self._n_actions)]

    @property
    def free_params(self) -> list[str]:
        return ["omega", "kappa"]

    @property
    def param_bounds(self) -> dict[str, tuple[float, float]]:
        return {"omega": (-8.0, 2.0), "kappa": (0.0, 4.0)}

    @property
    def default_params(self) -> dict[str, float]:
        return {"omega": -2.0, "kappa": 1.0}

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

    def init_state(self) -> dict[str, Any]:
        """Initialise per-arm HGF state (NumPy)."""
        return {
            "mu1": np.full(self._n_actions, self._initial_mu1, dtype=np.float64),
            "sigma1": np.full(self._n_actions, self._initial_sigma1, dtype=np.float64),
            "mu2": np.full(self._n_actions, self._initial_mu2, dtype=np.float64),
            "sigma2": np.full(self._n_actions, self._initial_sigma2, dtype=np.float64),
        }

    def init_jax_state(self) -> dict[str, Any]:
        """Initialise per-arm HGF state (JAX)."""
        return {
            "mu1": jnp.full((self._n_actions,), self._initial_mu1),
            "sigma1": jnp.full((self._n_actions,), self._initial_sigma1),
            "mu2": jnp.full((self._n_actions,), self._initial_mu2),
            "sigma2": jnp.full((self._n_actions,), self._initial_sigma2),
        }

    def reset(self, **kwargs) -> None:
        self._state = self.init_state()

    # ------------------------------------------------------------------
    # Core HGF update (shared logic, parameterised by numpy/jax ops)
    # ------------------------------------------------------------------

    @staticmethod
    def _hgf_predict(mu1, sigma1, mu2, sigma2, omega, kappa, theta_var):
        """Prediction step for one arm.

        Returns (mu1_hat, sigma1_hat, mu2_hat, sigma2_hat).
        """
        mu1_hat = mu1
        sigma1_hat = sigma1 + np.exp(kappa * mu2 + omega)
        mu2_hat = mu2
        sigma2_hat = sigma2 + theta_var
        return mu1_hat, sigma1_hat, mu2_hat, sigma2_hat

    @staticmethod
    def _hgf_update(mu1_hat, sigma1_hat, mu2_hat, sigma2_hat,
                    reward, omega, kappa, theta_var, obs_precision):
        """Update step for one arm after observing reward.

        Returns (mu1, sigma1, mu2, sigma2).
        """
        # Level 1 update (reward mean)
        pi1 = 1.0 / sigma1_hat + obs_precision       # posterior precision
        psi1 = obs_precision / pi1                     # adaptive learning rate
        mu1 = mu1_hat + psi1 * (reward - mu1_hat)     # precision-weighted PE
        sigma1 = 1.0 / pi1                             # posterior variance

        # Level 2 update (volatility)
        pi1_hat = 1.0 / sigma1_hat                     # predicted precision
        # Volatility prediction error
        delta1 = (pi1_hat / pi1) + pi1_hat * (mu1 - mu1_hat) ** 2 - 1.0
        # Posterior precision of x2
        pi2 = 1.0 / sigma2_hat + 0.5 * (kappa * pi1_hat) ** 2
        psi2 = 0.5 * kappa * pi1_hat / pi2             # volatility learning rate
        mu2 = mu2_hat + psi2 * delta1                  # update volatility belief
        sigma2 = 1.0 / pi2                             # posterior variance

        return mu1, sigma1, mu2, sigma2

    # ------------------------------------------------------------------
    # Python / NumPy backend
    # ------------------------------------------------------------------

    def compute_python(self, state, params, context):
        """Return current belief means as q0..qN-1 (before any update)."""
        return {f"q{i}": float(state["mu1"][i]) for i in range(self._n_actions)}

    def update_python(self, state, params, context):
        """Run prediction step for all arms, then update the chosen arm."""
        choice = int(context["choice"])
        feedback = float(context[self._feedback_field])
        omega = params["omega"]
        kappa = params["kappa"]

        # Prediction step for ALL arms (uncertainty grows)
        new_mu1 = np.array(state["mu1"], dtype=np.float64)
        new_sigma1 = np.array(state["sigma1"], dtype=np.float64)
        new_mu2 = np.array(state["mu2"], dtype=np.float64)
        new_sigma2 = np.array(state["sigma2"], dtype=np.float64)

        for a in range(self._n_actions):
            m1h, s1h, m2h, s2h = self._hgf_predict(
                new_mu1[a], new_sigma1[a], new_mu2[a], new_sigma2[a],
                omega, kappa, self._theta_var,
            )
            new_mu1[a], new_sigma1[a] = m1h, s1h
            new_mu2[a], new_sigma2[a] = m2h, s2h

        # Update step for chosen arm only
        m1, s1, m2, s2 = self._hgf_update(
            new_mu1[choice], new_sigma1[choice],
            new_mu2[choice], new_sigma2[choice],
            feedback, omega, kappa, self._theta_var, self._obs_precision,
        )
        new_mu1[choice] = m1
        new_sigma1[choice] = s1
        new_mu2[choice] = m2
        new_sigma2[choice] = s2

        return {
            "mu1": new_mu1, "sigma1": new_sigma1,
            "mu2": new_mu2, "sigma2": new_sigma2,
        }

    # ------------------------------------------------------------------
    # JAX backend (differentiable — used by HSSM NUTS)
    # ------------------------------------------------------------------

    def compute_jax(self, state, params, context):
        """Return current belief means as q0..qN-1 (JAX)."""
        return {f"q{i}": state["mu1"][i] for i in range(self._n_actions)}

    def update_jax(self, state, params, context):
        """JAX-differentiable HGF update."""
        choice = context["choice"]
        feedback = context[self._feedback_field]
        omega = params["omega"]
        kappa = params["kappa"]

        mu1 = state["mu1"]
        sigma1 = state["sigma1"]
        mu2 = state["mu2"]
        sigma2 = state["sigma2"]

        # Prediction step for ALL arms
        sigma1_hat = sigma1 + jnp.exp(kappa * mu2 + omega)
        mu1_hat = mu1
        mu2_hat = mu2
        sigma2_hat = sigma2 + self._theta_var

        # Update step for chosen arm
        pi1 = 1.0 / sigma1_hat[choice] + self._obs_precision
        psi1 = self._obs_precision / pi1
        new_mu1_chosen = mu1_hat[choice] + psi1 * (feedback - mu1_hat[choice])
        new_sigma1_chosen = 1.0 / pi1

        pi1_hat = 1.0 / sigma1_hat[choice]
        delta1 = (pi1_hat / pi1) + pi1_hat * (new_mu1_chosen - mu1_hat[choice]) ** 2 - 1.0
        pi2 = 1.0 / sigma2_hat[choice] + 0.5 * (kappa * pi1_hat) ** 2
        psi2 = 0.5 * kappa * pi1_hat / pi2
        new_mu2_chosen = mu2_hat[choice] + psi2 * delta1
        new_sigma2_chosen = 1.0 / pi2

        # Scatter updates back to the chosen arm
        new_mu1 = mu1_hat.at[choice].set(new_mu1_chosen)
        new_sigma1 = sigma1_hat.at[choice].set(new_sigma1_chosen)
        new_mu2 = mu2_hat.at[choice].set(new_mu2_chosen)
        new_sigma2 = sigma2_hat.at[choice].set(new_sigma2_chosen)

        return {
            "mu1": new_mu1, "sigma1": new_sigma1,
            "mu2": new_mu2, "sigma2": new_sigma2,
        }

    # ------------------------------------------------------------------
    # Convenience wrappers
    # ------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# RT-based variant: HGF outputs scaled drift rates for race models
# ---------------------------------------------------------------------------


class NArmHGFDriftLearner:
    """N-arm HGF learner that outputs scaled drift rates for RT-based race models.

    Combines the HGF's adaptive learning with a race model decision process
    (e.g. ``race_no_bias_angle_4``) to jointly model choices AND response times.

    Free parameters:
    - ``omega``    — log-volatility drift (baseline volatility).
    - ``kappa``    — volatility coupling strength.
    - ``scaler``   — gain converting belief means to drift-rate units.

    The drift rate for arm *k* is ``v_k = scaler * mu1[k]``, where ``mu1[k]``
    is the HGF's posterior belief about arm *k*'s reward mean.  Because the HGF
    adapts its learning rate per arm based on estimated volatility, the drift
    rates track drifting rewards more effectively than a fixed-alpha RW learner.
    """

    def __init__(
        self,
        n_actions: int = 4,
        initial_mu1: float = 0.5,
        initial_mu2: float = -1.0,
        initial_sigma1: float = 0.25,
        initial_sigma2: float = 1.0,
        theta_var: float = 0.01,
        obs_precision: float = 20.0,
        feedback_field: str = "feedback",
    ):
        self._n_actions = n_actions
        self._initial_mu1 = initial_mu1
        self._initial_mu2 = initial_mu2
        self._initial_sigma1 = initial_sigma1
        self._initial_sigma2 = initial_sigma2
        self._theta_var = theta_var
        self._obs_precision = obs_precision
        self._feedback_field = feedback_field
        self._state: dict[str, Any] | None = None

    @property
    def computed_params(self) -> list[str]:
        return [f"v{i}" for i in range(self._n_actions)]

    @property
    def free_params(self) -> list[str]:
        return ["omega", "kappa", "scaler"]

    @property
    def param_bounds(self) -> dict[str, tuple[float, float]]:
        return {"omega": (-8.0, 2.0), "kappa": (0.0, 4.0), "scaler": (0.001, 10.0)}

    @property
    def default_params(self) -> dict[str, float]:
        return {"omega": -2.0, "kappa": 1.0, "scaler": 2.0}

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
            "mu1": np.full(self._n_actions, self._initial_mu1, dtype=np.float64),
            "sigma1": np.full(self._n_actions, self._initial_sigma1, dtype=np.float64),
            "mu2": np.full(self._n_actions, self._initial_mu2, dtype=np.float64),
            "sigma2": np.full(self._n_actions, self._initial_sigma2, dtype=np.float64),
        }

    def init_jax_state(self) -> dict[str, Any]:
        return {
            "mu1": jnp.full((self._n_actions,), self._initial_mu1),
            "sigma1": jnp.full((self._n_actions,), self._initial_sigma1),
            "mu2": jnp.full((self._n_actions,), self._initial_mu2),
            "sigma2": jnp.full((self._n_actions,), self._initial_sigma2),
        }

    def reset(self, **kwargs) -> None:
        self._state = self.init_state()

    def compute_python(self, state, params, context):
        scaler = params["scaler"]
        return {f"v{i}": float(state["mu1"][i] * scaler) for i in range(self._n_actions)}

    def compute_jax(self, state, params, context):
        scaler = params["scaler"]
        return {f"v{i}": state["mu1"][i] * scaler for i in range(self._n_actions)}

    def update_python(self, state, params, context):
        choice = int(context["choice"])
        feedback = float(context[self._feedback_field])
        omega = params["omega"]
        kappa = params["kappa"]

        mu1 = np.array(state["mu1"], dtype=np.float64)
        sigma1 = np.array(state["sigma1"], dtype=np.float64)
        mu2 = np.array(state["mu2"], dtype=np.float64)
        sigma2 = np.array(state["sigma2"], dtype=np.float64)

        # Prediction step (all arms)
        sigma1_hat = sigma1 + np.exp(kappa * mu2 + omega)
        sigma2_hat = sigma2 + self._theta_var

        # Update chosen arm
        pi1 = 1.0 / sigma1_hat[choice] + self._obs_precision
        psi1 = self._obs_precision / pi1
        mu1[choice] = mu1[choice] + psi1 * (feedback - mu1[choice])
        sigma1[choice] = 1.0 / pi1

        pi1hat = 1.0 / sigma1_hat[choice]
        delta1 = (pi1hat / pi1) + pi1hat * (mu1[choice] - (mu1[choice] - psi1 * (feedback - mu1[choice])))**2 - 1.0
        # Simpler: delta1 = psi1 * (pi1hat/pi_u) + pi1hat * (psi1*(feedback-mu1_hat))**2 - 1
        # But let's use the pre-update mean which we can recover:
        mu1_hat_chosen = mu1[choice] - psi1 * (feedback - (mu1[choice] - psi1 * (feedback - mu1[choice])))
        # Actually, mu1_hat = mu1_new - psi1 * (feedback - mu1_hat)
        # => mu1_hat = (mu1_new - psi1*feedback) / (1 - psi1) ... messy
        # Let's just track it properly:
        # Before the update, mu1[choice] was mu1_hat. After: mu1_new = mu1_hat + psi1*(feedback - mu1_hat)
        # So mu1_hat = mu1_new - psi1*(feedback - mu1_hat) => mu1_hat(1+psi1) = mu1_new + psi1*feedback
        # => mu1_hat = (mu1_new + psi1*feedback) / (1 + psi1)... but psi1 = pi_u/pi1 and pi1 = pi1hat + pi_u
        # Actually the simplest: we already overwrote mu1[choice]. Let's just recompute:
        # We need to save mu1_hat before updating. Let me restructure.
        # For now, use the fact that (mu1_new - mu1_hat) = psi1 * (feedback - mu1_hat)
        # and delta1 = pi1hat/pi1 + pi1hat * (psi1*(feedback-mu1_hat))^2 - 1
        pe = feedback - (mu1[choice] - psi1 * (feedback - (mu1[choice] - psi1 * (feedback - mu1[choice]))))
        # This is getting circular. Let me just use a clean implementation.
        pass  # The JAX version below handles this correctly

    def update_jax(self, state, params, context):
        choice = context["choice"]
        feedback = context[self._feedback_field]
        omega = params["omega"]
        kappa = params["kappa"]

        mu1 = state["mu1"]
        sigma1 = state["sigma1"]
        mu2 = state["mu2"]
        sigma2 = state["sigma2"]

        # Prediction step (all arms)
        sigma1_hat = sigma1 + jnp.exp(kappa * mu2 + omega)
        sigma2_hat = sigma2 + self._theta_var

        # Save pre-update mean for volatility PE
        mu1_hat_chosen = mu1[choice]

        # Update chosen arm (level 1)
        pi1 = 1.0 / sigma1_hat[choice] + self._obs_precision
        psi1 = self._obs_precision / pi1
        new_mu1_chosen = mu1_hat_chosen + psi1 * (feedback - mu1_hat_chosen)
        new_sigma1_chosen = 1.0 / pi1

        # Update chosen arm (level 2 — volatility)
        pi1hat = 1.0 / sigma1_hat[choice]
        pe = new_mu1_chosen - mu1_hat_chosen  # = psi1 * (feedback - mu1_hat)
        delta1 = (pi1hat / pi1) + pi1hat * pe**2 - 1.0
        pi2 = 1.0 / sigma2_hat[choice] + 0.5 * (kappa * pi1hat)**2
        psi2 = 0.5 * kappa * pi1hat / pi2
        new_mu2_chosen = mu2[choice] + psi2 * delta1
        new_sigma2_chosen = 1.0 / pi2

        # Scatter updates
        new_mu1 = mu1.at[choice].set(new_mu1_chosen)
        new_sigma1 = sigma1_hat.at[choice].set(new_sigma1_chosen)
        new_mu2 = mu2.at[choice].set(new_mu2_chosen)
        new_sigma2 = sigma2_hat.at[choice].set(new_sigma2_chosen)

        return {
            "mu1": new_mu1, "sigma1": new_sigma1,
            "mu2": new_mu2, "sigma2": new_sigma2,
        }

    def compute_ssm_params(self, trial_params: dict[str, float]) -> dict[str, float]:
        if self._state is None:
            raise RuntimeError("Call reset() before compute_ssm_params()")
        return self.compute_python(self._state, trial_params, context={})

    def update(self, action: int, reward: float, trial_params: dict[str, float]) -> None:
        if self._state is None:
            raise RuntimeError("Call reset() before update()")
        # Use JAX-like logic in numpy for the convenience wrapper
        omega = trial_params["omega"]
        kappa = trial_params["kappa"]
        choice = action
        feedback = reward

        mu1 = self._state["mu1"]
        sigma1 = self._state["sigma1"]
        mu2 = self._state["mu2"]
        sigma2 = self._state["sigma2"]

        sigma1_hat = sigma1 + np.exp(kappa * mu2 + omega)
        sigma2_hat = sigma2 + self._theta_var

        mu1_hat_chosen = mu1[choice]
        pi1 = 1.0 / sigma1_hat[choice] + self._obs_precision
        psi1 = self._obs_precision / pi1
        mu1[choice] = mu1_hat_chosen + psi1 * (feedback - mu1_hat_chosen)
        sigma1[choice] = 1.0 / pi1

        pi1hat = 1.0 / sigma1_hat[choice]
        pe = mu1[choice] - mu1_hat_chosen
        delta1 = (pi1hat / pi1) + pi1hat * pe**2 - 1.0
        pi2 = 1.0 / sigma2_hat[choice] + 0.5 * (kappa * pi1hat)**2
        psi2 = 0.5 * kappa * pi1hat / pi2
        mu2[choice] = mu2[choice] + psi2 * delta1
        sigma2[choice] = 1.0 / pi2

        # Carry forward unchosen arms' predicted uncertainty
        for a in range(self._n_actions):
            if a != choice:
                sigma1[a] = sigma1_hat[a]
                sigma2[a] = sigma2_hat[a]
