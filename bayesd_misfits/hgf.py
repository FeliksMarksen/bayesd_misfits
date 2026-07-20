"""Per-arm 2-level continuous HGF learner for HSSM RLSSM.

Hierarchical Gaussian Filter (Mathys et al. 2011): tracks per-arm reward
mean (level 1) and log-volatility (level 2), giving adaptive learning rates.

Free parameters: omega (log-vol drift), kappa (volatility coupling).
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import jax.numpy as jnp
except ImportError:  # pragma: no cover
    jnp = None


class NArmHGF:
    """N-arm 2-level continuous HGF learner for restless bandit tasks."""

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
        return {f"q{i}": float(state["mu1"][i]) for i in range(self._n_actions)}

    def _update_hgf_numpy(self, mu1, sigma1, mu2, sigma2, choice, feedback,
                          omega, kappa):
        """Shared numpy HGF update: predict all arms, update chosen arm."""
        # Prediction (all arms)
        sigma1_hat = sigma1 + np.exp(kappa * mu2 + omega)
        sigma2_hat = sigma2 + self._theta_var
        mu1_hat_chosen = mu1[choice]

        # Level 1 update (chosen arm)
        pi1 = 1.0 / sigma1_hat[choice] + self._obs_precision
        psi1 = self._obs_precision / pi1
        mu1[choice] = mu1_hat_chosen + psi1 * (feedback - mu1_hat_chosen)
        sigma1[choice] = 1.0 / pi1

        # Level 2 update (chosen arm)
        pi1_hat = 1.0 / sigma1_hat[choice]
        pe = mu1[choice] - mu1_hat_chosen
        delta1 = (pi1_hat / pi1) + pi1_hat * pe ** 2 - 1.0
        pi2 = 1.0 / sigma2_hat[choice] + 0.5 * (kappa * pi1_hat) ** 2
        psi2 = 0.5 * kappa * pi1_hat / pi2
        mu2[choice] = mu2[choice] + psi2 * delta1
        sigma2[choice] = 1.0 / pi2

        # Unchosen arms: carry forward predicted uncertainty
        for a in range(self._n_actions):
            if a != choice:
                sigma1[a] = sigma1_hat[a]
                sigma2[a] = sigma2_hat[a]

        return mu1, sigma1, mu2, sigma2

    def update_python(self, state, params, context):
        choice = int(context["choice"])
        feedback = float(context[self._feedback_field])

        mu1 = np.array(state["mu1"], dtype=np.float64)
        sigma1 = np.array(state["sigma1"], dtype=np.float64)
        mu2 = np.array(state["mu2"], dtype=np.float64)
        sigma2 = np.array(state["sigma2"], dtype=np.float64)

        mu1, sigma1, mu2, sigma2 = self._update_hgf_numpy(
            mu1, sigma1, mu2, sigma2, choice, feedback,
            params["omega"], params["kappa"],
        )
        return {"mu1": mu1, "sigma1": sigma1, "mu2": mu2, "sigma2": sigma2}

    def compute_jax(self, state, params, context):
        return {f"q{i}": state["mu1"][i] for i in range(self._n_actions)}

    def update_jax(self, state, params, context):
        choice = context["choice"]
        feedback = context[self._feedback_field]
        omega = params["omega"]
        kappa = params["kappa"]

        mu1 = state["mu1"]
        sigma1 = state["sigma1"]
        mu2 = state["mu2"]
        sigma2 = state["sigma2"]

        # Prediction (all arms)
        sigma1_hat = sigma1 + jnp.exp(kappa * mu2 + omega)
        sigma2_hat = sigma2 + self._theta_var

        # Level 1 update (chosen arm)
        pi1 = 1.0 / sigma1_hat[choice] + self._obs_precision
        psi1 = self._obs_precision / pi1
        new_mu1_chosen = mu1[choice] + psi1 * (feedback - mu1[choice])
        new_sigma1_chosen = 1.0 / pi1

        # Level 2 update (chosen arm)
        pi1_hat = 1.0 / sigma1_hat[choice]
        pe = new_mu1_chosen - mu1[choice]
        delta1 = (pi1_hat / pi1) + pi1_hat * pe ** 2 - 1.0
        pi2 = 1.0 / sigma2_hat[choice] + 0.5 * (kappa * pi1_hat) ** 2
        psi2 = 0.5 * kappa * pi1_hat / pi2
        new_mu2_chosen = mu2[choice] + psi2 * delta1
        new_sigma2_chosen = 1.0 / pi2

        # Scatter: chosen arm gets posterior, unchosen arms get predicted sigma
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
        self._state = self.update_python(
            self._state, trial_params,
            context={"choice": action, self._feedback_field: reward},
        )


class NArmHGFDriftLearner:
    """N-arm HGF outputting scaled drift rates (v = scaler * mu1) for race models."""

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
        # Reuse the NArmHGF numpy update logic
        self._hgf = NArmHGF(
            n_actions=n_actions,
            initial_mu1=initial_mu1, initial_mu2=initial_mu2,
            initial_sigma1=initial_sigma1, initial_sigma2=initial_sigma2,
            theta_var=theta_var, obs_precision=obs_precision,
            feedback_field=feedback_field,
        )

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

        mu1 = np.array(state["mu1"], dtype=np.float64)
        sigma1 = np.array(state["sigma1"], dtype=np.float64)
        mu2 = np.array(state["mu2"], dtype=np.float64)
        sigma2 = np.array(state["sigma2"], dtype=np.float64)

        mu1, sigma1, mu2, sigma2 = self._hgf._update_hgf_numpy(
            mu1, sigma1, mu2, sigma2, choice, feedback,
            params["omega"], params["kappa"],
        )
        return {"mu1": mu1, "sigma1": sigma1, "mu2": mu2, "sigma2": sigma2}

    def update_jax(self, state, params, context):
        choice = context["choice"]
        feedback = context[self._feedback_field]
        omega = params["omega"]
        kappa = params["kappa"]

        mu1 = state["mu1"]
        sigma1 = state["sigma1"]
        mu2 = state["mu2"]
        sigma2 = state["sigma2"]

        sigma1_hat = sigma1 + jnp.exp(kappa * mu2 + omega)
        sigma2_hat = sigma2 + self._theta_var

        mu1_hat_chosen = mu1[choice]
        pi1 = 1.0 / sigma1_hat[choice] + self._obs_precision
        psi1 = self._obs_precision / pi1
        new_mu1_chosen = mu1_hat_chosen + psi1 * (feedback - mu1_hat_chosen)
        new_sigma1_chosen = 1.0 / pi1

        pi1hat = 1.0 / sigma1_hat[choice]
        pe = new_mu1_chosen - mu1_hat_chosen
        delta1 = (pi1hat / pi1) + pi1hat * pe ** 2 - 1.0
        pi2 = 1.0 / sigma2_hat[choice] + 0.5 * (kappa * pi1hat) ** 2
        psi2 = 0.5 * kappa * pi1hat / pi2
        new_mu2_chosen = mu2[choice] + psi2 * delta1
        new_sigma2_chosen = 1.0 / pi2

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
        self._state = self.update_python(
            self._state, trial_params,
            context={"choice": action, self._feedback_field: reward},
        )
