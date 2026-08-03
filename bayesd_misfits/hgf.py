"""Per-arm 2-level continuous HGF learners for HSSM RLSSM.

The level-2 update uses the unbounded HGF (uHGF) two-expansion update.  Unlike
the older single local quadratic update, it keeps the volatility posterior
well defined when the local curvature at the prediction is uninformative or
negative.

Free parameters: omega (log-vol drift), kappa (volatility coupling).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import expit, lambertw

try:
    import jax.numpy as jnp
    from jax.nn import sigmoid as jax_sigmoid
    from pyhgf.math import lambert_w0
except ImportError:  # pragma: no cover
    jnp = None
    jax_sigmoid = None
    lambert_w0 = None


_MAX_LOG_VOLATILITY = 30.0
_MAX_POSTERIOR_PRECISION = 1e10


def _unbounded_volatility_update_numpy(
    previous_variance: float,
    posterior_variance: float,
    prediction_error: float,
    expected_mean: float,
    expected_variance: float,
    omega: float,
    kappa: float,
) -> tuple[float, float]:
    """Return the uHGF volatility-parent posterior mean and variance."""
    previous_variance = max(float(previous_variance), 1e-128)
    posterior_variance = max(float(posterior_variance), 1e-128)
    expected_variance = max(float(expected_variance), 1e-128)
    expected_precision = 1.0 / expected_variance
    be_aux = max(posterior_variance + float(prediction_error) ** 2, 1e-128)
    kappa = float(kappa)

    # With zero coupling, the child carries no information about volatility.
    if abs(kappa) <= 1e-12:
        return float(expected_mean), expected_variance

    log_previous_variance = np.log(previous_variance)
    gamma_c = kappa * float(expected_mean) + float(omega)
    w1 = expit(gamma_c - log_previous_variance)
    log_expected_child_variance = np.logaddexp(log_previous_variance, gamma_c)
    delta = be_aux * np.exp(-log_expected_child_variance) - 1.0

    # Expansion around the volatility prediction.
    precision1 = expected_precision + 0.5 * kappa**2 * w1 * (1.0 - w1)
    mean1 = expected_mean + kappa * w1 * delta / (2.0 * precision1)

    # Expansion around an approximate mode obtained with Lambert W0.
    precision_y = expected_precision / kappa**2
    log_w_arg = (
        np.log(be_aux)
        - np.log(2.0 * precision_y)
        + 0.5 / precision_y
        - gamma_c
    )
    w_arg = np.exp(min(log_w_arg, np.log(np.finfo(np.float64).max)))
    w_value = float(np.real(lambertw(w_arg, k=0)))
    y_star = gamma_c + w_value - 0.5 / precision_y
    x_star = (y_star - float(omega)) / kappa

    log_s2 = kappa * x_star + float(omega)
    log_denom2 = np.logaddexp(log_previous_variance, log_s2)
    weight2 = expit(log_s2 - log_previous_variance)
    delta2 = be_aux * np.exp(-log_denom2) - 1.0
    precision2 = expected_precision + 0.5 * kappa**2 * weight2 * (
        weight2 + (2.0 * weight2 - 1.0) * delta2
    )
    if precision2 <= 0.0:
        precision2 = expected_precision + 0.5 * kappa**2 * weight2 * (1.0 - weight2)
    mean2 = x_star + (
        0.5 * kappa * weight2 * delta2
        - expected_precision * (x_star - expected_mean)
    ) / precision2
    if not (np.isfinite(precision2) and np.isfinite(mean2)):
        precision2, mean2 = precision1, mean1

    def energy(mean: float) -> float:
        log_denom = np.logaddexp(
            log_previous_variance, kappa * mean + float(omega)
        )
        return (
            -0.5 * log_denom
            - 0.5 * be_aux * np.exp(-log_denom)
            - 0.5 * expected_precision * (mean - expected_mean) ** 2
        )

    blend = expit(energy(mean2) - energy(mean1))
    posterior_mean = (1.0 - blend) * mean1 + blend * mean2
    posterior_variance_parent = (
        (1.0 - blend) / precision1
        + blend / precision2
        + blend * (1.0 - blend) * (mean1 - mean2) ** 2
    )
    posterior_variance_parent = max(
        float(posterior_variance_parent), 1.0 / _MAX_POSTERIOR_PRECISION
    )
    return float(posterior_mean), posterior_variance_parent


def _unbounded_volatility_update_jax(
    previous_variance,
    posterior_variance,
    prediction_error,
    expected_mean,
    expected_variance,
    omega,
    kappa,
):
    """JAX equivalent of :func:`_unbounded_volatility_update_numpy`."""
    if jnp is None or jax_sigmoid is None or lambert_w0 is None:  # pragma: no cover
        raise ImportError("The JAX HGF backend requires jax and pyhgf")

    previous_variance = jnp.maximum(previous_variance, 1e-30)
    posterior_variance = jnp.maximum(posterior_variance, 1e-30)
    expected_variance = jnp.maximum(expected_variance, 1e-30)
    expected_precision = 1.0 / expected_variance
    be_aux = jnp.maximum(posterior_variance + prediction_error**2, 1e-30)
    safe_kappa = jnp.where(jnp.abs(kappa) > 1e-7, kappa, 1e-7)

    log_previous_variance = jnp.log(previous_variance)
    gamma_c = safe_kappa * expected_mean + omega
    weight1 = jax_sigmoid(gamma_c - log_previous_variance)
    log_expected_child_variance = jnp.logaddexp(log_previous_variance, gamma_c)
    delta = be_aux * jnp.exp(-log_expected_child_variance) - 1.0
    precision1 = expected_precision + 0.5 * safe_kappa**2 * weight1 * (1.0 - weight1)
    mean1 = expected_mean + safe_kappa * weight1 * delta / (2.0 * precision1)

    precision_y = expected_precision / safe_kappa**2
    log_w_arg = (
        jnp.log(be_aux)
        - jnp.log(2.0 * precision_y)
        + 0.5 / precision_y
        - gamma_c
    )
    log_float_max = jnp.log(jnp.finfo(jnp.result_type(log_w_arg)).max)
    w_value = lambert_w0(jnp.exp(jnp.minimum(log_w_arg, log_float_max)))
    y_star = gamma_c + w_value - 0.5 / precision_y
    x_star = (y_star - omega) / safe_kappa

    log_s2 = safe_kappa * x_star + omega
    log_denom2 = jnp.logaddexp(log_previous_variance, log_s2)
    weight2 = jax_sigmoid(log_s2 - log_previous_variance)
    delta2 = be_aux * jnp.exp(-log_denom2) - 1.0
    precision2_full = expected_precision + 0.5 * safe_kappa**2 * weight2 * (
        weight2 + (2.0 * weight2 - 1.0) * delta2
    )
    precision2_safe = jnp.where(
        precision2_full <= 0.0,
        expected_precision + 0.5 * safe_kappa**2 * weight2 * (1.0 - weight2),
        precision2_full,
    )
    mean2_safe = x_star + (
        0.5 * safe_kappa * weight2 * delta2
        - expected_precision * (x_star - expected_mean)
    ) / precision2_safe

    expansion2_finite = jnp.isfinite(precision2_safe) & jnp.isfinite(mean2_safe)
    precision2 = jnp.where(
        expansion2_finite,
        jnp.where(expansion2_finite, precision2_safe, 1.0),
        precision1,
    )
    mean2 = jnp.where(
        expansion2_finite,
        jnp.where(expansion2_finite, mean2_safe, 0.0),
        mean1,
    )

    def energy(mean):
        log_denom = jnp.logaddexp(
            log_previous_variance, safe_kappa * mean + omega
        )
        return (
            -0.5 * log_denom
            - 0.5 * be_aux * jnp.exp(-log_denom)
            - 0.5 * expected_precision * (mean - expected_mean) ** 2
        )

    blend = jax_sigmoid(energy(mean2) - energy(mean1))
    posterior_mean = (1.0 - blend) * mean1 + blend * mean2
    posterior_variance_parent = (
        (1.0 - blend) / precision1
        + blend / precision2
        + blend * (1.0 - blend) * (mean1 - mean2) ** 2
    )
    posterior_variance_parent = jnp.maximum(
        posterior_variance_parent, 1.0 / _MAX_POSTERIOR_PRECISION
    )

    coupled = jnp.abs(kappa) > 1e-7
    safe_mean = jnp.where(jnp.isfinite(posterior_mean), posterior_mean, expected_mean)
    safe_variance = jnp.where(
        jnp.isfinite(posterior_variance_parent),
        posterior_variance_parent,
        expected_variance,
    )
    return (
        jnp.where(coupled, safe_mean, expected_mean),
        jnp.where(coupled, safe_variance, expected_variance),
    )


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
    def n_actions(self) -> int:
        return self._n_actions

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
        """Shared NumPy uHGF update: predict all arms, update the chosen arm."""
        previous_sigma1_chosen = sigma1[choice]

        # Prediction (all arms)
        log_volatility = np.clip(
            float(kappa) * mu2 + float(omega),
            -_MAX_LOG_VOLATILITY,
            _MAX_LOG_VOLATILITY,
        )
        sigma1_hat = sigma1 + np.exp(log_volatility)
        sigma2_hat = sigma2 + self._theta_var
        mu1_hat_chosen = mu1[choice]

        # Level 1 update (chosen arm)
        pi1 = 1.0 / sigma1_hat[choice] + self._obs_precision
        psi1 = self._obs_precision / pi1
        mu1[choice] = mu1_hat_chosen + psi1 * (feedback - mu1_hat_chosen)
        sigma1[choice] = 1.0 / pi1

        # Level 2 robust volatility update (chosen arm)
        pe = mu1[choice] - mu1_hat_chosen
        mu2[choice], sigma2[choice] = _unbounded_volatility_update_numpy(
            previous_variance=previous_sigma1_chosen,
            posterior_variance=sigma1[choice],
            prediction_error=pe,
            expected_mean=mu2[choice],
            expected_variance=sigma2_hat[choice],
            omega=omega,
            kappa=kappa,
        )

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
        previous_sigma1_chosen = sigma1[choice]

        # Prediction (all arms)
        log_volatility = jnp.clip(
            kappa * mu2 + omega,
            -_MAX_LOG_VOLATILITY,
            _MAX_LOG_VOLATILITY,
        )
        sigma1_hat = sigma1 + jnp.exp(log_volatility)
        sigma2_hat = sigma2 + self._theta_var

        # Level 1 update (chosen arm)
        pi1 = 1.0 / sigma1_hat[choice] + self._obs_precision
        psi1 = self._obs_precision / pi1
        new_mu1_chosen = mu1[choice] + psi1 * (feedback - mu1[choice])
        new_sigma1_chosen = 1.0 / pi1

        # Level 2 robust volatility update (chosen arm)
        pe = new_mu1_chosen - mu1[choice]
        new_mu2_chosen, new_sigma2_chosen = _unbounded_volatility_update_jax(
            previous_variance=previous_sigma1_chosen,
            posterior_variance=new_sigma1_chosen,
            prediction_error=pe,
            expected_mean=mu2[choice],
            expected_variance=sigma2_hat[choice],
            omega=omega,
            kappa=kappa,
        )

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


class NArmHGFSticky(NArmHGF):
    """N-arm uHGF with a one-trial choice perseveration term."""

    @property
    def free_params(self) -> list[str]:
        return ["omega", "kappa", "sticky"]

    @property
    def param_bounds(self) -> dict[str, tuple[float, float]]:
        return {
            "omega": (-8.0, 2.0),
            "kappa": (0.0, 4.0),
            "sticky": (-3.0, 3.0),
        }

    @property
    def default_params(self) -> dict[str, float]:
        return {"omega": -2.0, "kappa": 1.0, "sticky": 0.0}

    def init_state(self) -> dict[str, Any]:
        state = super().init_state()
        state["last_choice"] = -1
        return state

    def init_jax_state(self) -> dict[str, Any]:
        state = super().init_jax_state()
        state["last_choice"] = jnp.asarray(-1, dtype=jnp.int32)
        return state

    def compute_python(self, state, params, context):
        values = np.array(state["mu1"], dtype=np.float64)
        last_choice = int(state["last_choice"])
        if 0 <= last_choice < self._n_actions:
            values[last_choice] += float(params["sticky"])
        return {f"q{i}": float(values[i]) for i in range(self._n_actions)}

    def compute_jax(self, state, params, context):
        last_choice = state["last_choice"]
        valid = (last_choice >= 0) & (last_choice < self._n_actions)
        safe_choice = jnp.clip(last_choice, 0, self._n_actions - 1)
        values = state["mu1"].at[safe_choice].add(
            jnp.where(valid, params["sticky"], 0.0)
        )
        return {f"q{i}": values[i] for i in range(self._n_actions)}

    def update_python(self, state, params, context):
        updated = super().update_python(state, params, context)
        updated["last_choice"] = int(context["choice"])
        return updated

    def update_jax(self, state, params, context):
        updated = super().update_jax(state, params, context)
        updated["last_choice"] = context["choice"]
        return updated


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
    def n_actions(self) -> int:
        return self._n_actions

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
        return self._hgf.init_state()

    def init_jax_state(self) -> dict[str, Any]:
        return self._hgf.init_jax_state()

    def reset(self, **kwargs) -> None:
        self._state = self.init_state()

    def compute_python(self, state, params, context):
        scaler = params["scaler"]
        return {f"v{i}": float(state["mu1"][i] * scaler) for i in range(self._n_actions)}

    def compute_jax(self, state, params, context):
        scaler = params["scaler"]
        return {f"v{i}": state["mu1"][i] * scaler for i in range(self._n_actions)}

    def update_python(self, state, params, context):
        return self._hgf.update_python(state, params, context)

    def update_jax(self, state, params, context):
        return self._hgf.update_jax(state, params, context)

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
