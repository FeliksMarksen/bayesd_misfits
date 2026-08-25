"""Causal sampling bandit: a BMS-inspired process model for the 4-arm bandit.

This module adapts the Bayesian Mutation Sampler (Kolvoort, Temme & van Maanen,
2023; see :mod:`bayesd_misfits.bms`) to the MindRL 4-arm drifting bandit.

The agent represents a hypothesis about the *latent causes* of reward -- the
four arm values -- as a discretized level per arm.  It does not compute a point
estimate; instead it runs a Metropolis-Hastings "mutation" chain over the
discretized value space, warm-started from the previous trial's belief, with a
limited chain length ``n_samples`` and a ``Beta(beta, beta)`` prior that
regularizes the "which arm is best" posterior toward uniform.

Mechanism -> bias mapping (the point of the model):

* **warm start** (chain starts at the previous trial's final state) -> perseveration
* **limited N** (few mutation steps) -> anchoring / slow tracking of drift
* **mutation** (move one arm one level at a time) -> small incremental changes
* **Beta prior** (``log(count + beta)``) -> conservatism / bias toward indifferent choice

Reward scale
------------
The model is **scale-free**, matching the submitted agent
(:mod:`bayesd_misfits.causal_rw`).  Rewards are never assumed to live on a
known scale.  Each revealed reward is standardized causally -- using only the
running mean and sample standard deviation of rewards already revealed in the
current trajectory -- before it enters the likelihood.  The value levels are
therefore fixed *z-scores* (default ``{-2, -1, 0, 1, 2}``), and the observation
noise ``obs_sigma`` is in standardized units.  Adding a constant to all rewards
or multiplying them by a positive constant does not change the predicted
choices (up to floating-point precision).

Fitting note
------------
``prior_beta`` (the Beta-prior strength) enters the choice logits smoothly and
is differentiable, so it can be fit with HSSM gradient-based inference.  The
mutation chain itself depends on ``obs_sigma`` and ``n_samples`` only through
discrete accept/reject steps, so those are *hyperparameters* (constructor
arguments) to be selected by grid search / PDA, exactly as the BMS paper fits
its discrete chain length by grid search.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

try:
    import jax
    import jax.numpy as jnp
except ImportError:  # pragma: no cover
    jax = None
    jnp = None


def _default_level_values(n_levels: int) -> np.ndarray:
    """Symmetric z-score value levels (scale-free)."""
    return np.linspace(-2.0, 2.0, n_levels, dtype=np.float64)


def _standardize_numpy(
    reward: float, mean: float, m2: float, count: int
) -> float:
    """Causally standardize one reward using pre-reward running statistics.

    Returns 0.0 when fewer than two non-identical rewards have been revealed
    (no scale is yet identifiable), matching the submitted agent.
    """
    if count >= 2 and m2 > 0.0:
        sd = np.sqrt(m2 / (count - 1))
        if np.isfinite(sd) and sd > 0.0:
            return (reward - mean) / sd
    return 0.0


class CausalSamplingBandit:
    """NumPy forward model of the scale-free causal sampling bandit.

    Exposes an agent-like API (``reset`` / ``predict`` / ``update``) so it can
    be replayed over a trajectory and inspected directly, independent of HSSM.
    ``update`` accepts rewards in any units; they are standardized causally.
    """

    def __init__(
        self,
        n_actions: int = 4,
        n_levels: int = 5,
        level_values: Sequence[float] | None = None,
        n_samples: int = 10,
        n_particles: int = 32,
        beta: float = 1.0,
        obs_sigma: float = 1.0,
        forgetting: float = 1.0,
        temperature: float = 1.0,
        seed: int | None = None,
    ) -> None:
        if n_actions < 2:
            raise ValueError("n_actions must be at least 2")
        if n_levels < 2:
            raise ValueError("n_levels must be at least 2")
        if n_samples < 1:
            raise ValueError("n_samples must be at least 1")
        if n_particles < 1:
            raise ValueError("n_particles must be at least 1")
        if beta < 0.0:
            raise ValueError("beta must be non-negative")
        if obs_sigma <= 0.0:
            raise ValueError("obs_sigma must be positive")
        if not 0.0 < forgetting <= 1.0:
            raise ValueError("forgetting must be in (0, 1]")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        self._n_actions = n_actions
        self._n_levels = n_levels
        self._level_values = (
            np.asarray(level_values, dtype=np.float64)
            if level_values is not None
            else _default_level_values(n_levels)
        )
        if self._level_values.shape != (n_levels,):
            raise ValueError(
                f"level_values must have shape ({n_levels},), "
                f"got {self._level_values.shape}"
            )
        self._n_samples = n_samples
        self._n_particles = n_particles
        self._beta = beta
        self._obs_sigma = obs_sigma
        self._forgetting = forgetting
        self._temperature = temperature
        self._rng = np.random.default_rng(seed)
        self._state: dict[str, Any] | None = None
    # -- state ------------------------------------------------------------

    def reset(self) -> None:
        mid = self._n_levels // 2
        self._state = {
            "levels": np.full((self._n_particles, self._n_actions), mid, dtype=np.int64),
            "n": np.zeros(self._n_actions, dtype=np.float64),
            "S": np.zeros(self._n_actions, dtype=np.float64),
            "SS": np.zeros(self._n_actions, dtype=np.float64),
            "reward_count": 0,
            "reward_mean": 0.0,
            "reward_m2": 0.0,
            "counts": np.ones(self._n_actions, dtype=np.float64),
        }

    # -- likelihood -------------------------------------------------------

    def _log_likelihood(self, levels: np.ndarray) -> np.ndarray:
        """Log likelihood of the causally standardized history under levels (M, n_actions)."""
        safe_levels = np.clip(levels, 0, self._n_levels - 1)
        v = self._level_values[safe_levels]
        n = self._state["n"]
        S = self._state["S"]
        SS = self._state["SS"]
        resid = SS - 2.0 * v * S + n * v * v
        masked = np.where(n > 0.0, resid, 0.0)
        return -0.5 * np.sum(masked, axis=-1) / (self._obs_sigma**2)

    # -- mutation chain ---------------------------------------------------

    def _mutation_chain_ensemble(
        self, start_levels: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run M warm-started mutation chains in parallel (NumPy vectorized)."""
        levels = start_levels.copy()
        counts = np.zeros((self._n_particles, self._n_actions), dtype=np.float64)
        max_level = levels.max(axis=1, keepdims=True)
        is_best = levels == max_level
        counts += is_best / is_best.sum(axis=1, keepdims=True)

        curr_ll = self._log_likelihood(levels)
        idx = np.arange(self._n_particles)
        for _ in range(self._n_samples - 1):
            arms = self._rng.integers(0, self._n_actions, size=self._n_particles)
            directions = self._rng.choice([-1, 1], size=self._n_particles)
            proposed = levels.copy()
            proposed[idx, arms] += directions
            valid = (proposed[idx, arms] >= 0) & (proposed[idx, arms] < self._n_levels)

            prop_ll = self._log_likelihood(proposed)
            log_ratio = prop_ll - curr_ll
            accept = valid & (
                self._rng.random(size=self._n_particles)
                < np.exp(np.minimum(0.0, log_ratio))
            )

            levels = np.where(accept[:, None], proposed, levels)
            curr_ll = np.where(accept, prop_ll, curr_ll)

            max_level = levels.max(axis=1, keepdims=True)
            is_best = levels == max_level
            counts += is_best / is_best.sum(axis=1, keepdims=True)

        return levels, counts.mean(axis=0)

    # -- agent API --------------------------------------------------------

    def predict(self) -> np.ndarray:
        """Return choice probabilities for the current trial."""
        if self._state is None:
            raise RuntimeError("Call reset() before predict()")
        logits = self._temperature * np.log(
            self._state["counts"] + self._beta + 1e-12
        )
        logits -= logits.max()
        probs = np.exp(logits)
        return probs / probs.sum()

    def update(self, action: int, reward: float) -> None:
        """Apply one revealed outcome and advance the warm-started belief."""
        if self._state is None:
            raise RuntimeError("Call reset() before update()")
        action = int(action)
        reward = float(reward)

        # Causal standardization using pre-reward running statistics.
        z = _standardize_numpy(
            reward,
            self._state["reward_mean"],
            self._state["reward_m2"],
            self._state["reward_count"],
        )
        # Exponential forgetting: decay per-arm value statistics so recent
        # rewards dominate (the bandit drifts).  The running scale statistics
        # below are NOT decayed -- they only define the reward scale.
        self._state["n"] *= self._forgetting
        self._state["S"] *= self._forgetting
        self._state["SS"] *= self._forgetting
        self._state["n"][action] += 1.0
        self._state["S"][action] += z
        self._state["SS"][action] += z * z

        # Welford's stable one-pass running mean and variance update.
        count = self._state["reward_count"]
        delta = reward - self._state["reward_mean"]
        new_count = count + 1
        new_mean = self._state["reward_mean"] + delta / new_count
        self._state["reward_m2"] = max(
            self._state["reward_m2"] + delta * (reward - new_mean), 0.0
        )
        self._state["reward_mean"] = new_mean
        self._state["reward_count"] = new_count

        new_levels, avg_counts = self._mutation_chain_ensemble(
            self._state["levels"]
        )
        self._state["counts"] = avg_counts
        self._state["levels"] = new_levels


class NArmCausalSamplingBandit:
    """HSSM ``LearningProcess`` adapter for the scale-free causal sampling bandit.

    Emits categorical logits ``logit0..logit3`` for the direct-logit decision
    process in :mod:`bayesd_misfits.categorical_logits`.  The only free
    parameter is ``prior_beta`` (the Beta-prior strength); ``n_samples``,
    ``obs_sigma``, ``n_levels`` and ``level_values`` are hyperparameters.

    Rewards are standardized causally inside the learner, so the incoming
    ``feedback`` may be in any units (the HSSM pipeline's normalized [0, 1]
    column is fine).  The mutation chain is run once per trial inside
    ``update_jax`` (after the outcome is revealed) and its best-arm counts are
    stored in the state, so ``compute_jax`` only applies the differentiable
    ``log(count + prior_beta)`` mapping.  This mirrors the ``last_choice``
    pattern in the RW learners and keeps the chain's discrete accept/reject
    steps out of the gradient path.
    """

    def __init__(
        self,
        n_actions: int = 4,
        *,
        n_levels: int = 5,
        level_values: Sequence[float] | None = None,
        n_samples: int = 10,
        n_particles: int = 32,
        obs_sigma: float = 1.0,
        forgetting: float = 1.0,
        seed: int = 0,
        feedback_field: str = "feedback",
    ) -> None:
        if n_actions < 2:
            raise ValueError("n_actions must be at least 2")
        if n_levels < 2:
            raise ValueError("n_levels must be at least 2")
        if n_samples < 1:
            raise ValueError("n_samples must be at least 1")
        if n_particles < 1:
            raise ValueError("n_particles must be at least 1")
        if obs_sigma <= 0.0:
            raise ValueError("obs_sigma must be positive")
        if not 0.0 < forgetting <= 1.0:
            raise ValueError("forgetting must be in (0, 1]")
        self._n_actions = n_actions
        self._n_levels = n_levels
        self._level_values = (
            np.asarray(level_values, dtype=np.float64)
            if level_values is not None
            else _default_level_values(n_levels)
        )
        if self._level_values.shape != (n_levels,):
            raise ValueError(
                f"level_values must have shape ({n_levels},), "
                f"got {self._level_values.shape}"
            )
        self._n_samples = n_samples
        self._n_particles = n_particles
        self._obs_sigma = float(obs_sigma)
        self._forgetting = float(forgetting)
        self._seed = seed
        self._feedback_field = feedback_field
        self._state: dict[str, Any] | None = None
    # -- protocol properties ---------------------------------------------

    @property
    def n_actions(self) -> int:
        return self._n_actions

    @property
    def n_particles(self) -> int:
        return self._n_particles

    @property
    def computed_params(self) -> list[str]:
        return [f"logit{i}" for i in range(self._n_actions)]

    @property
    def free_params(self) -> list[str]:
        return ["prior_beta", "temperature"]

    @property
    def param_bounds(self) -> dict[str, tuple[float, float]]:
        return {"prior_beta": (0.0, 10.0), "temperature": (0.0, 15.0)}

    @property
    def default_params(self) -> dict[str, float]:
        return {"prior_beta": 1.0, "temperature": 1.0}

    @property
    def available_backends(self) -> tuple[str, ...]:
        return ("python", "jax")

    @property
    def supports_gradient(self) -> bool:
        return True

    @property
    def required_context_fields(self) -> list[str]:
        return ["choice", self._feedback_field]

    # -- state ------------------------------------------------------------

    def init_state(self) -> dict[str, Any]:
        mid = self._n_levels // 2
        return {
            "levels": np.full((self._n_particles, self._n_actions), mid, dtype=np.int64),
            "n": np.zeros(self._n_actions, dtype=np.float64),
            "S": np.zeros(self._n_actions, dtype=np.float64),
            "SS": np.zeros(self._n_actions, dtype=np.float64),
            "reward_count": 0,
            "reward_mean": 0.0,
            "reward_m2": 0.0,
            "counts": np.ones(self._n_actions, dtype=np.float64),
            "rng": np.random.default_rng(self._seed),
        }

    def init_jax_state(self) -> dict[str, Any]:
        if jnp is None:  # pragma: no cover
            raise RuntimeError("JAX is required for init_jax_state()")
        mid = self._n_levels // 2
        return {
            "levels": jnp.full((self._n_particles, self._n_actions), mid, dtype=jnp.int32),
            "n": jnp.zeros((self._n_actions,)),
            "S": jnp.zeros((self._n_actions,)),
            "SS": jnp.zeros((self._n_actions,)),
            "reward_count": jnp.asarray(0, dtype=jnp.int32),
            "reward_mean": jnp.asarray(0.0),
            "reward_m2": jnp.asarray(0.0),
            "counts": jnp.ones((self._n_actions,)),
            "key": jax.random.PRNGKey(self._seed),
        }

    def reset(self, **kwargs: Any) -> None:
        del kwargs
        self._state = self.init_state()

    # -- JAX backend ------------------------------------------------------

    def _log_likelihood_jax(self, levels: Any, n: Any, S: Any, SS: Any) -> Any:
        v = jnp.asarray(self._level_values)[levels]
        resid = SS - 2.0 * v * S + n * v * v
        return -0.5 * jnp.sum(jnp.where(n > 0.0, resid, 0.0)) / (
            self._obs_sigma**2
        )

    def _mutation_chain_jax(
        self, start_levels: Any, n: Any, S: Any, SS: Any, key: Any
    ) -> Any:
        """Run one mutation chain; returns ``(n_samples, n_actions)`` levels."""

        def step(levels, step_key):
            k1, k2, k3 = jax.random.split(step_key, 3)
            arm = jax.random.randint(k1, (), 0, self._n_actions)
            direction = jax.random.randint(k2, (), 0, 2) * 2 - 1  # -1 or +1
            proposed = levels.at[arm].add(direction)
            valid = (proposed[arm] >= 0) & (proposed[arm] < self._n_levels)
            log_ratio = self._log_likelihood_jax(
                proposed, n, S, SS
            ) - self._log_likelihood_jax(levels, n, S, SS)
            accept = valid & (
                jax.random.uniform(k3)
                < jnp.exp(jnp.minimum(0.0, log_ratio))
            )
            new_levels = jnp.where(accept, proposed, levels)
            return new_levels, new_levels

        keys = jax.random.split(key, self._n_samples - 1)
        _, chain = jax.lax.scan(step, start_levels, keys)
        return jnp.concatenate([start_levels[None, :], chain], axis=0)

    @staticmethod
    def _best_arm_counts_jax(chain: Any) -> Any:
        """Tie-split best-arm counts over a chain of level vectors."""
        max_level = chain.max(axis=1, keepdims=True)
        is_best = chain == max_level
        n_best = is_best.sum(axis=1, keepdims=True)
        weights = is_best / n_best
        return weights.sum(axis=0)

    def compute_jax(self, state, params, context):
        del context
        logits = jnp.asarray(params["temperature"]) * jnp.log(
            state["counts"] + jnp.asarray(params["prior_beta"]) + 1e-12
        )
        return {f"logit{i}": logits[i] for i in range(self._n_actions)}

    def update_jax(self, state, params, context):
        del params
        choice = jnp.asarray(context["choice"], dtype=jnp.int32)
        reward = jnp.asarray(context[self._feedback_field])

        # Causal standardization using pre-reward running statistics.
        count = state["reward_count"]
        mean = state["reward_mean"]
        m2 = state["reward_m2"]
        has_scale = (count >= 2) & (m2 > 0.0)
        sd = jnp.sqrt(jnp.maximum(m2, 0.0) / jnp.maximum(count - 1, 1))
        z = jnp.where(has_scale, (reward - mean) / jnp.maximum(sd, 1e-12), 0.0)

        # Exponential forgetting of per-arm value statistics.
        n = self._forgetting * state["n"].at[choice].add(1.0)
        S = self._forgetting * state["S"].at[choice].add(z)
        SS = self._forgetting * state["SS"].at[choice].add(z * z)

        # Welford update of the running reward statistics.
        new_count = count + 1
        delta = reward - mean
        new_mean = mean + delta / new_count
        new_m2 = jnp.maximum(m2 + delta * (reward - new_mean), 0.0)

        keys = jax.random.split(state["key"], self._n_particles + 1)
        particle_keys = keys[:self._n_particles]
        new_key = keys[-1]

        vmapped_chain = jax.vmap(
            self._mutation_chain_jax, in_axes=(0, None, None, None, 0)
        )
        all_chains = vmapped_chain(
            state["levels"], n, S, SS, particle_keys
        )
        vmapped_best_arm = jax.vmap(self._best_arm_counts_jax)
        all_counts = vmapped_best_arm(all_chains)
        counts = jnp.mean(all_counts, axis=0)
        new_levels = all_chains[:, -1, :]

        return {
            "levels": new_levels,
            "n": n,
            "S": S,
            "SS": SS,
            "reward_count": new_count,
            "reward_mean": new_mean,
            "reward_m2": new_m2,
            "counts": counts,
            "key": new_key,
        }

    # -- Python backend ---------------------------------------------------

    def compute_python(self, state, params, context):
        del context
        logits = float(params["temperature"]) * np.log(
            np.asarray(state["counts"]) + float(params["prior_beta"]) + 1e-12
        )
        return {f"logit{i}": float(logits[i]) for i in range(self._n_actions)}

    def update_python(self, state, params, context):
        del params
        choice = int(context["choice"])
        reward = float(context[self._feedback_field])

        z = _standardize_numpy(
            reward, state["reward_mean"], state["reward_m2"], state["reward_count"]
        )
        n = self._forgetting * np.asarray(state["n"], dtype=np.float64)
        S = self._forgetting * np.asarray(state["S"], dtype=np.float64)
        SS = self._forgetting * np.asarray(state["SS"], dtype=np.float64)
        n[choice] += 1.0
        S[choice] += z
        SS[choice] += z * z

        count = int(state["reward_count"])
        delta = reward - state["reward_mean"]
        new_count = count + 1
        new_mean = state["reward_mean"] + delta / new_count
        new_m2 = max(state["reward_m2"] + delta * (reward - new_mean), 0.0)

        levels = np.asarray(state["levels"], dtype=np.int64)
        rng = state["rng"]
        new_levels, avg_counts = self._mutation_chain_numpy(
            levels, n, S, SS, rng
        )
        return {
            "levels": new_levels,
            "n": n,
            "S": S,
            "SS": SS,
            "reward_count": new_count,
            "reward_mean": new_mean,
            "reward_m2": new_m2,
            "counts": avg_counts,
            "rng": rng,
        }

    def _log_likelihood_numpy(
        self, levels: np.ndarray, n: np.ndarray, S: np.ndarray, SS: np.ndarray
    ) -> np.ndarray:
        safe_levels = np.clip(levels, 0, self._n_levels - 1)
        v = self._level_values[safe_levels]
        resid = SS - 2.0 * v * S + n * v * v
        masked = np.where(n > 0.0, resid, 0.0)
        return -0.5 * np.sum(masked, axis=-1) / (self._obs_sigma**2)

    def _mutation_chain_numpy(
        self,
        start_levels: np.ndarray,
        n: np.ndarray,
        S: np.ndarray,
        SS: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        levels = start_levels.copy()
        counts = np.zeros((self._n_particles, self._n_actions), dtype=np.float64)
        max_level = levels.max(axis=1, keepdims=True)
        is_best = levels == max_level
        counts += is_best / is_best.sum(axis=1, keepdims=True)

        curr_ll = self._log_likelihood_numpy(levels, n, S, SS)
        idx = np.arange(self._n_particles)
        for _ in range(self._n_samples - 1):
            arms = rng.integers(0, self._n_actions, size=self._n_particles)
            directions = rng.choice([-1, 1], size=self._n_particles)
            proposed = levels.copy()
            proposed[idx, arms] += directions
            valid = (proposed[idx, arms] >= 0) & (proposed[idx, arms] < self._n_levels)

            prop_ll = self._log_likelihood_numpy(proposed, n, S, SS)
            log_ratio = prop_ll - curr_ll
            accept = valid & (
                rng.random(size=self._n_particles)
                < np.exp(np.minimum(0.0, log_ratio))
            )

            levels = np.where(accept[:, None], proposed, levels)
            curr_ll = np.where(accept, prop_ll, curr_ll)

            max_level = levels.max(axis=1, keepdims=True)
            is_best = levels == max_level
            counts += is_best / is_best.sum(axis=1, keepdims=True)

        return levels, counts.mean(axis=0)

    # -- convenience wrappers --------------------------------------------

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
class CausalSamplingBanditTrace(CausalSamplingBandit):
    """Forward model of the causal sampling bandit with an explicit choice trace.

    Adds a gradual motor perseveration trace:
    ``logit = temperature * log(counts + beta) + repetition_weight * choice_trace``
    where ``choice_trace`` updates toward the one-hot chosen action at
    ``choice_trace_rate``.
    """

    def __init__(
        self,
        n_actions: int = 4,
        n_levels: int = 5,
        level_values: Sequence[float] | None = None,
        n_samples: int = 10,
        n_particles: int = 32,
        beta: float = 1.0,
        obs_sigma: float = 1.0,
        forgetting: float = 1.0,
        temperature: float = 1.0,
        repetition_weight: float = 0.0,
        choice_trace_rate: float = 0.5,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            n_actions=n_actions,
            n_levels=n_levels,
            level_values=level_values,
            n_samples=n_samples,
            n_particles=n_particles,
            beta=beta,
            obs_sigma=obs_sigma,
            forgetting=forgetting,
            temperature=temperature,
            seed=seed,
        )
        if not 0.0 <= choice_trace_rate <= 1.0:
            raise ValueError("choice_trace_rate must be in [0, 1]")
        self._repetition_weight = float(repetition_weight)
        self._choice_trace_rate = float(choice_trace_rate)

    def reset(self) -> None:
        super().reset()
        self._state["choice_trace"] = np.zeros(self._n_actions, dtype=np.float64)

    def predict(self) -> np.ndarray:
        if self._state is None:
            raise RuntimeError("Call reset() before predict()")
        logits = (
            self._temperature * np.log(self._state["counts"] + self._beta + 1e-12)
            + self._repetition_weight * self._state["choice_trace"]
        )
        logits -= logits.max()
        probs = np.exp(logits)
        return probs / probs.sum()

    def update(self, action: int, reward: float) -> None:
        if self._state is None:
            raise RuntimeError("Call reset() before update()")
        super().update(action, reward)
        choice = int(action)
        target = np.zeros(self._n_actions, dtype=np.float64)
        target[choice] = 1.0
        trace = self._state["choice_trace"]
        self._state["choice_trace"] = trace + self._choice_trace_rate * (
            target - trace
        )


class NArmCausalSamplingBanditTrace(NArmCausalSamplingBandit):
    """HSSM ``LearningProcess`` adapter for Causal Sampling Bandit + Choice Trace.

    Combines the BMS-inspired causal sampling belief process with an explicit
    gradual choice trace.  Free parameters fit via hierarchical MCMC:
    * ``prior_beta`` -- Beta-prior strength (conservatism)
    * ``temperature`` -- Softmax inverse temperature on sampling counts
    * ``repetition_weight`` -- Scaling of the choice trace logit bonus
    * ``choice_trace_rate`` -- Decay / update rate of the choice trace
    """

    @property
    def free_params(self) -> list[str]:
        return [
            "prior_beta",
            "temperature",
            "repetition_weight",
            "choice_trace_rate",
        ]

    @property
    def param_bounds(self) -> dict[str, tuple[float, float]]:
        return {
            "prior_beta": (0.0, 10.0),
            "temperature": (0.0, 15.0),
            "repetition_weight": (-5.0, 5.0),
            "choice_trace_rate": (0.0, 1.0),
        }

    @property
    def default_params(self) -> dict[str, float]:
        return {
            "prior_beta": 1.0,
            "temperature": 1.0,
            "repetition_weight": 0.0,
            "choice_trace_rate": 0.5,
        }

    def init_state(self) -> dict[str, Any]:
        state = super().init_state()
        state["choice_trace"] = np.zeros(self._n_actions, dtype=np.float64)
        return state

    def init_jax_state(self) -> dict[str, Any]:
        state = super().init_jax_state()
        state["choice_trace"] = jnp.zeros((self._n_actions,))
        return state

    def compute_python(self, state, params, context):
        del context
        logits = (
            float(params["temperature"])
            * np.log(np.asarray(state["counts"]) + float(params["prior_beta"]) + 1e-12)
            + float(params["repetition_weight"])
            * np.asarray(state["choice_trace"], dtype=np.float64)
        )
        return {f"logit{i}": float(logits[i]) for i in range(self._n_actions)}

    def compute_jax(self, state, params, context):
        del context
        logits = (
            jnp.asarray(params["temperature"])
            * jnp.log(state["counts"] + jnp.asarray(params["prior_beta"]) + 1e-12)
            + jnp.asarray(params["repetition_weight"]) * state["choice_trace"]
        )
        return {f"logit{i}": logits[i] for i in range(self._n_actions)}

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
        target = (jnp.arange(self._n_actions) == choice).astype(
            state["choice_trace"].dtype
        )
        updated["choice_trace"] = (
            state["choice_trace"] + rate * (target - state["choice_trace"])
        )
        return updated
