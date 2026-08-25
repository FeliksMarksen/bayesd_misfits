"""Tests for the causal sampling bandit (BMS-inspired bandit learner)."""

from __future__ import annotations

import numpy as np
import pytest
from bayesd_misfits.causal_sampling_bandit import (
    CausalSamplingBandit,
    CausalSamplingBanditTrace,
    NArmCausalSamplingBandit,
    NArmCausalSamplingBanditTrace,
)

def _softmax(logits):
    logits = np.asarray(logits, dtype=float)
    logits -= logits.max()
    probs = np.exp(logits)
    return probs / probs.sum()


def test_forward_model_predict_is_valid_distribution():
    model = CausalSamplingBandit(n_actions=4, seed=0)
    model.reset()
    probs = model.predict()
    assert probs.shape == (4,)
    assert np.all(probs >= 0.0)
    assert np.isclose(probs.sum(), 1.0)


def test_forward_model_learns_to_prefer_high_reward_arm():
    model = CausalSamplingBandit(n_actions=4, n_samples=20, seed=1)
    model.reset()
    # Arm 2 always pays 90, others pay 10.
    for _ in range(60):
        model.update(2, 90.0)
        model.update(0, 10.0)
        model.update(1, 10.0)
        model.update(3, 10.0)
    probs = model.predict()
    assert probs.argmax() == 2


def test_forward_model_perseveration_from_warm_start():
    # The warm start + limited N should produce a non-uniform (perseverative)
    # choice distribution once the model has learned an arm advantage.
    model = CausalSamplingBandit(n_actions=4, n_samples=5, beta=0.5, seed=2)
    model.reset()
    for h, l in zip((80.0, 85.0, 75.0, 90.0), (20.0, 15.0, 25.0, 10.0)):
        model.update(0, h)
        model.update(1, l)
    probs = model.predict()
    assert probs.max() > 0.25  # non-uniform choice


def test_forward_model_beta_prior_induces_conservatism():
    # A large beta pulls the choice toward uniform (indifferent) even when one
    # arm is clearly best.
    strong = CausalSamplingBandit(n_actions=4, n_samples=20, beta=0.0, seed=3)
    weak = CausalSamplingBandit(n_actions=4, n_samples=20, beta=50.0, seed=3)
    strong.reset()
    weak.reset()
    for h, l in zip((90.0, 85.0, 95.0, 88.0), (10.0, 15.0, 5.0, 12.0)):
        strong.update(0, h)
        strong.update(1, l)
        weak.update(0, h)
        weak.update(1, l)
    p_strong = strong.predict()
    p_weak = weak.predict()
    assert p_strong[0] > p_weak[0]  # strong prior flattens the preference


def test_hssm_protocol_properties():
    learner = NArmCausalSamplingBandit(n_actions=4, n_particles=16)
    assert learner.n_particles == 16
    assert learner.computed_params == ["logit0", "logit1", "logit2", "logit3"]
    assert learner.free_params == ["prior_beta", "temperature"]
    assert learner.param_bounds["prior_beta"] == (0.0, 10.0)
    assert learner.param_bounds["temperature"] == (0.0, 15.0)
    assert learner.available_backends == ("python", "jax")
    assert learner.supports_gradient is True
    assert learner.required_context_fields == ["choice", "feedback"]


def test_python_backend_emits_logits_and_updates():
    learner = NArmCausalSamplingBandit(n_actions=4, n_samples=10, seed=0)
    learner.reset()
    state = learner.init_state()
    logits = learner.compute_python(state, {"prior_beta": 1.0, "temperature": 1.0}, context={})
    assert set(logits) == {"logit0", "logit1", "logit2", "logit3"}
    # Initial state: uniform counts -> uniform logits.
    assert np.allclose(list(logits.values()), logits["logit0"])
    # Update with a reward and check the state advances.
    new_state = learner.update_python(
        state, {"prior_beta": 1.0, "temperature": 1.0}, context={"choice": 0, "feedback": 0.9}
    )
    assert new_state["n"][0] == 1.0
    assert new_state["counts"].shape == (4,)
    assert new_state["levels"].shape == (learner.n_particles, 4)

def test_jax_backend_matches_python_backend_shape_and_validity():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    learner = NArmCausalSamplingBandit(n_actions=4, n_samples=10, seed=0)
    state = learner.init_jax_state()
    logits = learner.compute_jax(state, {"prior_beta": 1.0, "temperature": 1.0}, context={})
    assert set(logits) == {"logit0", "logit1", "logit2", "logit3"}
    for v in logits.values():
        assert bool(jnp.isfinite(v))
    new_state = learner.update_jax(
        state, {"prior_beta": 1.0, "temperature": 1.0}, context={"choice": 0, "feedback": 0.9}
    )
    assert new_state["counts"].shape == (4,)
    assert new_state["levels"].shape == (learner.n_particles, 4)

def test_jax_chain_learns_high_reward_arm():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    learner = NArmCausalSamplingBandit(n_actions=4, n_samples=20, seed=1)
    state = learner.init_jax_state()
    for h, l in zip((0.9, 0.85, 0.95, 0.88, 0.92), (0.1, 0.15, 0.05, 0.12, 0.08)):
        state = learner.update_jax(
            state, {"prior_beta": 1.0, "temperature": 1.0}, context={"choice": 0, "feedback": h}
        )
        state = learner.update_jax(
            state, {"prior_beta": 1.0, "temperature": 1.0}, context={"choice": 1, "feedback": l}
        )
    logits = learner.compute_jax(state, {"prior_beta": 1.0, "temperature": 1.0}, context={})
    values = np.asarray([float(logits[f"logit{i}"]) for i in range(4)])
    assert values.argmax() == 0


def test_scale_invariance():
    # Choices must be invariant to positive affine reward transformations.
    def run(rewards_by_arm, transform):
        model = CausalSamplingBandit(n_actions=4, n_samples=10, beta=1.0, seed=0)
        model.reset()
        for arm, rs in rewards_by_arm.items():
            for r in rs:
                model.update(arm, transform(r))
        return model.predict()

    rewards = {
        0: (90.0, 85.0, 95.0),
        1: (10.0, 15.0, 5.0),
        2: (50.0, 55.0, 45.0),
        3: (30.0, 25.0, 35.0),
    }
    p_raw = run(rewards, lambda r: r)
    p_affine = run(rewards, lambda r: 10.0 * r + 500.0)
    assert np.allclose(p_raw, p_affine, atol=1e-6)
def test_n_particles_validation():
    with pytest.raises(ValueError):
        CausalSamplingBandit(n_actions=4, n_particles=0)
    with pytest.raises(ValueError):
        NArmCausalSamplingBandit(n_actions=4, n_particles=0)


def test_level_values_validation():
    with pytest.raises(ValueError):
        NArmCausalSamplingBandit(n_actions=4, level_values=[0.1, 0.5])
    with pytest.raises(ValueError):
        NArmCausalSamplingBandit(n_actions=4, n_samples=0)
    with pytest.raises(ValueError):
        NArmCausalSamplingBandit(n_actions=4, obs_sigma=0.0)
def test_causal_sampling_bandit_trace_model():
    model = CausalSamplingBanditTrace(
        n_actions=4,
        n_samples=5,
        n_particles=16,
        beta=1.0,
        repetition_weight=2.0,
        choice_trace_rate=0.4,
        seed=42,
    )
    model.reset()
    assert np.allclose(model._state["choice_trace"], 0.0)

    # Choice trace updates with actions
    model.update(action=1, reward=50.0)
    assert np.isclose(model._state["choice_trace"][1], 0.4)
    assert np.allclose(model._state["choice_trace"][[0, 2, 3]], 0.0)

    probs = model.predict()
    assert probs.shape == (4,)
    assert probs[1] > probs[0]  # Trace boosts chosen arm


def test_hssm_causal_sampling_bandit_trace_protocol():
    learner = NArmCausalSamplingBanditTrace(n_actions=4, n_particles=16)
    assert learner.free_params == [
        "prior_beta",
        "temperature",
        "repetition_weight",
        "choice_trace_rate",
    ]
    params = {
        "prior_beta": 1.0,
        "temperature": 1.0,
        "repetition_weight": 1.5,
        "choice_trace_rate": 0.3,
    }
    state = learner.init_state()
    assert "choice_trace" in state
    new_state = learner.update_python(
        state, params, context={"choice": 2, "feedback": 0.8}
    )
    assert new_state["choice_trace"][2] == 0.3
    logits = learner.compute_python(new_state, params, context={})
    assert set(logits.keys()) == {"logit0", "logit1", "logit2", "logit3"}


def test_jax_causal_sampling_bandit_trace_backend():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    learner = NArmCausalSamplingBanditTrace(n_actions=4, n_particles=16)
    params = {
        "prior_beta": 1.0,
        "temperature": 1.0,
        "repetition_weight": 1.5,
        "choice_trace_rate": 0.3,
    }
    state = learner.init_jax_state()
    assert "choice_trace" in state
    new_state = learner.update_jax(
        state, params, context={"choice": 2, "feedback": 0.8}
    )
    assert float(new_state["choice_trace"][2]) == pytest.approx(0.3)
    logits = learner.compute_jax(new_state, params, context={})
    assert set(logits.keys()) == {"logit0", "logit1", "logit2", "logit3"}
    for v in logits.values():
        assert bool(jnp.isfinite(v))
