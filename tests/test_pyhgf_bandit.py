"""Correctness tests for the official-pyhgf bandit adapter."""

from __future__ import annotations

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from bayesd_misfits.pyhgf_bandit import (
    ContinuousGHGFConfig,
    NArmPyHGF,
    NArmPyHGFSticky,
    NArmPyHGFUncertainty,
    NArmPyHGFUncertaintySticky,
    OnlineContinuousGHGF,
    build_continuous_ghgf,
    encode_partial_feedback,
    replay_continuous_ghgf,
)


CHOICES = np.asarray([0, 1, 0, 3, 2, 0, 1, 3], dtype=int)
REWARDS = np.asarray([0.8, 0.2, 0.7, 0.6, 0.5, 0.1, 0.9, 0.4])


def test_network_has_four_independent_three_node_branches():
    network, observations, values, volatilities = build_continuous_ghgf()
    assert network.n_nodes == 12
    assert observations == (0, 1, 2, 3)
    assert values == (4, 5, 6, 7)
    assert volatilities == (8, 9, 10, 11)
    for action in range(4):
        assert network.edges[observations[action]].value_parents == (values[action],)
        assert network.edges[values[action]].value_children == (observations[action],)
        assert network.edges[values[action]].volatility_parents == (
            volatilities[action],
        )
        assert network.edges[volatilities[action]].volatility_children == (
            values[action],
        )


def test_partial_feedback_encoding_observes_exactly_the_chosen_arm():
    values, observed = encode_partial_feedback(CHOICES, REWARDS)
    np.testing.assert_array_equal(observed.sum(axis=1), 1)
    np.testing.assert_allclose(values[np.arange(len(CHOICES)), CHOICES], REWARDS)
    np.testing.assert_allclose(values[observed == 0], 0.0)


def test_invalid_rewards_and_actions_are_rejected():
    with pytest.raises(ValueError, match="normalized"):
        encode_partial_feedback([0], [50.0])
    with pytest.raises(ValueError, match="actions must be in"):
        encode_partial_feedback([4], [0.5])
    with pytest.raises(ValueError, match="must not be empty"):
        encode_partial_feedback([], [])


def test_first_update_matches_pinned_pyhgf_reference_values():
    replay = replay_continuous_ghgf([0], [0.8])
    np.testing.assert_allclose(replay.prior_means[0], 0.5, atol=1e-7)
    np.testing.assert_allclose(
        replay.posterior_means[0],
        [0.75180125, 0.5, 0.5, 0.5],
        rtol=2e-6,
        atol=2e-7,
    )
    np.testing.assert_array_equal(replay.observed[0], [1, 0, 0, 0])


def test_unchosen_means_stay_fixed_while_uncertainty_grows():
    replay = replay_continuous_ghgf([0, 0, 0], [0.8, 0.7, 0.6])
    np.testing.assert_allclose(replay.posterior_means[:, 1:], 0.5, atol=1e-7)
    assert np.all(np.diff(replay.prior_variances[:, 1], axis=0) > 0.0)


def test_online_updates_match_batch_replay_one_step_ahead():
    batch = replay_continuous_ghgf(CHOICES, REWARDS)
    online = OnlineContinuousGHGF()
    means = []
    variances = []
    for action, reward in zip(CHOICES, REWARDS, strict=True):
        prior_mean, prior_variance = online.predict()
        means.append(prior_mean)
        variances.append(prior_variance)
        online.update(int(action), float(reward))
    np.testing.assert_allclose(means, batch.prior_means, rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(
        variances, batch.prior_variances, rtol=2e-6, atol=2e-7
    )


def test_choice_probabilities_are_pre_outcome_and_normalized():
    replay = replay_continuous_ghgf(CHOICES, REWARDS, beta=5.0, sticky=0.2)
    np.testing.assert_allclose(replay.action_probs.sum(axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(replay.action_probs[0], 0.25, atol=1e-12)
    # Trial 2 repeats arm 0, so stickiness raises its probability relative to
    # an equally valued unchosen arm.
    assert replay.action_probs[1, 0] > replay.action_probs[1, 2]


def test_configuration_validation_rejects_nonpositive_precision():
    with pytest.raises(ValueError, match="sensory_precision"):
        build_continuous_ghgf(ContinuousGHGFConfig(sensory_precision=0.0))


def test_ssms_learner_matches_stateful_official_pyhgf_predictions():
    learner = NArmPyHGF()
    params = {"ghgf_omega": jnp.asarray(-4.0)}
    state = learner.init_jax_state()
    reference = OnlineContinuousGHGF()

    learner_means = []
    reference_means = []
    for action, reward in zip(CHOICES, REWARDS, strict=True):
        computed = learner.compute_jax(state, params, context={})
        learner_means.append([computed[f"q{i}"] for i in range(4)])
        reference_means.append(reference.predict()[0])
        state = learner.update_jax(
            state,
            params,
            context={"choice": jnp.asarray(action), "feedback": jnp.asarray(reward)},
        )
        reference.update(int(action), float(reward))

    np.testing.assert_allclose(
        learner_means, reference_means, rtol=2e-6, atol=2e-7
    )


def test_sticky_learner_is_nested_and_only_offsets_the_previous_choice():
    base = NArmPyHGF()
    sticky = NArmPyHGFSticky()
    base_params = {"ghgf_omega": jnp.asarray(-4.0)}
    sticky_params = {
        "ghgf_omega": jnp.asarray(-4.0),
        "sticky": jnp.asarray(0.3),
    }
    base_state = base.init_jax_state()
    sticky_state = sticky.init_jax_state()
    context = {"choice": jnp.asarray(2), "feedback": jnp.asarray(0.7)}
    base_state = base.update_jax(base_state, base_params, context)
    sticky_state = sticky.update_jax(sticky_state, sticky_params, context)
    base_values = base.compute_jax(base_state, base_params, context={})
    sticky_values = sticky.compute_jax(sticky_state, sticky_params, context={})

    offsets = np.asarray([
        sticky_values[f"q{i}"] - base_values[f"q{i}"] for i in range(4)
    ])
    np.testing.assert_allclose(offsets, [0.0, 0.0, 0.3, 0.0], atol=2e-7)


def test_ssms_learner_is_jittable_and_differentiable_in_tonic_volatility():
    learner = NArmPyHGF()
    choices = jnp.asarray(CHOICES, dtype=jnp.int32)
    rewards = jnp.asarray(REWARDS)

    def negative_log_likelihood(omega):
        params = {"ghgf_omega": omega}

        def trial(state, observation):
            choice, reward = observation
            computed = learner.compute_jax(state, params, context={})
            values = jnp.stack([computed[f"q{i}"] for i in range(4)])
            log_probability = jax.nn.log_softmax(5.0 * values)[choice]
            state = learner.update_jax(
                state,
                params,
                context={"choice": choice, "feedback": reward},
            )
            return state, log_probability

        _, log_probabilities = jax.lax.scan(
            trial,
            learner.init_jax_state(),
            (choices, rewards),
        )
        return -jnp.sum(log_probabilities)

    objective = jax.jit(negative_log_likelihood)(jnp.asarray(-4.0))
    gradient = jax.grad(negative_log_likelihood)(jnp.asarray(-4.0))
    assert bool(jnp.isfinite(objective))
    assert bool(jnp.isfinite(gradient))
    assert float(jnp.abs(gradient)) > 0.0


def test_ssms_protocol_exposes_only_the_intended_initial_free_parameters():
    base = NArmPyHGF()
    sticky = NArmPyHGFSticky()
    assert base.available_backends == ("jax",)
    assert base.free_params == ["ghgf_omega"]
    assert sticky.free_params == ["ghgf_omega", "sticky"]
    assert base.param_bounds == {"ghgf_omega": (-8.0, 0.0)}
    assert set(base.default_params) == {"ghgf_omega"}


def test_zero_uncertainty_weight_reduces_to_value_logits():
    learner = NArmPyHGFUncertainty()
    params = {
        "ghgf_omega": jnp.asarray(-4.0),
        "beta": jnp.asarray(3.0),
        "uncertainty_weight": jnp.asarray(0.0),
    }
    state = learner.init_jax_state()
    state = learner.update_jax(
        state,
        params,
        context={"choice": jnp.asarray(0), "feedback": jnp.asarray(0.8)},
    )
    means, _ = learner.predict_moments_jax(state, params)
    computed = learner.compute_jax(state, params, context={})
    logits = jnp.stack([computed[f"logit{i}"] for i in range(4)])
    np.testing.assert_allclose(logits, 3.0 * means, rtol=1e-6, atol=1e-7)


def test_positive_uncertainty_weight_prefers_more_uncertain_unchosen_arm():
    learner = NArmPyHGFUncertainty()
    params = {
        "ghgf_omega": jnp.asarray(-4.0),
        "beta": jnp.asarray(0.0),
        "uncertainty_weight": jnp.asarray(2.0),
    }
    state = learner.init_jax_state()
    state = learner.update_jax(
        state,
        params,
        context={"choice": jnp.asarray(0), "feedback": jnp.asarray(0.8)},
    )
    _, standard_deviations = learner.predict_moments_jax(state, params)
    computed = learner.compute_jax(state, params, context={})
    assert float(standard_deviations[1]) > float(standard_deviations[0])
    assert float(computed["logit1"]) > float(computed["logit0"])


def test_repetition_logit_is_independent_of_beta_and_uncertainty():
    learner = NArmPyHGFUncertaintySticky()
    params = {
        "ghgf_omega": jnp.asarray(-4.0),
        "beta": jnp.asarray(0.0),
        "uncertainty_weight": jnp.asarray(0.0),
        "repetition_weight": jnp.asarray(1.25),
    }
    state = learner.init_jax_state()
    state = learner.update_jax(
        state,
        params,
        context={"choice": jnp.asarray(3), "feedback": jnp.asarray(0.5)},
    )
    computed = learner.compute_jax(state, params, context={})
    logits = np.asarray([computed[f"logit{i}"] for i in range(4)])
    np.testing.assert_allclose(logits, [0.0, 0.0, 0.0, 1.25], atol=1e-7)
