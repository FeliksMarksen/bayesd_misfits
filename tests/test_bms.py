"""Tests for the Bayesian Mutation Sampler port."""

from __future__ import annotations

import numpy as np
import pytest

from bayesd_misfits.bms import (
    CHAIN_FREQ,
    COMMON_CAUSE_FREQ,
    COMMON_EFFECT_FREQ,
    FIGURE2_QUERIES,
    N_STATES,
    X1,
    Y,
    X2,
    bits_to_state,
    ms_response_category,
    mutation_chain,
    mutation_chains_jax,
    neighbors,
    normative_response,
    normalize_joint,
    response_from_chain,
    simulate_responses,
    state_to_bits,
    table1_breakdown,
)


def test_state_bit_roundtrip():
    for state in range(N_STATES):
        assert bits_to_state(*state_to_bits(state)) == state


def test_state_encoding_matches_reference():
    # 0=111, 1=110, 2=101, 3=100, 4=011, 5=010, 6=001, 7=000
    expected = [
        (1, 1, 1),
        (1, 1, 0),
        (1, 0, 1),
        (1, 0, 0),
        (0, 1, 1),
        (0, 1, 0),
        (0, 0, 1),
        (0, 0, 0),
    ]
    for state, bits in enumerate(expected):
        assert state_to_bits(state) == bits


def test_neighbors_are_hamming_distance_one():
    for state in range(N_STATES):
        for nb in neighbors(state):
            a = state_to_bits(state)
            b = state_to_bits(nb)
            assert sum(x != y for x, y in zip(a, b, strict=True)) == 1
        assert len(set(neighbors(state))) == 3


def test_normalize_joint():
    joint = normalize_joint(CHAIN_FREQ)
    assert np.isclose(joint.sum(), 1.0)
    assert np.all(joint >= 0.0)


def test_canonical_joints_sum_to_32():
    for freq in (CHAIN_FREQ, COMMON_CAUSE_FREQ, COMMON_EFFECT_FREQ):
        assert np.isclose(freq.sum(), 32.0)


def test_normative_responses_figure2():
    # Hand-computed normative answers for the common cause structure.
    expected = {
        "A: P(X1=1 | Y=1, X2=0)": 0.75,
        "B: P(Y=1 | X1=1, X2=1)": 0.9,
        "C: P(X1=1 | Y=1)": 0.75,
        "D: P(X1=1 | Y=0, X2=0)": 0.25,
    }
    for name, (event, given) in FIGURE2_QUERIES.items():
        assert np.isclose(
            normative_response(COMMON_CAUSE_FREQ, event, given),
            expected[name],
            atol=1e-12,
        )


def test_mutation_chain_converges_to_peak():
    # A joint peaked at state 0 (111) should trap the chain there.
    joint = np.zeros(N_STATES)
    joint[0] = 1.0
    rng = np.random.default_rng(0)
    chain = mutation_chain(joint, start_state=0, n_steps=50, rng=rng)
    assert set(chain) == {0}


def test_mutation_chain_uniform_is_random_walk():
    joint = np.ones(N_STATES) / N_STATES
    rng = np.random.default_rng(0)
    chain = mutation_chain(joint, start_state=0, n_steps=200, rng=rng)
    # With a uniform joint every proposal is accepted, so the chain should
    # visit more than one state.
    assert len(set(chain)) > 1


def test_response_beta_zero_is_relative_frequency():
    # Chain visiting state 1 (110) twice and state 5 (010) once.
    # Query P(X1=1 | Y=1, X2=0): event state is 110, non-event is 010.
    chain = [1, 1, 5]
    event = (X1, 1)
    given = [(Y, 1), (X2, 0)]
    assert np.isclose(response_from_chain(chain, event, given, beta=0.0), 2 / 3)


def test_response_beta_pulls_toward_half():
    chain = [1, 1, 5]
    event = (X1, 1)
    given = [(Y, 1), (X2, 0)]
    raw = response_from_chain(chain, event, given, beta=0.0)
    regularized = response_from_chain(chain, event, given, beta=2.0)
    assert 0.5 < regularized < raw  # pulled toward 0.5 from above


def test_response_unvisited_states_default_to_half():
    # Neither required state visited -> 0/0 -> 0.5 for any beta.
    chain = [0, 0, 0]  # only 111, which does not match Y=1, X2=0
    event = (X1, 1)
    given = [(Y, 1), (X2, 0)]
    assert np.isclose(response_from_chain(chain, event, given, beta=0.0), 0.5)
    assert np.isclose(response_from_chain(chain, event, given, beta=1.0), 0.5)


def test_ms_response_categories():
    event = (X1, 1)
    given = [(Y, 1), (X2, 0)]
    # both states visited -> computed
    assert ms_response_category([1, 5], event, given) == "computed"
    # only event state (110) -> 100%
    assert ms_response_category([1, 1], event, given) == "100%"
    # only non-event state (010) -> 0%
    assert ms_response_category([5, 5], event, given) == "0%"
    # neither -> 50%
    assert ms_response_category([0, 0], event, given) == "50%"


def test_beta_removes_extreme_spikes():
    # With beta=0 the MS produces 0%/100% responses; with beta>0 the BMS
    # response is strictly interior.
    joint = COMMON_CAUSE_FREQ
    event, given = FIGURE2_QUERIES["A: P(X1=1 | Y=1, X2=0)"]
    rng = np.random.default_rng(0)
    ms = simulate_responses(joint, event, given, beta=0.0, n_chains=2000,
                            chain_len=12, rng=rng)
    rng = np.random.default_rng(0)
    bms = simulate_responses(joint, event, given, beta=1.0, n_chains=2000,
                             chain_len=12, rng=rng)
    assert np.any((ms == 0.0) | (ms == 1.0))
    assert np.all((bms > 0.0) & (bms < 1.0))


def test_table1_qualitative_pattern():
    # Reproduce the paper's Table 1 qualitative pattern: as chain length grows,
    # "computed" responses increase and "50%" responses decrease.
    event, given = FIGURE2_QUERIES["A: P(X1=1 | Y=1, X2=0)"]
    breakdown = table1_breakdown(
        COMMON_CAUSE_FREQ, event, given,
        chain_lens=[2, 6, 12, 24, 48], n_chains=2000, seed=0,
    )
    computed = [breakdown[n]["computed"] for n in (2, 6, 12, 24, 48)]
    half = [breakdown[n]["50%"] for n in (2, 6, 12, 24, 48)]
    assert computed == sorted(computed)  # monotonically increasing
    assert half == sorted(half, reverse=True)  # monotonically decreasing
    assert breakdown[2]["computed"] == 0.0  # 2 samples cannot visit both states


def test_jax_mutation_chains_valid_and_peaked():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    joint = np.zeros(N_STATES)
    joint[0] = 1.0
    start = jnp.zeros(10, dtype=jnp.int32)
    key = jax.random.PRNGKey(0)
    chains = mutation_chains_jax(joint, start, n_steps=20, key=key)
    assert chains.shape == (10, 20)
    assert bool(jnp.all(chains == 0))  # trapped at the peak
