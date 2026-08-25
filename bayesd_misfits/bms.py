"""Bayesian Mutation Sampler (Kolvoort, Temme & van Maanen, 2023).

A clean, tested Python port of the BMS, kept deliberately separate from the
bandit adaptation in :mod:`bayesd_misfits.causal_sampling_bandit`.

The BMS is a *process model* of causal judgment.  A reasoner does not compute
the exact posterior over a causal Bayesian network; instead they run a
Metropolis-Hastings "mutation" chain over the network's discrete states and
estimate the queried probability from sample frequencies, regularized by a
symmetric ``Beta(beta, beta)`` prior.

Four mechanisms (and the biases they produce):

1. **Mutation sampling** -- propose flipping one variable at a time (small,
   incremental hypothesis changes).
2. **Biased starting point** -- chains start at a *prototype* (all-present
   ``111`` or all-absent ``000``), not at the normative answer.
3. **Limited chain length N** -- the resource budget; few samples mean the
   chain cannot reach distant states (anchoring).
4. **Beta(beta, beta) prior** -- the response is ``(count + beta) /
   (count_total + 2*beta)``, pulling judgments toward 50% (conservatism) and
   removing the Mutation Sampler's spurious 0%/100% spikes.

The module reproduces the paper's Figure 2 / Table 1 behavior: with ``beta=0``
(the original Mutation Sampler) responses concentrate at 0%, 50% and 100%;
with ``beta>0`` the distribution is smoothed toward the normative answer.

Reference
---------
Kolvoort, I. R., Temme, N., & van Maanen, L. (2023). The Bayesian Mutation
Sampler Explains Distributions of Causal Judgments. *Open Mind*, 7, 318-349.
https://doi.org/10.1162/opmi_a_00080
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


# ---------------------------------------------------------------------------
# State encoding
# ---------------------------------------------------------------------------
# Three binary variables (X1, Y, X2).  A state index 0..7 encodes the bits
# (X1, Y, X2) with 1 = "present":
#   0=111  1=110  2=101  3=100  4=011  5=010  6=001  7=000
# This matches the row order in Kolvoort's OSF code (rownames 111..000).

N_STATES = 8
N_VARS = 3

# Variable indices for queries.
X1, Y, X2 = 0, 1, 2


def state_to_bits(state: int) -> tuple[int, int, int]:
    """Return ``(X1, Y, X2)`` as 0/1 for a state index 0..7.

    State 0 is the all-present prototype ``111`` and state 7 is all-absent
    ``000``, matching the row order in Kolvoort's OSF code.
    """
    value = 7 - int(state)
    return ((value >> 2) & 1, (value >> 1) & 1, value & 1)


def bits_to_state(x1: int, y: int, x2: int) -> int:
    """Return the state index for the given ``(X1, Y, X2)`` bits."""
    value = (int(x1) << 2) | (int(y) << 1) | int(x2)
    return 7 - value


def neighbors(state: int) -> tuple[int, int, int]:
    """States reachable by flipping exactly one variable (Hamming distance 1)."""
    return (state ^ 1, state ^ 2, state ^ 4)


# ---------------------------------------------------------------------------
# Canonical joints (unnormalized frequencies from Kolvoort's OSF code).
# Order: 111, 110, 101, 100, 011, 010, 001, 000.
# ---------------------------------------------------------------------------

# Chain X1 -> Y -> X2 (base rates .5, effects .75/.25).  dagf1 in the R code.
CHAIN_FREQ = np.asarray([9, 3, 1, 3, 3, 1, 3, 9], dtype=np.float64)

# Common cause X1 <- Y -> X2 (base rates .5, effects .75/.25).  For these
# symmetric parameters the common-cause joint coincides with the chain joint,
# which is why Kolvoort's OSF code labels dagf1 "Common cause/Chain".
COMMON_CAUSE_FREQ = np.asarray([9, 3, 1, 3, 3, 1, 3, 9], dtype=np.float64)

# Common effect X1 -> Y <- X2 (noisy-OR: P(Y=1|00)=0, |10)=|01)=.5, |11)=.75).
COMMON_EFFECT_FREQ = np.asarray([6, 4, 2, 4, 4, 0, 4, 8], dtype=np.float64)


def normalize_joint(freq: Sequence[float]) -> np.ndarray:
    """Normalize an unnormalized joint frequency vector to probabilities."""
    freq = np.asarray(freq, dtype=np.float64)
    total = freq.sum()
    if total <= 0.0:
        raise ValueError("joint frequencies must sum to a positive number")
    return freq / total


# ---------------------------------------------------------------------------
# Mutation chain (NumPy)
# ---------------------------------------------------------------------------


def mutation_chain(
    joint: Sequence[float],
    start_state: int,
    n_steps: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Run one Metropolis-Hastings mutation chain.

    Parameters
    ----------
    joint : sequence of 8 floats
        Normalized joint probabilities over the 8 states.
    start_state : int
        Starting state (0..7).  Prototypes are 0 (111) and 7 (000).
    n_steps : int
        Chain length *including* the starting state (so ``n_steps - 1`` MH
        moves), matching the R reference where ``x[1] = startstate``.
    rng : np.random.Generator
        Random source.

    Returns
    -------
    np.ndarray of int, shape ``(n_steps,)``
        The sequence of visited states.
    """
    if n_steps < 1:
        raise ValueError("n_steps must be at least 1")
    joint = np.asarray(joint, dtype=np.float64)
    chain = np.empty(n_steps, dtype=np.int64)
    state = int(start_state)
    chain[0] = state
    for i in range(1, n_steps):
        proposed = int(rng.choice(neighbors(state)))
        acceptance = min(1.0, joint[proposed] / joint[state])
        if rng.random() < acceptance:
            state = proposed
        chain[i] = state
    return chain


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------


def _counts_for_query(
    chain: Sequence[int],
    event: tuple[int, int],
    given: Sequence[tuple[int, int]],
) -> tuple[float, float]:
    """Return ``(count_event, count_not_event)`` for a conditional query.

    ``event`` is ``(var_index, value)``; ``given`` is a sequence of
    ``(var_index, value)`` conditioning pairs.  Only states matching ``given``
    contribute; among them, states where the event variable equals the event
    value contribute to ``count_event`` and the rest to ``count_not_event``.
    """
    counts = np.bincount(np.asarray(chain, dtype=np.int64), minlength=N_STATES)
    ev_var, ev_val = event
    count_event = 0.0
    count_not = 0.0
    for state in range(N_STATES):
        bits = state_to_bits(state)
        if not all(bits[var] == val for var, val in given):
            continue
        if bits[ev_var] == ev_val:
            count_event += counts[state]
        else:
            count_not += counts[state]
    return float(count_event), float(count_not)


def response_from_chain(
    chain: Sequence[int],
    event: tuple[int, int],
    given: Sequence[tuple[int, int]],
    beta: float,
) -> float:
    """BMS response for ``P(event_var = event_val | given)``.

    ``response = (count_event + beta) / (count_event + count_not_event + 2*beta)``.

    With ``beta=0`` this is the original Mutation Sampler's relative frequency
    (with the 0/0 case returning 0.5 by convention).  With ``beta>0`` the
    estimate is pulled toward 0.5 and never reaches 0 or 1.
    """
    if beta < 0.0:
        raise ValueError("beta must be non-negative")
    count_event, count_not = _counts_for_query(chain, event, given)
    denominator = count_event + count_not + 2.0 * beta
    if denominator <= 0.0:
        return 0.5
    return (count_event + beta) / denominator


def ms_response_category(
    chain: Sequence[int],
    event: tuple[int, int],
    given: Sequence[tuple[int, int]],
) -> str:
    """Classify a Mutation Sampler (``beta=0``) response.

    Returns one of ``"computed"``, ``"0%"``, ``"50%"``, ``"100%"``, matching
    the paper's Table 1 breakdown:

    * ``computed`` -- both the event and non-event states were visited;
    * ``0%`` / ``100%`` -- only the non-event / event state was visited;
    * ``50%`` -- neither state was visited (the 0/0 default).
    """
    count_event, count_not = _counts_for_query(chain, event, given)
    if count_event > 0.0 and count_not > 0.0:
        return "computed"
    if count_event == 0.0 and count_not == 0.0:
        return "50%"
    if count_event == 0.0:
        return "0%"
    return "100%"


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


def simulate_responses(
    joint: Sequence[float],
    event: tuple[int, int],
    given: Sequence[tuple[int, int]],
    beta: float,
    n_chains: int,
    chain_len: int,
    rng: np.random.Generator,
    bias: float = 0.5,
) -> np.ndarray:
    """Simulate ``n_chains`` BMS responses for one query.

    Each chain starts at prototype 0 (111) with probability ``bias`` and at
    prototype 7 (000) otherwise, matching the R reference.
    """
    joint = normalize_joint(joint)
    responses = np.empty(n_chains, dtype=np.float64)
    for i in range(n_chains):
        start = 0 if rng.random() < bias else 7
        chain = mutation_chain(joint, start, chain_len, rng)
        responses[i] = response_from_chain(chain, event, given, beta)
    return responses


def table1_breakdown(
    joint: Sequence[float],
    event: tuple[int, int],
    given: Sequence[tuple[int, int]],
    chain_lens: Sequence[int],
    n_chains: int,
    seed: int = 0,
) -> dict[int, dict[str, float]]:
    """Reproduce the paper's Table 1 (Mutation Sampler, ``beta=0``).

    For each chain length, returns the probability of a ``computed``, ``0%``,
    ``50%`` or ``100%`` response.
    """
    joint = normalize_joint(joint)
    rng = np.random.default_rng(seed)
    out: dict[int, dict[str, float]] = {}
    for chain_len in chain_lens:
        categories = np.empty(n_chains, dtype=object)
        for i in range(n_chains):
            start = 0 if rng.random() < 0.5 else 7
            chain = mutation_chain(joint, start, chain_len, rng)
            categories[i] = ms_response_category(chain, event, given)
        values, counts = np.unique(categories, return_counts=True)
        probs = {str(v): 0.0 for v in ("computed", "0%", "50%", "100%")}
        for v, c in zip(values, counts, strict=True):
            probs[str(v)] = c / n_chains
        out[chain_len] = probs
    return out


# ---------------------------------------------------------------------------
# JAX vectorized mutation chains
# ---------------------------------------------------------------------------


def mutation_chains_jax(
    joint: Sequence[float],
    start_states: Any,
    n_steps: int,
    key: Any,
) -> Any:
    """Run many mutation chains in parallel (JAX).

    Parameters
    ----------
    joint : sequence of 8 floats
        Normalized joint probabilities.
    start_states : jax array of int, shape ``(n_chains,)``
        Starting state per chain.
    n_steps : int
        Chain length including the starting state.
    key : jax.random.PRNGKey

    Returns
    -------
    jax array of int, shape ``(n_chains, n_steps)``
    """
    if jnp is None:  # pragma: no cover
        raise ImportError("JAX is required for mutation_chains_jax()")
    joint = jnp.asarray(joint, dtype=jnp.float32)
    start_states = jnp.asarray(start_states, dtype=jnp.int32)
    n_chains = start_states.shape[0]

    def step(state, step_key):
        k1, k2 = jax.random.split(step_key)
        neighbor_idx = jax.random.randint(k1, (n_chains,), 0, 3)
        proposed = state ^ (1 << neighbor_idx)  # 1, 2, 4
        acceptance = jnp.minimum(1.0, joint[proposed] / joint[state])
        accept = jax.random.uniform(k2, (n_chains,)) < acceptance
        new_state = jnp.where(accept, proposed, state)
        return new_state, new_state

    keys = jax.random.split(key, n_steps - 1)
    _, chain = jax.lax.scan(step, start_states, keys)  # (n_steps-1, n_chains)
    chain = jnp.concatenate([start_states[None, :], chain], axis=0)
    return chain.T  # (n_chains, n_steps)


# ---------------------------------------------------------------------------
# Figure 2 queries (common cause structure)
# ---------------------------------------------------------------------------

# The four inferences plotted in the paper's Figure 2, as (event, given).
FIGURE2_QUERIES: dict[str, tuple[tuple[int, int], list[tuple[int, int]]]] = {
    "A: P(X1=1 | Y=1, X2=0)": ((X1, 1), [(Y, 1), (X2, 0)]),
    "B: P(Y=1 | X1=1, X2=1)": ((Y, 1), [(X1, 1), (X2, 1)]),
    "C: P(X1=1 | Y=1)": ((X1, 1), [(Y, 1)]),
    "D: P(X1=1 | Y=0, X2=0)": ((X1, 1), [(Y, 0), (X2, 0)]),
}


def normative_response(
    joint: Sequence[float],
    event: tuple[int, int],
    given: Sequence[tuple[int, int]],
) -> float:
    """Exact CBN answer for a query (the green dashed line in Figure 2)."""
    joint = normalize_joint(joint)
    ev_var, ev_val = event
    num = 0.0
    den = 0.0
    for state in range(N_STATES):
        bits = state_to_bits(state)
        if not all(bits[var] == val for var, val in given):
            continue
        den += joint[state]
        if bits[ev_var] == ev_val:
            num += joint[state]
    return num / den if den > 0.0 else 0.5
