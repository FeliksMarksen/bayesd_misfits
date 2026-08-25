"""Reproduce the BMS paper's Figure 2 / Table 1 behavior.

Run from the repo root:

    .venv/bin/python research/reproduce_bms_figure2.py

Prints (a) the Mutation Sampler's Table 1 breakdown for the query
P(X1=1 | Y=1, X2=0) on the common-cause structure, and (b) the mean response
for each Figure 2 query under the MS (beta=0) and the BMS (beta=1), showing the
conservatism the Beta prior introduces.  Optionally writes a plot with
``--plot``.
"""

from __future__ import annotations

import argparse

import numpy as np

from bayesd_misfits.bms import (
    COMMON_CAUSE_FREQ,
    FIGURE2_QUERIES,
    normative_response,
    simulate_responses,
    table1_breakdown,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-chains", type=int, default=20000)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    n_chains = args.n_chains

    print("=" * 72)
    print("Table 1 reproduction (Mutation Sampler, beta=0)")
    print("Query: P(X1=1 | Y=1, X2=0), common-cause structure")
    print("=" * 72)
    event, given = FIGURE2_QUERIES["A: P(X1=1 | Y=1, X2=0)"]
    breakdown = table1_breakdown(
        COMMON_CAUSE_FREQ, event, given,
        chain_lens=[2, 6, 12, 24, 48], n_chains=n_chains, seed=0,
    )
    print(f"{'chain len':>10} {'computed':>10} {'0%':>8} {'50%':>8} {'100%':>8}")
    for n in (2, 6, 12, 24, 48):
        b = breakdown[n]
        print(
            f"{n:>10} {b['computed']:>10.3f} {b['0%']:>8.3f} "
            f"{b['50%']:>8.3f} {b['100%']:>8.3f}"
        )
    print()
    print("(Compare with the paper's Table 1: computed 0.00/0.06/0.18/0.42/0.73,"
          "\n 50% 0.93/0.67/0.39/0.15/0.02, 100% 0.06/0.21/0.33/0.35/0.22.)")
    print()

    print("=" * 72)
    print("Figure 2 queries: MS (beta=0) vs BMS (beta=1) mean responses")
    print("=" * 72)
    print(f"{'query':<26} {'normative':>10} {'MS mean':>9} {'BMS mean':>9}")
    for name, (ev, gv) in FIGURE2_QUERIES.items():
        norm = normative_response(COMMON_CAUSE_FREQ, ev, gv)
        rng = np.random.default_rng(0)
        ms = simulate_responses(
            COMMON_CAUSE_FREQ, ev, gv, beta=0.0,
            n_chains=n_chains, chain_len=12, rng=rng,
        )
        rng = np.random.default_rng(0)
        bms = simulate_responses(
            COMMON_CAUSE_FREQ, ev, gv, beta=1.0,
            n_chains=n_chains, chain_len=12, rng=rng,
        )
        print(f"{name:<26} {norm:>10.3f} {ms.mean():>9.3f} {bms.mean():>9.3f}")
    print()
    print("The BMS mean sits between the MS mean and 0.5 (conservatism), and the"
          "\nBeta prior removes the MS's 0%/100% spikes (see the test suite).")

    if args.plot:
        _plot_figure2(n_chains)


def _plot_figure2(n_chains: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (name, (ev, gv)) in zip(axes.ravel(), FIGURE2_QUERIES.items()):
        norm = normative_response(COMMON_CAUSE_FREQ, ev, gv)
        rng = np.random.default_rng(0)
        ms = simulate_responses(
            COMMON_CAUSE_FREQ, ev, gv, beta=0.0,
            n_chains=n_chains, chain_len=12, rng=rng,
        )
        rng = np.random.default_rng(0)
        bms = simulate_responses(
            COMMON_CAUSE_FREQ, ev, gv, beta=1.0,
            n_chains=n_chains, chain_len=12, rng=rng,
        )
        ax.hist(ms, bins=50, density=True, alpha=0.5, label="MS (beta=0)")
        ax.hist(bms, bins=50, density=True, alpha=0.5, label="BMS (beta=1)")
        ax.axvline(norm, color="g", linestyle="--", label="normative")
        ax.set_title(name)
        ax.set_xlim(-0.05, 1.05)
        ax.legend(fontsize=8)
    fig.suptitle("Bayesian Mutation Sampler: Figure 2 reproduction")
    fig.tight_layout()
    out = "research/bms_figure2_reproduction.png"
    fig.savefig(out, dpi=120)
    print(f"\nSaved plot to {out}")


if __name__ == "__main__":
    main()
