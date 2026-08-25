"""Grid-search hyperparameters for the causal sampling bandit.

The mutation chain's discrete accept/reject steps are not differentiable, so
``n_samples`` and ``obs_sigma`` (and, for a forward-model sweep, ``beta``) are
selected by grid search over a small subject-disjoint split -- exactly as the
BMS paper grid-searches its chain length and Beta prior.

This uses the NumPy forward model (:class:`CausalSamplingBandit`), so it is
fast and needs no MCMC.  The result is a rough operating point to feed into
``research/run_comparison.py`` (which fits ``prior_beta`` with HSSM).

Run from the repo root:

    .venv/bin/python research/grid_search_causal_sampling.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bayesd_misfits.data import ensure_data_downloaded, load_challenge_data
from bayesd_misfits.causal_sampling_bandit import CausalSamplingBandit

N_TRIALS = 120


def load_split(
    n_train: int, n_valid: int, seed: int
) -> tuple[list[dict], list[dict]]:
    """Return subject-disjoint train/valid trajectories (raw rewards)."""
    ensure_data_downloaded()
    df = load_challenge_data(
        feedback_transform="raw",
        rt_placeholder=-1.0,
        group_by="trajectory",
    )
    counts = df.groupby("participant_id").size()
    complete = counts[counts == N_TRIALS].index
    df = df[df["participant_id"].isin(complete)].copy()

    rng = np.random.default_rng(seed)
    subjects = np.asarray(sorted(df["subject_id"].unique()))
    rng.shuffle(subjects)
    split_at = max(1, int(0.7 * len(subjects)))
    train_subjects = set(subjects[:split_at])
    valid_subjects = set(subjects[split_at:])

    def pick(subject_set: set[str], n: int) -> list[dict]:
        ids = df.loc[df["subject_id"].isin(subject_set), "participant_id"].unique()
        ids = rng.choice(ids, size=min(n, len(ids)), replace=False)
        sub = df[df["participant_id"].isin(ids)].copy()
        trajectories = []
        for _, traj in sub.groupby("participant_id", sort=True):
            traj = traj.sort_values("trial_id")
            trajectories.append(
                {
                    "actions": traj["response"].to_numpy(dtype=int),
                    "rewards": traj["feedback"].to_numpy(dtype=float),
                }
            )
        return trajectories

    return pick(train_subjects, n_train), pick(valid_subjects, n_valid)


def replay(model: CausalSamplingBandit, trajectories: list[dict]) -> dict[str, float]:
    """One-step-ahead NLL and accuracy over a set of trajectories."""
    total_nll = 0.0
    total_correct = 0
    total_trials = 0
    for traj in trajectories:
        model.reset()
        for action, reward in zip(
            traj["actions"], traj["rewards"], strict=True
        ):
            probs = model.predict()
            total_nll -= np.log(max(probs[action], 1e-12))
            total_correct += int(probs.argmax() == action)
            total_trials += 1
            model.update(action, reward)  # forward model standardizes causally
    return {
        "nll": total_nll / total_trials,
        "accuracy": total_correct / total_trials,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-valid", type=int, default=20)
    parser.add_argument("--n-particles", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args()
    _, valid = load_split(n_train=0, n_valid=args.n_valid, seed=args.seed)
    print(f"Grid search over {len(valid)} held-out trajectories "
          f"({len(valid) * N_TRIALS} trials)")

    grid = {
        "n_samples": [5, 10, 20],
        "obs_sigma": [0.1, 0.2, 0.4],
        "beta": [0.5, 1.0, 2.0, 5.0],
        "forgetting": [1.0, 0.98, 0.95, 0.9],
    }

    results = []
    for n_samples in grid["n_samples"]:
        for obs_sigma in grid["obs_sigma"]:
            for beta in grid["beta"]:
                for forgetting in grid["forgetting"]:
                    model = CausalSamplingBandit(
                        n_actions=4,
                        n_samples=n_samples,
                        n_particles=args.n_particles,
                        obs_sigma=obs_sigma,
                        beta=beta,
                        forgetting=forgetting,
                        seed=args.seed,
                    )
                    metrics = replay(model, valid)
                    results.append(
                        {
                            "n_samples": n_samples,
                            "obs_sigma": obs_sigma,
                            "beta": beta,
                            "forgetting": forgetting,
                            **metrics,
                        }
                    )

    results.sort(key=lambda r: r["nll"])
    print(f"\n{'n_samples':>10} {'obs_sigma':>10} {'beta':>6} "
          f"{'forget':>8} {'NLL':>9} {'acc':>7}")
    print("-" * 58)
    for r in results[:20]:
        print(
            f"{r['n_samples']:>10} {r['obs_sigma']:>10.2f} {r['beta']:>6.1f} "
            f"{r['forgetting']:>8.2f} {r['nll']:>9.4f} {r['accuracy']:>7.3f}"
        )

    best = results[0]
    print(
        f"\nBest: n_samples={best['n_samples']}, obs_sigma={best['obs_sigma']}, "
        f"beta={best['beta']}, forgetting={best['forgetting']} "
        f"-> NLL {best['nll']:.4f}, acc {best['accuracy']:.3f}"
    )
    print("(Uniform random NLL = %.4f)" % np.log(4))

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(
                {"grid": grid, "n_valid": len(valid), "results": results},
                f,
                indent=2,
            )
        print(f"Saved full grid to {out}")


if __name__ == "__main__":
    main()
