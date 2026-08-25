"""Replay one public trajectory through the official pyhgf gHGF adapter.

Example
-------
    .venv/bin/python research/replay_pyhgf.py --trajectory 0 \
        --output /tmp/pyhgf_replay.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bayesd_misfits.data import load_trajectories
from bayesd_misfits.pyhgf_bandit import replay_continuous_ghgf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=int, default=0)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("../hf_cache/public/public_train.jsonl"),
    )
    parser.add_argument("--output", type=Path, default=Path("/tmp/pyhgf_replay.png"))
    parser.add_argument("--beta", type=float, default=5.0)
    parser.add_argument("--sticky", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trajectories = load_trajectories(args.data)
    if not 0 <= args.trajectory < len(trajectories):
        raise IndexError(
            f"trajectory must be in [0, {len(trajectories) - 1}]"
        )
    trajectory = trajectories[args.trajectory]
    actions = np.asarray([trial["action"] for trial in trajectory["trials"]])
    rewards = np.asarray(
        [trial["reward"] / 100.0 for trial in trajectory["trials"]], dtype=float
    )
    replay = replay_continuous_ghgf(
        actions,
        rewards,
        beta=args.beta,
        sticky=args.sticky,
    )

    trials = np.arange(len(actions))
    colors = ("tab:blue", "tab:orange", "tab:green", "tab:red")
    figure, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    axes[0].scatter(trials, rewards, c=[colors[a] for a in actions], s=16)
    axes[0].set_ylabel("Observed reward")
    axes[0].set_ylim(0.0, 1.0)
    for action, color in enumerate(colors):
        mean = replay.prior_means[:, action]
        sd = np.sqrt(replay.prior_variances[:, action])
        axes[1].plot(trials, mean, color=color, label=f"arm {action}")
        axes[1].fill_between(
            trials,
            mean - sd,
            mean + sd,
            color=color,
            alpha=0.12,
        )
        axes[2].plot(
            trials,
            replay.action_probs[:, action],
            color=color,
            label=f"arm {action}",
        )
    axes[1].set_ylabel("Prior value belief")
    axes[1].legend(ncols=4)
    axes[2].set_ylabel("Choice probability")
    axes[2].set_xlabel("Trial")
    axes[2].set_ylim(0.0, 1.0)
    figure.suptitle(
        f"pyhgf replay: {trajectory['context']['trajectory_id']} "
        f"(beta={args.beta:g}, sticky={args.sticky:g})"
    )
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160)
    print(f"Saved {args.output}")
    print(
        "Mean chosen-action probability: "
        f"{replay.action_probs[np.arange(len(actions)), actions].mean():.4f}"
    )


if __name__ == "__main__":
    main()
