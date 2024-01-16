import time
from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributions as dists
from utils import (AvgRewardAndTrueOptTracker, EpsGreedyEnv, UCBEnv,
                   get_normal_bandits, run_algo)


def plot_testbed(file_name, labels, avg_rewards, prop_true_optimal):
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(7, 10))
    axs[0].set_ylabel("avg reward")
    axs[0].set_xlabel("Steps")
    axs[1].set_ylabel("prop optimal picked")
    axs[1].set_xlabel("Steps")
    for idx, label in enumerate(labels):
        axs[0].plot(
            range(avg_rewards.shape[-1]),
            avg_rewards[idx, :],
            label=label,
        )
        axs[1].plot(
            range(prop_true_optimal.shape[-1]),
            prop_true_optimal[idx, :],
            label=label,
        )
        axs[0].legend()
        axs[1].legend()
    fig.tight_layout()
    plt.savefig(file_name)


if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)

    NUM_BANDITS, NUM_ARMS, NUM_STEPS = 2000, 10, 1000
    NUM_ALGOS = 3

    SAVE_PATH = Path(__file__).absolute().parent.parent.parent / "assets/imgs"
    if not SAVE_PATH.exists():
        SAVE_PATH.mkdir(parents=True)

    file_name = SAVE_PATH / "ucb-vs-epsgr.png"

    # get 2000 bandit problems with 10 arms;
    bandits = get_normal_bandits(NUM_BANDITS, NUM_ARMS)

    # get idx of true optimal for each bandit;
    true_optimal = bandits.mean.argmax(-1).numpy()

    # init metric tracker;
    metric_tracker = AvgRewardAndTrueOptTracker(
        NUM_ALGOS, NUM_STEPS, true_optimal
    )

    cs = [2, sqrt(2)]

    # run experiments;
    for idx in range(NUM_ALGOS):
        # init action values at zero;
        action_values = np.zeros((NUM_BANDITS, NUM_ARMS))
        if not idx:
            bandit_env = EpsGreedyEnv(dists.Bernoulli(0.9), action_values)
        else:
            bandit_env = UCBEnv(action_values, cs[idx - 1])
        now = time.time()
        run_algo(idx, bandits, bandit_env, NUM_STEPS, metric_tracker)
        print(f"run {idx} took: {time.time() - now:.2f} secs")

    # save plot;
    plot_testbed(
        file_name,
        (f"$\epsilon$=0.1", f"UCB, c=2", f"UCB, c=$\sqrt{{2}}$"),
        metric_tracker.avg_rewards,
        metric_tracker.prop_true_optimal,
    )
