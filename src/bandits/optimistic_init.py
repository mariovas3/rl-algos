import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributions as dists
from utils import (AvgRewardAndTrueOptTracker, EpsGreedyEnv,
                   get_normal_bandits, run_algo)


def plot_testbed(file_name, algos, avg_rewards, prop_true_optimal, init_vals):
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(7, 10))
    axs[0].set_ylabel("avg reward")
    axs[0].set_xlabel("Steps")
    axs[1].set_ylabel("prop optimal picked")
    axs[1].set_xlabel("Steps")
    for idx, algo in enumerate(algos):
        axs[0].plot(
            range(avg_rewards.shape[-1]),
            avg_rewards[idx, :],
            label=(
                f"$\epsilon$={1-algo.probs.item():.2f}, "
                f"$Q_0$={init_vals[idx]:.2f}"
            ),
        )
        axs[1].plot(
            range(prop_true_optimal.shape[-1]),
            prop_true_optimal[idx, :],
            label=(
                f"$\epsilon$={1-algo.probs.item():.2f}, "
                f"$Q_0$={init_vals[idx]:.2f}"
            ),
        )
        axs[0].legend()
        axs[1].legend()
    fig.tight_layout()
    plt.savefig(file_name)


if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)

    NUM_BANDITS, NUM_ARMS, NUM_STEPS = 2000, 10, 1000

    SAVE_PATH = Path(__file__).absolute().parent.parent.parent / "assets/imgs"
    if not SAVE_PATH.exists():
        SAVE_PATH.mkdir(parents=True)

    file_name = SAVE_PATH / "optimistic_init.png"

    # get 2000 bandit problems with 10 arms;
    bandits = get_normal_bandits(NUM_BANDITS, NUM_ARMS)

    # get idx of true optimal for each bandit;
    true_optimal = bandits.mean.argmax(-1).numpy()

    # eps = .1, .01, 0;
    algos = [dists.Bernoulli(probs=p) for p in (0.9, 1)]

    # init metric tracker;
    metric_tracker = AvgRewardAndTrueOptTracker(
        algos, NUM_STEPS, true_optimal
    )

    # run experiments;
    for idx, algo in enumerate(algos):
        # init action values at zero;
        action_values = np.zeros((NUM_BANDITS, NUM_ARMS)) + idx * 5
        bandit_env = EpsGreedyEnv(algo, action_values, lr=0.1)
        now = time.time()
        run_algo(idx, bandits, bandit_env, NUM_STEPS, metric_tracker)
        print(f"run {idx} took: {time.time() - now:.2f} secs")

    # save plot;
    plot_testbed(
        file_name,
        algos,
        metric_tracker.avg_rewards,
        metric_tracker.prop_true_optimal,
        init_vals=(0, 5),
    )
