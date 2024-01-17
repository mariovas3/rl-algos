import time
from pathlib import Path

import numpy as np
import torch
import torch.distributions as dists
from utils import (
    AvgRewardAndTrueOptTracker,
    SoftmaxPGEnv,
    get_normal_bandits,
    plot_testbed,
    run_algo,
)

if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)

    NUM_BANDITS, NUM_ARMS, NUM_STEPS = 2000, 10, 1000
    NUM_ALGOS = 4

    SAVE_PATH = Path(__file__).absolute().parent.parent.parent / "assets/imgs"
    if not SAVE_PATH.exists():
        SAVE_PATH.mkdir(parents=True)

    file_name = SAVE_PATH / "pg-bandits.png"

    # get 2000 bandit problems with 10 arms;
    bandits = get_normal_bandits(NUM_BANDITS, NUM_ARMS, mean=4.0)

    # get idx of true optimal for each bandit;
    true_optimal = bandits.mean.argmax(-1).numpy()

    # init metric tracker;
    metric_tracker = AvgRewardAndTrueOptTracker(
        NUM_ALGOS, NUM_STEPS, true_optimal
    )

    labels = []

    # run experiments;
    for idx in range(NUM_ALGOS):
        # init action values at zero;
        preferences = torch.zeros((NUM_BANDITS, NUM_ARMS))
        bandit_env = SoftmaxPGEnv(
            preferences, lr=0.1 + (idx % 2) * 0.3, with_baseline=idx < 2
        )
        labels.append(
            f"PG, lr={bandit_env.lr:.2f}, baseline={bandit_env.with_baseline}"
        )
        now = time.time()
        run_algo(idx, bandits, bandit_env, NUM_STEPS, metric_tracker)
        print(f"run {idx} took: {time.time() - now:.2f} secs")

    # save plot;
    plot_testbed(
        file_name,
        labels,
        metric_tracker.avg_rewards,
        metric_tracker.prop_true_optimal,
    )
