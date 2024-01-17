import time
from math import sqrt
from pathlib import Path

import numpy as np
import torch
import torch.distributions as dists
from utils import (
    AvgRewardAndTrueOptTracker,
    EpsGreedyEnv,
    SoftmaxPGEnv,
    UCBEnv,
    get_normal_bandits,
    plot_testbed,
    run_algo,
)

if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)

    NUM_BANDITS, NUM_ARMS, NUM_STEPS = 2000, 10, 1000

    SAVE_PATH = Path(__file__).absolute().parent.parent.parent / "assets/imgs"
    if not SAVE_PATH.exists():
        SAVE_PATH.mkdir(parents=True)

    file_name = SAVE_PATH / "all-algos-0-mean.png"

    # get 2000 bandit problems with 10 arms;
    bandits = get_normal_bandits(NUM_BANDITS, NUM_ARMS, mean=0.0)

    # get idx of true optimal for each bandit;
    true_optimal = bandits.mean.argmax(-1).numpy()

    algos = (
        EpsGreedyEnv(
            algo=dists.Bernoulli(probs=1),
            action_values=np.zeros((NUM_BANDITS, NUM_ARMS)),
        ),
        EpsGreedyEnv(
            algo=dists.Bernoulli(probs=0.9),
            action_values=np.zeros((NUM_BANDITS, NUM_ARMS)),
        ),
        UCBEnv(
            action_values=np.zeros((NUM_BANDITS, NUM_ARMS)),
            c=sqrt(2),
        ),
        SoftmaxPGEnv(
            preferences=torch.zeros((NUM_BANDITS, NUM_ARMS)),
            lr=0.1,
            with_baseline=True,
        ),
        SoftmaxPGEnv(
            preferences=torch.zeros((NUM_BANDITS, NUM_ARMS)),
            lr=0.1,
            with_baseline=False,
        ),
    )

    # init metric tracker;
    metric_tracker = AvgRewardAndTrueOptTracker(
        len(algos), NUM_STEPS, true_optimal
    )

    # run experiments;
    for idx, algo in enumerate(algos):
        # init action values at zero;
        now = time.time()
        run_algo(idx, bandits, algo, NUM_STEPS, metric_tracker)
        print(f"run {idx} took: {time.time() - now:.2f} secs")

    # save plot;
    plot_testbed(
        file_name,
        (algo.get_label() for algo in algos),
        metric_tracker.avg_rewards,
        metric_tracker.prop_true_optimal,
        title="True action values from $\mathcal{N}(0, 1)$",
    )
