import time
from pathlib import Path

import numpy as np
import torch
import torch.distributions as dists
from utils import (
    AvgRewardAndTrueOptTracker,
    EpsGreedyEnv,
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

    file_name = SAVE_PATH / "ten-arm-testbed.png"

    # get 2000 bandit problems with 10 arms;
    bandits = get_normal_bandits(NUM_BANDITS, NUM_ARMS)

    # get idx of true optimal for each bandit;
    true_optimal = bandits.mean.argmax(-1).numpy()

    # eps = .1, .01, 0;
    algos = [dists.Bernoulli(probs=p) for p in (0.9, 0.99, 1)]

    # init metric tracker;
    metric_tracker = AvgRewardAndTrueOptTracker(
        len(algos), NUM_STEPS, true_optimal
    )

    # run experiments;
    for idx, algo in enumerate(algos):
        # init action values at zero;
        action_values = np.zeros((NUM_BANDITS, NUM_ARMS))
        bandit_env = EpsGreedyEnv(algo, action_values)
        now = time.time()
        run_algo(idx, bandits, bandit_env, NUM_STEPS, metric_tracker)
        print(f"run {idx} took: {time.time() - now:.2f} secs")

    # save plot;
    plot_testbed(
        file_name,
        (f"$\epsilon$={1-algo.probs.item():.2f}" for algo in algos),
        metric_tracker.avg_rewards,
        metric_tracker.prop_true_optimal,
    )
