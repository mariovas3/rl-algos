from typing import Tuple

import numpy as np
import torch
import torch.distributions as dists


class AvgRewardAndTrueOptTracker:
    def __init__(self, algos, num_steps, true_optimal):
        self.algos = algos
        self.num_steps = num_steps
        self.avg_rewards = np.zeros((len(algos), num_steps))
        self.prop_true_optimal = np.zeros((len(algos), num_steps))
        self.true_optimal = true_optimal

    def __call__(self, rewards, msk, idx, t):
        # get avg reward for this time step across bandits;
        self.avg_rewards[idx, t] = rewards.mean()

        # get avg time optimal was selected;
        self.prop_true_optimal[idx, t] = (msk == self.true_optimal).mean()


class MyNormal(dists.Normal):
    def __getitem__(self, idx):
        return dists.Normal(loc=self.mean[idx], scale=self.scale[idx])


def get_normal_bandits(num_bandits, num_arms):
    return MyNormal(
        loc=dists.Normal(0.0, 1.0).sample(
            sample_shape=(num_bandits, num_arms)
        ),
        scale=1.0,
    )


class EpsGreedyEnv:
    def __init__(self, algo, action_values):
        self.action_values = action_values
        self.pick_count = np.zeros_like(action_values)
        self.algo = algo
        self.num_bandits, self.num_arms = self.pick_count.shape
        self.temp = np.arange(len(action_values))

    def __call__(self, bandits) -> Tuple[np.ndarray]:
        """Get rewards and indexes of chosen arms."""
        # see which bandits will be greedy
        is_greedy = (
            self.algo.sample(sample_shape=(self.num_bandits,)).int().numpy()
        )

        # presample 2000 random action indexes;
        randoms = np.random.randint(
            0, self.num_arms, size=(self.num_bandits,)
        )

        # get indexes of greedy with random tie breaking;
        greedies = (
            dists.Categorical(
                probs=torch.from_numpy(
                    self.action_values
                    == self.action_values.max(-1, keepdims=True)
                )
            )
            .sample()
            .int()
            .numpy()
        )

        # see which actions to play;
        msk = is_greedy * greedies + (1 - is_greedy) * randoms

        # sample rewards
        rewards = bandits[self.temp, msk].sample().numpy()

        # update pick count;
        self.pick_count[self.temp, msk] += 1

        # update action values;
        self.action_values[self.temp, msk] += (
            rewards - self.action_values[self.temp, msk]
        ) / self.pick_count[self.temp, msk]

        return rewards, msk


def run_algo(idx, bandits, bandit_env, num_steps, metric_tracker):
    for t in range(num_steps):
        # get rewards and which arms got picked
        # also update action value estimates;
        rewards, msk = bandit_env(bandits)

        # track metrics;
        metric_tracker(rewards, msk, idx, t)
