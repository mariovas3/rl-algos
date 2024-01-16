from math import log
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributions as dists


class AvgRewardAndTrueOptTracker:
    def __init__(self, num_algos, num_steps, true_optimal):
        self.num_steps = num_steps
        self.avg_rewards = np.zeros((num_algos, num_steps))
        self.prop_true_optimal = np.zeros((num_algos, num_steps))
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


class AvgOrTrackingValUpdater:
    def __init__(self, action_values, lr=None):
        self.action_values = action_values
        self.lr = lr
        if lr is None:
            self.pick_count = np.zeros_like(action_values)

    def __call__(self, rewards, msk, temp=None):
        """
        Updates the mutable object action_values.
        """
        if temp is None:
            temp = np.arange(len(self.action_values))
        if self.lr is None:
            # update pick count;
            self.pick_count[temp, msk] += 1

            # update action values;
            self.action_values[temp, msk] += (
                rewards - self.action_values[temp, msk]
            ) / self.pick_count[temp, msk]
        else:
            self.action_values[temp, msk] += (
                rewards - self.action_values[temp, msk]
            ) * self.lr


class EpsGreedyEnv:
    def __init__(self, algo, action_values, lr=None):
        self.value_updater = AvgOrTrackingValUpdater(action_values, lr)
        self.algo = algo
        self.num_bandits, self.num_arms = self.action_values.shape
        self.temp = np.arange(len(action_values))

    @property
    def action_values(self):
        return self.value_updater.action_values

    @property
    def pick_count(self):
        if self.lr is None:
            return self.value_updater.pick_count
        return None

    def __call__(self, bandits) -> Tuple[np.ndarray]:
        """Get rewards and indexes of chosen arms."""
        # see which bandits will be greedy
        is_greedy = (
            self.algo.sample(sample_shape=(self.num_bandits,)).int().numpy()
        )

        # presample 2000 random action indexes;
        randoms = np.random.randint(0, self.num_arms, size=(self.num_bandits,))

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

        # update action values;
        self.value_updater(rewards, msk, self.temp)
        return rewards, msk


class UCBEnv:
    def __init__(self, action_values, c):
        self.value_updater = AvgOrTrackingValUpdater(action_values)
        self.t = 0
        self.c = c
        self.num_bandits, self.num_arms = action_values.shape
        self.temp = np.arange(len(action_values))
        self.never_tried = 0

    @property
    def action_values(self):
        return self.value_updater.action_values

    @property
    def pick_count(self):
        return self.value_updater.pick_count

    def __call__(self, bandits) -> Tuple[np.ndarray]:
        if self.never_tried < self.num_arms:
            msk = np.ones(self.num_bandits, dtype=int) * self.never_tried
            self.never_tried += 1
        else:
            msk = (
                self.action_values
                + self.c * np.sqrt(log(self.t) / self.pick_count)
            ).argmax(-1)
        rewards = bandits[self.temp, msk].sample().numpy()
        # print(rewards.shape, msk.shape, self.action_values.shape)
        self.value_updater(rewards, msk, self.temp)
        self.t += 1
        return rewards, msk


def run_algo(idx, bandits, bandit_env, num_steps, metric_tracker):
    for t in range(num_steps):
        # get rewards and which arms got picked
        # also update action value estimates;
        rewards, msk = bandit_env(bandits)

        # track metrics;
        metric_tracker(rewards, msk, idx, t)


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
