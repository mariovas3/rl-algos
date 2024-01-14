import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributions as dists

np.random.seed(0)
torch.manual_seed(0)

SAVE_PATH = Path(__file__).absolute().parent.parent.parent / "assets/imgs"
if not SAVE_PATH.exists():
    SAVE_PATH.mkdir(parents=True)


class MyNormal(dists.Normal):
    def __getitem__(self, idx):
        return dists.Normal(loc=self.mean[idx], scale=self.scale[idx])


# get 2000 bandit problems with 10 arms;
bandits = MyNormal(
    loc=dists.Normal(0.0, 1.0).sample(sample_shape=(2000, 10)), scale=1.0
)

# get idx of true optimal for each bandit;
true_optimal = bandits.mean.argmax(-1).numpy()

# eps = .1, .01, 0;
algos = [dists.Bernoulli(probs=p) for p in (0.9, 0.99, 1)]

# track avg reward and avg time optimal actions picked;
avg_rewards = np.zeros((3, 1000))
prop_true_optimal = np.zeros((3, 1000))
temp = np.arange(2000)

# start experiments;
for idx, algo in enumerate(algos):
    # init value estimates at 0;
    action_values = np.zeros((2000, 10))
    pick_count = np.zeros((2000, 10))
    now = time.time()

    for t in range(1000):
        # see which bandits will be greedy
        is_greedy = algo.sample(sample_shape=(2000,)).int().numpy()

        # presample 2000 random action indexes;
        randoms = np.random.randint(0, 10, size=(2000,))

        # get indexes of greedy with random tie breaking;
        greedies = (
            dists.Categorical(
                probs=torch.from_numpy(
                    action_values == action_values.max(-1, keepdims=True)
                )
            )
            .sample()
            .int()
            .numpy()
        )

        # see which actions to play;
        msk = is_greedy * greedies + (1 - is_greedy) * randoms
        # print(msk[:5])

        # sample rewards
        rewards = bandits[temp, msk].sample().numpy()
        # print(msk.shape, rewards.shape, action_values.shape, pick_count.shape,
        #   action_values[:, msk].shape, pick_count[:, msk].shape)

        # get avg reward for this time step across bandits;
        avg_rewards[idx, t] = rewards.mean()

        # get avg time optimal was selected;
        prop_true_optimal[idx, t] = (msk == true_optimal).mean()

        # update pick count;
        pick_count[temp, msk] += 1

        # update action values;
        action_values[temp, msk] += (
            rewards - action_values[temp, msk]
        ) / pick_count[temp, msk]
    print(f"run {idx} took: {time.time() - now:.2f} secs")


def plot_testbed():
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(7, 10))
    axs[0].set_ylabel("avg reward")
    axs[0].set_xlabel("Steps")
    axs[1].set_ylabel("prop optimal picked")
    axs[1].set_xlabel("Steps")
    for idx, algo in enumerate(algos):
        axs[0].plot(
            range(1000),
            avg_rewards[idx, :],
            label=f"$\epsilon$={algo.probs.item():.2f}",
        )
        axs[1].plot(
            range(1000),
            prop_true_optimal[idx, :],
            label=f"$\epsilon$={algo.probs.item():.2f}",
        )
        axs[0].legend()
        axs[1].legend()
    fig.tight_layout()
    plt.savefig(SAVE_PATH / "ten-arm-testbed.png")


plot_testbed()
