import random
from pathlib import Path

import torch

# path to rl-algos/src
p = Path(__file__).absolute().parent.parent
import sys

sys.path.append(str(p))
from envs.grid_world import MyCategorical


class TabularGreedy:
    def __init__(self, Q):
        # Q is dict[list[qval]]
        self.Q = Q

    def sample(self, obs):
        maxq = max(self.Q[obs])
        # random tie breaks;
        return MyCategorical(
            probs=torch.tensor([maxq == qval for qval in self.Q[obs]])
        ).sample()

    def prob(self, obs, action):
        maxq = max(self.Q[obs])
        if self.Q[obs][action] != maxq:
            return 0
        return 1 / sum([maxq == qval for qval in self.Q[obs]])


class TabularEpsGreedy(TabularGreedy):
    def __init__(self, Q, action_space, eps):
        super().__init__(Q)
        self.action_space = action_space
        self.eps = eps

    def sample(self, obs):
        u = random.random()
        if u < self.eps:
            return self.action_space.sample()
        return super().sample(obs)

    def prob(self, obs, action):
        maxq = max(self.Q[obs])
        N = self.action_space.n
        if self.Q[obs][action] == maxq:
            return 1 - self.eps + self.eps / N
        return self.eps * (N - 1) / N
