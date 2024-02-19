import random

from torch import ones as tones
from torch.distributions import Categorical


class MyCategorical(Categorical):
    @property
    def n(self):
        return super().param_shape[0]

    def sample(self):
        return super().sample().item()


class GridWorld2d:
    def __init__(self):
        self.HEIGHT, self.WIDTH = 10, 10
        self.wall = set((h, 7) for h in range(self.HEIGHT - 3))
        self.goal = (0, 9)  # top right;
        self.action_space = MyCategorical(probs=tones((4,)))

    def reset(self, start=None):
        if start is None:
            # pick start row arbitrarily
            self.current = (random.randint(0, self.HEIGHT - 1), 0)
        else:
            self.current = start
        return self.current, {}

    def step(self, action):
        """
        0 - left
        1 - right
        2 - down
        3 - up
        """
        assert 0 <= action < 4
        if action == 0:
            temp = (self.current[0], max(0, self.current[-1] - 1))
        elif action == 1:
            temp = (self.current[0], min(self.WIDTH - 1, self.current[-1] + 1))
        elif action == 2:
            temp = (
                min(self.HEIGHT - 1, self.current[0] + 1),
                self.current[-1],
            )
        else:
            temp = (max(0, self.current[0] - 1), self.current[-1])
        if not (temp in self.wall):
            self.current = temp
        g = self.current == self.goal
        return self.current, int(g), g, g, {}
