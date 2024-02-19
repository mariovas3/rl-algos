import random
from collections import namedtuple
from numbers import Integral, Number

ActionQval = namedtuple("ActionQval", "a qval")


class TabularGreedy:
    def __init__(self, action_space):
        self.estimates = {}
        self.action_space = action_space

    @classmethod
    def init_from_Q(cls, action_space, Q):
        ob = cls(action_space)
        for obs in Q:
            ob.estimates[obs] = max(Q[obs], key=lambda x: x[-1])
        return ob

    def update(self, obs, a, qval):
        assert isinstance(a, Integral)
        assert isinstance(qval, Number)
        if obs in self.estimates:
            if self.estimates[obs].qval < qval:
                self.estimates[obs] = ActionQval(a, qval)
        else:
            self.estimates[obs] = ActionQval(a, qval)

    def sample(self, obs):
        if obs in self.estimates:
            return self.estimates[obs].a
        return self.action_space.sample()

    def prob(self, obs, action):
        if obs in self.estimates:
            return float(action == self.estimates[obs].a)
        return 1 / self.action_space.n


class TabularEpsGreedy(TabularGreedy):
    def __init__(self, action_space, eps):
        super().__init__(action_space)
        self.eps = eps

    def sample(self, obs):
        u = random.random()
        if u < self.eps:
            return self.action_space.sample()
        return super().sample(obs)

    def prob(self, obs, action):
        if obs in self.estimates:
            temp = self.eps / self.action_space.n
            if action == self.estimates[obs].a:
                return 1 - self.eps + temp
            return temp
        return 1 / self.action_space.n
