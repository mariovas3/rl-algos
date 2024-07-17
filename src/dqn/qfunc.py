import math

from src.actor_critic.policy import EpsGreedyDiscrete


class Qfunc(EpsGreedyDiscrete):
    def __init__(self, annealer=None, **kwargs):
        super().__init__(**kwargs)
        self.annealer = annealer
        if annealer is not None:
            self.eps = annealer.max_val

    def anneal_epsilon(self):
        if self.annealer is not None:
            self.eps = self.annealer.step()


class CosineAnnealer:
    def __init__(self, max_val, min_val, anneal_steps):
        self.max_val = max_val
        self.min_val = min_val
        self.anneal_steps = anneal_steps
        self.curr_step = 0

    def step(self):
        if self.curr_step > self.anneal_steps:
            return self.min_val
        # when curr_step == 0, coef = 1, and goes down to 0
        # as curr_step increases.
        coef = 0.5 * (
            1 + math.cos(math.pi * (self.curr_step / self.anneal_steps))
        )
        assert 0 <= coef <= 1
        self.curr_step += 1
        return self.min_val + coef * (self.max_val - self.min_val)


class LinearAnnealer:
    def __init__(self, max_val, min_val, anneal_steps):
        self.max_val = max_val
        self.min_val = min_val
        self.anneal_steps = anneal_steps
        self.curr_step = 0

    def step(self):
        if self.curr_step > self.anneal_steps:
            return self.min_val
        # coef goes from 0 to 1;
        coef = self.curr_step / self.anneal_steps
        assert 0 <= coef <= 1
        self.curr_step += 1
        return self.max_val - coef * (self.max_val - self.min_val)
