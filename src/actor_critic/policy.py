import math

import torch
import torch.distributions as dists
from torch import nn


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, n_hidden, hidden_dim):
        super().__init__()
        # add first hidden layer;
        self.net = nn.Sequential()
        self.net.add_module("hidden_0", nn.Linear(in_dim, hidden_dim))
        self.net.add_module("tanh_0", nn.Tanh())
        # add remaining hidden layers;
        for i in range(n_hidden - 1):
            self.net.add_module(
                f"hidden_{i+1}", nn.Linear(hidden_dim, hidden_dim)
            )
            self.net.add_module(f"tanh_{i+1}", nn.Tanh())
        self.net.add_module(f"out_layer", nn.Linear(hidden_dim, out_dim))

    def forward(self, x):
        return self.net(x)


class GaussPolicy(nn.Module):
    def __init__(self, in_dim, out_dim, n_hidden, hidden_dim, isotropic=None):
        """
        If isotropic is None,
        mlp outputs 2 * out_dim for means and stds of diag covariance
        else just out_dim for the mean."""
        super().__init__()
        self.net = MLP(
            in_dim=in_dim,
            out_dim=out_dim if isotropic else 2 * out_dim,
            n_hidden=n_hidden,
            hidden_dim=hidden_dim,
        )
        self.isotropic = isotropic

    def forward(self, state):
        out = self.net(state)
        if self.isotropic:
            loc = out
            scale = self.isotropic
        else:
            l = out.shape[-1]
            mid = l // 2
            assert mid * 2 == l
            loc = out[..., :-mid]
            scale = nn.functional.softplus(
                out[..., mid:], beta=1.0, threshold=20.0
            )
        return dists.Normal(loc=loc, scale=scale)

    def log_prob(self, action, state):
        d = self(state)
        return d.log_prob(action)

    def sample(self, state):
        d = self(state)
        return d.sample()

    def rsample(self, state):
        d = self(state)
        return d.rsample()


class DiscretePolicy(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        n_hidden,
        hidden_dim,
        out_layer_gain,
        hidden_gain=math.sqrt(2),
        deterministic=False,
        ortho_init=True,
    ):
        """
        Policy for discrete action spaces.
        """
        super().__init__()
        self.net = MLP(
            in_dim=in_dim,
            out_dim=out_dim,
            n_hidden=n_hidden,
            hidden_dim=hidden_dim,
        )
        if ortho_init:
            ortho_init_(
                self.net,
                hidden_gain=hidden_gain,
                out_layer_gain=out_layer_gain,
            )
        self.num_actions = out_dim
        self.deterministic = deterministic

    def greedify(self):
        self.deterministic = True

    def forward(self, state):
        out = self.net(state)
        return dists.Categorical(logits=out)

    def get_scores(self, state):
        return self.net(state)

    def log_prob(self, action, state):
        d = self(state)
        return d.log_prob(action)

    def sample(self, state):
        """Sample action."""
        if self.deterministic:
            return self.net(state).argmax(-1)
        return self(state).sample()


class EpsGreedyDiscrete(DiscretePolicy):
    def __init__(self, eps, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps

    def sample(self, state):
        """Sample action."""
        out = self.net(state)
        if self.deterministic:
            return out.argmax(-1)
        # get the greatest score;
        greedy_val = out.max(-1, keepdims=True).values
        # sample one among the greedy actions;
        greedy_samples = dists.Categorical(probs=out == greedy_val).sample()

        if state.ndim == 1:
            size = 1
        else:
            assert state.ndim > 0
            size = len(state)

        # sample random actions;
        randoms = torch.randint(low=0, high=self.num_actions, size=(size,))
        # explore w.p. eps;
        explore = (torch.rand(size=(size,)) <= self.eps).int()
        action = randoms * explore + greedy_samples * (1 - explore)
        return action.squeeze()

    def log_prob(self):
        # no need for log prob for now;
        raise NotImplementedError


def ortho_init_(net, hidden_gain, out_layer_gain):
    for name, p in net.named_parameters():
        if "hidden" in name:
            if "weight" in name:
                nn.init.orthogonal_(p, gain=hidden_gain)
            elif "bias" in name:
                nn.init.constant_(p, val=0)
        elif "out_layer" in name:
            if "weight" in name:
                nn.init.orthogonal_(p, gain=out_layer_gain)
            elif "bias" in name:
                nn.init.constant_(p, val=0)
