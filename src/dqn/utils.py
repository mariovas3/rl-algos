import math
import random
from collections import namedtuple

import numpy as np
import torch
from gymnasium.vector import VectorEnv
from torch import nn
from tqdm import tqdm

import wandb


def copy_first_to_second(first: nn.Module, second: nn.Module):
    p = nn.utils.parameters_to_vector(first.parameters())
    nn.utils.vector_to_parameters(p, second.parameters())


def get_loss(Qfunc, Qtarget, discount, batch, batch_idx, do_abs_loss=False):
    assert batch.actions.ndim == 1
    batch_size = len(batch.rewards)
    # batch_idx is torch.arange(batch_size)
    # I accept it as arg, because don't want the overhead
    # of generating torch.arange each time a loss is computed;
    assert batch_idx[-1] == batch_size - 1
    targets = batch.rewards + discount * Qtarget.get_scores(batch.obs_tp1).max(
        -1
    ).values * (1 - batch.dones)
    preds = Qfunc.get_scores(batch.obs_t)[batch_idx, batch.actions]
    assert preds.shape == targets.shape

    if do_abs_loss:
        # make the mse loss behave like absolute error loss
        # when error is outside the (-1, 1) interval
        # this was used in the paper; the idea is that
        # for abs loss, derivative(|x|) = sign(x) if x != 0
        return nn.functional.smooth_l1_loss(preds, targets.detach(), beta=1.0)
    return nn.functional.mse_loss(preds, targets.detach())


def get_grad_norm(net):
    with torch.no_grad():
        norm = math.sqrt(
            sum((p.grad**2).sum().item() for p in net.parameters())
        )
    return norm


Batch = namedtuple(
    "Batch",
    [
        "obs_t",
        "actions",
        "rewards",
        "obs_tp1",
        "dones",
    ],
)


class ReplayBuffer:
    def __init__(
        self, max_steps, obs_dim, action_dim=None, batch_size=32, seed=0
    ):
        # assert action_dim or discrete_env, 'must specify either action_dim or discrete_env=True'
        self.max_steps = max_steps
        self.curr_capacity = 0
        self.batch_size = batch_size
        self.seed = 0
        self.is_full = False
        self.idxs = list(range(self.max_steps))

        self.obs_t = np.zeros((max_steps, obs_dim), dtype=np.float32)
        self.obs_tp1 = np.zeros((max_steps, obs_dim), dtype=np.float32)
        discrete_env = action_dim is None or action_dim == 1
        if discrete_env:
            action_dim = 1
        action_dtype = np.int32 if discrete_env else np.float32
        if action_dim > 1:
            self.actions = np.zeros(
                (max_steps, action_dim), dtype=action_dtype
            )
        else:
            self.actions = np.zeros((max_steps,), dtype=action_dtype)
        self.rewards = np.zeros((max_steps,), dtype=np.float32)
        self.dones = np.zeros((max_steps,), dtype=np.int32)

    def collect_uniform_experience(self, num_steps, env: VectorEnv):
        num_envs = env.unwrapped.num_envs
        num_iters = round(num_steps // num_envs)
        tot_steps = num_envs * num_iters
        print(f"COLLECTING {tot_steps} STEPS FROM UNIFORM POLICY...")
        obs_t, info = env.reset(seed=self.seed)
        for _ in tqdm(range(num_iters)):
            action = env.action_space.sample()
            obs_tp1, reward, truncated, terminated, info = env.step(action)
            dones = truncated + terminated
            self.add_experience(
                obs_t=obs_t,
                actions=action,
                rewards=reward,
                obs_tp1=obs_tp1,
                dones=dones,
            )
            obs_t = obs_tp1
        env.close()

    def sample(self):
        if self.is_full:
            assert self.max_steps >= self.batch_size
        else:
            assert self.curr_capacity >= self.batch_size
        idxs = self.idxs if self.is_full else self.idxs[: self.curr_capacity]
        ids = random.sample(idxs, k=self.batch_size)
        return Batch(
            torch.from_numpy(self.obs_t[ids]),
            torch.from_numpy(self.actions[ids]),
            torch.from_numpy(self.rewards[ids]),
            torch.from_numpy(self.obs_tp1[ids]),
            torch.from_numpy(self.dones[ids]),
        )

    def add_experience(self, obs_t, actions, rewards, obs_tp1, dones):
        num_to_add = len(rewards)
        assert num_to_add <= self.max_steps
        curr = self.curr_capacity
        # see how much space is left until end of buffer;
        forward = self.max_steps - self.curr_capacity
        # add at most num_to_add;
        forward_add = min(num_to_add, forward)
        # forward itself should be at least 1;
        assert forward_add > 0
        self.obs_t[curr : curr + forward_add] = obs_t[:forward_add]
        self.actions[curr : curr + forward_add] = actions[:forward_add]
        self.rewards[curr : curr + forward_add] = rewards[:forward_add]
        self.obs_tp1[curr : curr + forward_add] = obs_tp1[:forward_add]
        self.dones[curr : curr + forward_add] = dones[:forward_add]

        # see if forward was less than num_to_add;
        # if so, add remaining data in beginning of buffer;
        remaining_to_add = num_to_add - forward_add
        if remaining_to_add > 0:
            self.obs_t[:remaining_to_add] = obs_t[forward_add:]
            self.actions[:remaining_to_add] = actions[forward_add:]
            self.rewards[:remaining_to_add] = rewards[forward_add:]
            self.obs_tp1[:remaining_to_add] = obs_tp1[forward_add:]
            self.dones[:remaining_to_add] = dones[forward_add:]
        self.curr_capacity = (self.curr_capacity + num_to_add) % self.max_steps
        if curr > self.curr_capacity:
            self.is_full = True
