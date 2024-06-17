from collections import namedtuple

import numpy as np
import torch
from gymnasium.vector import VectorEnv
from torch.utils.data import DataLoader, Dataset

import wandb


def collect_T_steps(
    env: VectorEnv,
    policy,
    T,
    discount,
    vfunc,
    start_obs=None,
    seed=None,
):
    """Return lists of obs_t, action, reward, obs_tp1, terminated or truncated."""
    curr_obs, next_obs = [], []
    rewards, actions = [], []
    done = []
    policy.eval()
    obs_t = env.reset(seed=seed)[0] if start_obs is None else start_obs
    for _ in range(T):
        curr_obs.append(obs_t)
        # sample action;
        action = policy.sample(
            torch.tensor(obs_t, dtype=torch.float32)
        ).numpy()
        # env step;
        obs_tp1, reward, terminated, truncated, info = env.step(action)

        # do some episode logging;
        if "episode" in info and hasattr(env, "length_queue"):
            maxlen = env.length_queue.maxlen
            lens, returns = env.length_queue, env.return_queue
            if len(env.length_queue) == maxlen:
                lens, returns = np.array(lens), np.array(returns)
                avg_len = np.mean(lens)
                avg_return = np.mean(returns)
                avg_reward = np.mean(returns / lens)
                wandb.log(
                    {
                        f"rollout/len_{maxlen}_ma": avg_len,
                        f"rollout/returns_{maxlen}_ma": avg_return,
                        f"rollout/avg_reward_{maxlen}_ma": avg_reward,
                    }
                )

        # if truncate and not terminate done is still true;
        # so add discount * vfunc(obs_tp1) for this obs to the reward;
        # since will be ignored later when adding depends on (1-done);
        mask = np.logical_and(truncated, 1 - terminated)
        if any(mask):
            with torch.no_grad():
                reward[mask] = (
                    reward[mask]
                    + discount
                    * vfunc(torch.tensor(obs_tp1[mask], dtype=torch.float32))
                    .squeeze()
                    .numpy()
                )

        # in vec envs obs_tp1 will never be terminal
        # there is autoreset if termination reached;
        # to get the terminal obs_tp1 you should
        # get it from info['final_observation'][info['_final_observation'], :]
        next_obs.append(obs_tp1)
        rewards.append(reward)
        actions.append(action)
        done.append(terminated + truncated)
        # update current obs;
        obs_t = obs_tp1
    policy.train()
    return curr_obs, actions, rewards, next_obs, done


def compute_advantages_returns_and_log_probs(
    policy: torch.nn.Module,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    obs_t: torch.Tensor,
    obs_tp1: torch.Tensor,
    dones: torch.Tensor,
    vfunc: torch.nn.Module,
    discount: float,
    lam: float,
    batch_size: int = 50,
) -> torch.Tensor:
    assert (
        len(actions)
        == len(rewards)
        == len(obs_t)
        == len(obs_tp1)
        == len(dones)
    )
    advantages = torch.zeros(rewards.shape)
    log_probs = torch.zeros(rewards.shape)
    returns = torch.zeros(rewards.shape)

    # get advantages;
    vfunc.eval()
    policy.eval()
    with torch.no_grad():
        adv = torch.zeros(rewards.shape[-1])
        for end in range(len(rewards), 0, -batch_size):
            # get log probs;
            log_probs[end - batch_size : end] = policy.log_prob(
                actions[end - batch_size : end], obs_t[end - batch_size : end]
            )
            # get values;
            v_t = vfunc(obs_t[end - batch_size : end]).squeeze()
            # set v_tp1 to zero if terminated;
            v_tp1 = vfunc(obs_tp1[end - batch_size : end]).squeeze() * (
                1 - dones[end - batch_size : end]
            )
            # TD(0) error;
            deltas = rewards[end - batch_size : end] + discount * v_tp1 - v_t
            j = len(deltas) - 1
            for i in range(end - 1, max(end - batch_size, 0) - 1, -1):
                assert j >= 0
                assert advantages[i, 0] == 0
                # reset advantage to 0 if terminated now;
                mask = dones[i] == 1
                adv[mask] = 0
                adv = deltas[j] + discount * lam * adv
                advantages[i] = adv
                j -= 1
            returns[end - batch_size : end] = (
                advantages[end - batch_size : end] + v_t
            )

    vfunc.train()
    policy.train()
    assert not advantages.requires_grad
    assert not log_probs.requires_grad
    assert not returns.requires_grad
    return advantages, returns, log_probs


def objective_clip(
    prob_ratios: torch.Tensor,
    advantages: torch.Tensor,
    eps: float = 0.2,
    standardise_advantage: bool = False,
):
    """
    Args:
        prob_ratios: T x num_envs, or T * num_envs
        advantages: T x num_envs, or T * num_envs
        eps: the clip parameter from the ppo paper
        standardise_advantage: if True, subtract mean and divided by std
    """
    assert not advantages.requires_grad
    assert prob_ratios.requires_grad
    if standardise_advantage:
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-8
        )
    # instead of avg over num_envs only, also avg over the timesteps
    # this is to have somewhat stable updates regardless of num timesteps.
    obj = torch.minimum(
        prob_ratios * advantages,
        torch.clip(prob_ratios, 1 - eps, 1 + eps) * advantages,
    ).mean()
    return obj


Batch = namedtuple(
    "Batch",
    [
        "obs_t",
        "actions",
        "rewards",
        "obs_tp1",
        "dones",
        "advantages",
        "returns",
        "log_probs",
    ],
)


class MyDataset(Dataset):
    def __init__(
        self,
        obs_t: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        obs_tp1: torch.Tensor,
        dones: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        log_probs: torch.Tensor,
    ):
        super().__init__()
        self.obs_t = obs_t
        self.actions = actions
        self.rewards = rewards
        self.obs_tp1 = obs_tp1
        self.dones = dones
        self.advantages = advantages
        self.returns = returns
        self.log_probs = log_probs

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        return Batch(
            self.obs_t[idx],
            self.actions[idx],
            self.rewards[idx],
            self.obs_tp1[idx],
            self.dones[idx],
            self.advantages[idx],
            self.returns[idx],
            self.log_probs[idx],
        )


def get_loader(
    obs_t: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    obs_tp1: torch.Tensor,
    dones: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    log_probs: torch.Tensor,
    batch_size=50,
):
    assert (
        actions.shape
        == advantages.shape
        == returns.shape
        == dones.shape
        == rewards.shape
        == log_probs.shape
    )
    dataset = MyDataset(
        obs_t=obs_t,
        actions=actions,
        rewards=rewards,
        obs_tp1=obs_tp1,
        dones=dones,
        advantages=advantages,
        returns=returns,
        log_probs=log_probs,
    )
    return DataLoader(
        dataset, shuffle=True, batch_size=batch_size, drop_last=True
    )
