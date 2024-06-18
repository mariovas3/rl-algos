import math
import time
from itertools import chain
from pathlib import Path

import gymnasium as gym
import hydra
import numpy as np
import torch
from gymnasium.vector import VectorEnv
from omegaconf import DictConfig, OmegaConf
from torch import nn

import wandb
from src.actor_critic.policy import MLP, DiscretePolicy, ortho_init_
from src.actor_critic.utils import (
    collect_T_steps,
    compute_advantages_returns_and_log_probs,
    get_loader,
    objective_clip,
)


def train_loop(
    vfunc,
    policy,
    env: VectorEnv,
    num_iters,
    epochs_per_iter,
    steps_per_iter,
    policy_optim,
    vfunc_optim,
    seed=0,
    discount=1,
    lam=1,
    batch_size=50,
    eps=0.2,
    max_grad_norm=0.5,
    standardise_advantage=False,
):
    """
    Train ppo components.

    The policy is trained by maximising the ppo clip objective.

    The value func is trained by minimising MSE with sampled
    returns from the old setting of the policy - effectively
    aiming to do policy eval for the old policy.
    """
    assert isinstance(env.unwrapped, VectorEnv)
    obs_dim = math.prod(env.unwrapped.single_observation_space.shape)

    start_obs = None
    for it in range(num_iters):
        obs_t, actions, rewards, obs_tp1, dones = collect_T_steps(
            env=env,
            policy=policy,
            T=steps_per_iter,
            discount=discount,
            vfunc=vfunc,
            start_obs=start_obs,
            seed=seed if it == 0 else None,
        )

        # get the last obs;
        start_obs = obs_tp1[-1]

        # make numpy arrays to torch tensors;
        actions = torch.tensor(np.array(actions))
        assert actions.shape == (steps_per_iter, env.num_envs)
        obs_t = torch.tensor(
            np.array(obs_t), dtype=torch.float32
        )  # (T, num_envs, obs_dim)
        obs_tp1 = torch.tensor(
            np.array(obs_tp1), dtype=torch.float32
        )  # (T, num_envs, obs_dim)
        dones = torch.from_numpy(np.array(dones)).int()  # (T, num_envs)
        rewards = torch.from_numpy(np.array(rewards))  # (T, num_envs)

        (
            advantages,
            returns,
            log_probs,
        ) = compute_advantages_returns_and_log_probs(
            policy=policy,
            actions=actions,
            rewards=rewards,
            obs_t=obs_t,
            obs_tp1=obs_tp1,
            dones=dones,
            vfunc=vfunc,
            discount=discount,
            lam=lam,
            batch_size=batch_size,
        )

        loader = get_loader(
            obs_t=obs_t.view(-1, obs_dim),
            actions=actions.view(-1),
            rewards=rewards.view(-1),
            obs_tp1=obs_tp1.view(-1, obs_dim),
            dones=dones.view(-1),
            advantages=advantages.view(-1),
            returns=returns.view(-1),
            log_probs=log_probs.view(-1),
            batch_size=batch_size,
        )

        for epoch in range(epochs_per_iter):
            for i, batch in enumerate(loader):
                policy_optim.zero_grad()
                # get prob ratios;
                prob_ratios = (
                    policy.log_prob(batch.actions, batch.obs_t)
                    - batch.log_probs
                ).exp()
                policy_loss = -objective_clip(
                    prob_ratios=prob_ratios,
                    advantages=batch.advantages,
                    eps=eps,
                    standardise_advantage=standardise_advantage,
                )

                # train vfunc with mse on sampled returns;
                vfunc_optim.zero_grad()
                v_loss = nn.functional.mse_loss(
                    vfunc(batch.obs_t).squeeze(), batch.returns
                )

                # calculate grads;
                policy_loss.backward()
                v_loss.backward()

                # clip grads;
                norm = nn.utils.clip_grad_norm_(
                    chain(policy.parameters(), vfunc.parameters()),
                    max_norm=max_grad_norm,
                )
                # optimiser step;
                policy_optim.step()
                vfunc_optim.step()
        wandb.log(
            {
                "training/policy_loss": policy_loss.item(),
                "training/vfunc_loss": v_loss.item(),
                "training/tot_grad_norm": norm.item(),
                "training/imp_weight_mean": prob_ratios.mean().item(),
                "training/imp_weight_std": prob_ratios.std().item(),
                "training/approx_kl": (prob_ratios - 1 - prob_ratios.log())
                .mean()
                .item(),
                "training/iter_step": it,
            }
        )
        print(f"\nTraining Iter {it + 1}:\n{'-' * 30}")
        print(
            f"policy_loss: {policy_loss.item()}\n"
            f"vfunc_loss: {v_loss.item()}\n"
            f"grad_norm: {norm.item()}"
        )


def eval_loop(policy, env, greedy=False, seed=0):
    if greedy:
        policy.greedify()
    policy.eval()
    torch.manual_seed(seed=seed)

    obs_t, info = env.reset(seed=seed)
    done = False
    avg_reward = 0.0
    ep_len = 0.0
    ep_return = 0.0
    with torch.no_grad():
        while not done:
            action = policy.sample(
                torch.tensor(obs_t, dtype=torch.float32)
            ).numpy()
            obs_tp1, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            obs_t = obs_tp1
            ep_return += reward
            ep_len += 1
            avg_reward = avg_reward + (reward - avg_reward) / ep_len
    return ep_len, ep_return, avg_reward


config_path = Path(__file__).absolute().parents[2] / "conf"


@hydra.main(
    config_path=str(config_path),
    config_name="actor_critic",
    version_base="1.3",
)
def main(config: DictConfig):
    # make the DictConfig object a mutable dict;
    config = OmegaConf.to_container(config, resolve=True)
    tot_steps = (
        config["ac_agent"]["num_iters"]
        * config["ac_agent"]["num_envs"]
        * config["ac_agent"]["steps_per_iter"]
    )
    assert tot_steps <= config["ac_agent"]["max_time_steps"]

    # set seed;
    torch.manual_seed(config["ac_agent"]["seed"])

    p = (
        Path(__file__).absolute().parents[2]
        / "saved_models"
        / f"my_ppo_{time.time()}"
    )
    p.mkdir(parents=True, exist_ok=True)

    # the whole config thing;
    NUM_ENVS = config["ac_agent"]["num_envs"]
    env = gym.vector.make("LunarLander-v2", num_envs=NUM_ENVS)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=100)
    config["ac_agent"]["mlp_config"][
        "in_dim"
    ] = env.unwrapped.single_observation_space.shape[0]

    # init vfunc;
    vfunc = MLP(**config["ac_agent"]["mlp_config"], out_dim=1)
    if config["ac_agent"]["ortho_init"]:
        ortho_init_(vfunc, hidden_gain=math.sqrt(2), out_layer_gain=1)

    # init policy;
    policy = DiscretePolicy(
        **config["ac_agent"]["mlp_config"],
        out_dim=env.unwrapped.single_action_space.n,
        hidden_gain=math.sqrt(2),
        out_layer_gain=0.01,
        ortho_init=config["ac_agent"]["ortho_init"],
    )

    # init optimisers;
    policy_optim = torch.optim.Adam(
        policy.parameters(),
        lr=config["ac_agent"]["lr"],
        eps=config["ac_agent"]["adam_eps"],
    )
    vfunc_optim = torch.optim.Adam(
        vfunc.parameters(),
        lr=config["ac_agent"]["lr"],
        eps=config["ac_agent"]["adam_eps"],
    )

    # init wandb;
    run = wandb.init(
        project="rl-algos", name="ppo-local-run", config=config["ac_agent"]
    )

    print(f"Training for {tot_steps} steps...\n\n")
    # train the agent;
    train_loop(
        vfunc=vfunc,
        policy=policy,
        env=env,
        num_iters=config["ac_agent"]["num_iters"],
        epochs_per_iter=config["ac_agent"]["epochs_per_iter"],
        steps_per_iter=config["ac_agent"]["steps_per_iter"],
        policy_optim=policy_optim,
        vfunc_optim=vfunc_optim,
        seed=config["ac_agent"]["seed"],
        discount=config["ac_agent"]["discount"],
        lam=config["ac_agent"]["lam"],
        batch_size=config["ac_agent"]["batch_size"],
        eps=config["ac_agent"]["eps"],
        max_grad_norm=config["ac_agent"]["max_grad_norm"],
        standardise_advantage=config["ac_agent"]["standardise_advantage"],
    )

    # release resources;
    env.close()
    wandb.finish()

    # save checkpoint assuming other config is saved in wandb;
    torch.save(
        {
            "iters_done": config["ac_agent"]["num_iters"],
            "steps_per_iter": config["ac_agent"]["steps_per_iter"],
            "epochs_per_iter": config["ac_agent"]["epochs_per_iter"],
            "batch_size": config["ac_agent"]["batch_size"],
            "policy_state_dict": policy.state_dict(),
            "policy_optim_state_dict": policy_optim.state_dict(),
            "vfunc_state_dict": vfunc.state_dict(),
            "vfunc_optim_state_dict": vfunc_optim.state_dict(),
        },
        p / f"model_checkpoint",
    )

    # eval the model;
    env = gym.make("LunarLander-v2", render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env,
        p / "videos",
        episode_trigger=lambda x: True,
        name_prefix="eval_lunarlander_v2",
    )

    ep_len, ep_return, avg_reward = eval_loop(
        policy=policy, env=env, greedy=False, seed=0
    )
    env.close()
    print(
        f"ep_len: {ep_len}\nep_return: {ep_return}\navg_reward: {avg_reward}"
    )


if __name__ == "__main__":
    main()
