import math
import time
from copy import deepcopy

import gymnasium as gym
import hydra
import numpy as np
import torch
from gymnasium.vector import VectorEnv
from omegaconf import DictConfig, OmegaConf
from torch import nn
from tqdm import tqdm

import wandb
from src.actor_critic.utils import eval_loop
from src.dqn import qfunc, utils
from src.metadata import metadata


def train_loop(
    num_iters,
    env: VectorEnv,
    Qfunc,
    Qfunc_optim,
    Qtarget,
    update_target_every_n_grad_steps,
    replay_buffer,
    discount=0.99,
    seed=None,
    max_grad_norm=None,
    do_abs_loss=False,
):
    assert isinstance(env.unwrapped, VectorEnv)
    num_envs = env.num_envs
    batch_idx = torch.arange(replay_buffer.batch_size)
    tot_steps = num_envs * num_iters
    print(f"WILL TRAIN FOR {tot_steps} STEPS...")

    obs_t, info = env.reset(seed=seed)
    for _ in tqdm(range(num_iters)):
        # get action;
        action = Qfunc.sample(torch.tensor(obs_t, dtype=torch.float32)).numpy()
        # do env step;
        obs_tp1, reward, truncated, terminated, info = env.step(action)
        done = truncated + terminated
        # add experience to replay buffer;
        replay_buffer.add_experience(obs_t, action, reward, obs_tp1, done)
        # sample data from buffer;
        batch = replay_buffer.sample()
        # zero grad;
        Qfunc_optim.zero_grad()
        # get loss;
        loss = utils.get_loss(
            Qfunc=Qfunc,
            Qtarget=Qtarget,
            discount=discount,
            batch=batch,
            batch_idx=batch_idx,
            do_abs_loss=do_abs_loss,
        )
        # get grads;
        loss.backward()

        # see if should clip grads;
        if max_grad_norm is None:
            norm = utils.get_grad_norm(Qfunc)
        else:
            norm = nn.utils.clip_grad_norm_(
                Qfunc.parameters(),
                max_norm=max_grad_norm,
                norm_type=2,
                error_if_nonfinite=True,
            )
        # grad step;
        Qfunc_optim.step()

        # do some reward logging;
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
                        f"rollout/reward_{maxlen}_ma": avg_reward,
                    }
                )
        # log training metrics;
        wandb.log(
            {
                "training/qfunc_loss": loss.item(),
                "training/grad_norm": norm.item(),
            }
        )
        # Anneal epsilon each grad step;
        Qfunc.anneal_epsilon()
        # check if Qtarget needs to get updated;
        if (_ + 1) % update_target_every_n_grad_steps == 0:
            utils.copy_first_to_second(Qfunc, Qtarget)

        # update state;
        obs_t, obs_tp1


@hydra.main(
    config_path=str(metadata.CONFIG_PATH),
    config_name="dqn",
    version_base="1.3",
)
def main(config: DictConfig):
    # make the DictConfig object a mutable dict;
    config = OmegaConf.to_container(config, resolve=True)
    tot_steps = (
        config["dqn_config"]["num_iters"] * config["dqn_config"]["num_envs"]
    )
    assert tot_steps <= config["dqn_config"]["max_time_steps"]

    # set seed;
    torch.manual_seed(config["dqn_config"]["seed"])
    p = metadata.SAVED_MODELS_PATH / f"my_dqn_{time.time()}"
    p.mkdir(parents=True, exist_ok=True)

    # instantiate necessary objects:
    # get replay buffer;
    num_envs = config["dqn_config"]["num_envs"]
    env = gym.vector.make(config["dqn_config"]["env_name"], num_envs=num_envs)
    obs_dim = math.prod(env.unwrapped.single_observation_space.shape)
    replay_buffer = utils.ReplayBuffer(
        config["dqn_config"]["buffer_capacity"],
        obs_dim=obs_dim,
        action_dim=None,
        batch_size=config["dqn_config"]["batch_size"],
        seed=0,
    )
    replay_buffer.collect_uniform_experience(
        num_steps=config["dqn_config"]["uniform_experience"], env=env
    )
    # close dummy env;
    env.close()
    del env

    # get env for training;
    env = gym.vector.make(config["dqn_config"]["env_name"], num_envs=num_envs)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=100)
    config["dqn_config"]["mlp_config"][
        "in_dim"
    ] = env.unwrapped.single_observation_space.shape[0]

    # init qfunc;
    annealer_type = config["dqn_config"]["annealer"]
    if annealer_type == "cos":
        annealer = qfunc.CosineAnnealer(
            max_val=config["dqn_config"]["max_eps"],
            min_val=config["dqn_config"]["min_eps"],
            anneal_steps=config["dqn_config"]["anneal_steps"],
        )
    elif annealer_type == "linear":
        annealer = qfunc.LinearAnnealer(
            max_val=config["dqn_config"]["max_eps"],
            min_val=config["dqn_config"]["min_eps"],
            anneal_steps=config["dqn_config"]["anneal_steps"],
        )
    else:
        annealer = None
    # if annealer is not None eps will be set
    # accordingly in the constructor of Qfunc;
    eps = config["dqn_config"]["min_eps"]
    Qfunc = qfunc.Qfunc(
        annealer=annealer,
        eps=eps,
        **config["dqn_config"]["mlp_config"],
        out_dim=env.unwrapped.single_action_space.n,
        hidden_gain=math.sqrt(2),
        out_layer_gain=0.01,
        ortho_init=config["dqn_config"]["ortho_init"],
    )
    Qtarget = deepcopy(Qfunc)
    # don't track grads for Qtarget;
    Qtarget.requires_grad_(False)
    Qfunc_optim = torch.optim.Adam(
        Qfunc.parameters(), lr=config["dqn_config"]["lr"]
    )

    # init wandb;
    run = wandb.init(
        project="rl-algos", name="dqn-local-run", config=config["dqn_config"]
    )

    print(f"TRAINING FOR {tot_steps} STEPS...\n\n")
    train_loop(
        num_iters=config["dqn_config"]["num_iters"],
        env=env,
        Qfunc=Qfunc,
        Qfunc_optim=Qfunc_optim,
        Qtarget=Qtarget,
        update_target_every_n_grad_steps=config["dqn_config"][
            "update_target_every_n_grad_steps"
        ],
        replay_buffer=replay_buffer,
        discount=config["dqn_config"]["discount"],
        seed=config["dqn_config"]["seed"],
        max_grad_norm=config["dqn_config"]["max_grad_norm"],
        do_abs_loss=config["dqn_config"]["do_abs_loss"],
    )
    env.close()
    wandb.finish()

    # save checkpoint assuming other config is saved in wandb;
    torch.save(
        {
            "iters_done": config["dqn_config"]["num_iters"],
            "batch_size": config["dqn_config"]["batch_size"],
            "Qfunc_state_dict": Qfunc.state_dict(),
            "Qfunc_optim_state_dict": Qfunc_optim.state_dict(),
            "Qtarget_state_dict": Qtarget.state_dict(),
        },
        p / f"model_checkpoint",
    )

    # eval the model;
    env = gym.make(config["dqn_config"]["env_name"], render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env,
        p / "videos",
        episode_trigger=lambda x: True,
        name_prefix=config["dqn_config"]["env_name"],
    )

    ep_len, ep_return, avg_reward = eval_loop(
        policy=Qfunc, env=env, greedy=False, seed=0
    )
    env.close()
    print(
        f"ep_len: {ep_len}\nep_return: {ep_return}\navg_reward: {avg_reward}"
    )


if __name__ == "__main__":
    main()
