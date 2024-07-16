from argparse import ArgumentParser
from math import prod
from pathlib import Path

import gymnasium as gym
import torch

from src.actor_critic.policy import MLP, DiscretePolicy
from src.actor_critic.utils import eval_loop

parser = ArgumentParser()
parser.add_argument("--greedy", action="store_true", help="set greedy eval")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("checkpoint_path", type=str)
args = parser.parse_args()
p = Path(args.checkpoint_path).absolute().parent


env = gym.make("LunarLander-v2", render_mode="rgb_array")
env = gym.wrappers.RecordVideo(
    env,
    p / "eval_script_videos",
    episode_trigger=lambda x: True,
    name_prefix="eval_lunarlander",
)
checkpoint = torch.load(args.checkpoint_path)
policy = DiscretePolicy(
    in_dim=prod(env.observation_space.shape),
    out_dim=env.action_space.n,
    n_hidden=2,
    hidden_dim=64,
    deterministic=False,
    hidden_gain=1,
    out_layer_gain=1,
    ortho_init=False,
)
policy.load_state_dict(checkpoint["policy_state_dict"])

ep_len, ep_return, avg_reward = eval_loop(
    policy, env, greedy=args.greedy, seed=args.seed
)
print(f"ep_len: {ep_len}\nep_return: {ep_return}\navg_reward: {avg_reward}")
env.close()
