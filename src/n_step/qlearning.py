from policies import TabularGreedy


def qlearning(num_episodes, env, Q, behaviour, discount, lr):
    pi = TabularGreedy(Q)
    ep_lens = []

    while num_episodes > 0:
        obs, _ = env.reset(start=(2, 0))
        action = behaviour.sample(obs)
        finished = False
        t = 1  # count num rewards;
        while not finished:
            obstp1, R, terminated, truncated, _ = env.step(action)
            print(f"obs: {obs}, action={action}, R={R}, t={t}")
            if terminated:
                Q[obs][action] = Q[obs][action] + lr * (R - Q[obs][action])
            else:
                action_greedy = pi.sample(obstp1)
                Q[obs][action] = Q[obs][action] + lr * (
                    R + discount * Q[obstp1][action_greedy] - Q[obs][action]
                )
                obs, action = obstp1, behaviour.sample(obstp1)
            finished = terminated or truncated
            if finished:
                ep_lens.append(t)
            t += 1
        num_episodes -= 1
    return pi, ep_lens


experiment_config = dict(
    num_episodes=100,
    discount=0.95,
    lr=0.1,
    eps=0.1,
)


if __name__ == "__main__":
    import pickle
    import sys
    import time
    from itertools import product
    from pathlib import Path

    import matplotlib.pyplot as plt

    p = Path(__file__).absolute().parent.parent
    sys.path.append(str(p))
    logs_path = p.parent / "logs"
    if not logs_path.exists():
        logs_path.mkdir()

    import random
    import re

    import numpy as np
    import torch
    import utils
    from policies import TabularEpsGreedy

    from envs.grid_world import GridWorld2d

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    float_pattern = re.compile("[0-9]\.[0-9]*")
    int_pattern = re.compile("[1-9][0-9]*")

    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            k, v = arg.split("=")
            if k in experiment_config:
                if re.match(float_pattern, v):
                    experiment_config[k] = float(v)
                elif re.match(int_pattern, v):
                    experiment_config[k] = int(v)

    print(experiment_config, end="\n\n")

    env = GridWorld2d()
    Q = {
        (row, col): [0] * 4
        for row, col in product(range(env.HEIGHT), range(env.WIDTH))
        if (row, col) not in env.wall
    }

    behaviour = TabularEpsGreedy(
        Q, env.action_space, eps=experiment_config["eps"]
    )
    greedy_policy, ep_lens = qlearning(
        num_episodes=experiment_config["num_episodes"],
        env=env,
        Q=Q,
        behaviour=behaviour,
        discount=experiment_config["discount"],
        lr=experiment_config["lr"],
    )

    action_decoder = {
        0: "<",
        1: ">",
        2: "v",
        3: "^",
    }

    greedy_policy_vis = utils.get_policy_vis(
        env, greedy_policy, action_decoder
    )

    traj = utils.eval_from_start((2, 0), env, greedy_policy)

    greedy_policy_vis = utils.vis_traj(env, traj, action_decoder)

    this_logs = logs_path / f"1_step_Qlearning_logs_{time.time()}"
    if not this_logs.exists():
        this_logs.mkdir()

    with open(this_logs / f"1_step_Qlearning_greedy_policy.pkl", "wb") as f:
        pickle.dump(greedy_policy.Q, f)

    with open(this_logs / f"1_step_Qlearning_policy_vis.pkl", "wb") as f:
        pickle.dump(greedy_policy_vis, f)

    with open(
        this_logs / f"1_step_Qlearning_greedy_policy_vis.pkl", "wb"
    ) as f:
        pickle.dump(greedy_policy_vis, f)

    with open(this_logs / f"1_step_Qlearning_ep_lens.pkl", "wb") as f:
        pickle.dump(ep_lens, f)

    plt.plot(range(len(ep_lens)), ep_lens)
    plt.yscale("log")
    plt.xlabel("num episodes")
    plt.ylabel("steps per episode")
    plt.tight_layout()
    plt.savefig(this_logs / f"1_step_Qlearning_ep_lens.png")
