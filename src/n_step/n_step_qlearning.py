from collections import deque

from policies import TabularGreedy


def qlearning(num_steps, num_episodes, env, Q, behaviour, discount, lr):
    policy = TabularGreedy(Q)
    ep_lens = []
    K = 0
    shortest_ep = 1e8
    shortest_ep_idx = None

    while K < num_episodes:
        obs_action = deque()
        rewards = deque()
        imp_weights = deque()
        obs, _ = env.reset(start=(2, 0))
        action = behaviour.sample(obs)
        t = 1  # count num actions;
        G, W = 0.0, 1.0
        finished = False
        while t == 1 or obs_action or (num_steps == 1 and not finished):
            if not finished:
                obstp1, R, terminated, truncated, _ = env.step(action)
                obs_action.append((obs, action))
                rewards.append(R)
                # print(f"obs: {obs}, action={action}, R={R}, t={t}")

                if t < num_steps:
                    # num_steps >= 2;
                    # obs and action are given for first step so
                    # start imp sampling from the second step;
                    if t > 1:
                        rho = policy.prob(obs, action) / behaviour.prob(
                            obs, action
                        )
                        rho = max(rho, 1e-8)
                        imp_weights.append(rho)
                        W *= discount * rho
                    G += W * R
                else:
                    # 1 step Q-learning doesn't imp sample;
                    if num_steps > 1:
                        rho = policy.prob(obs, action) / behaviour.prob(
                            obs, action
                        )
                        rho = max(rho, 1e-8)
                        imp_weights.append(rho)
                        W *= rho
                        G += W * R
                    else:
                        W = discount
                        G = R

                    # do Q update;
                    old_obs, old_a = obs_action[0]
                    if terminated:
                        Q[old_obs][old_a] = Q[old_obs][old_a] + lr * (
                            G - Q[old_obs][old_a]
                        )
                    else:
                        action_greedy = policy.sample(obstp1)
                        Q[old_obs][old_a] = Q[old_obs][old_a] + lr * (
                            G
                            + W * Q[obstp1][action_greedy]
                            - Q[old_obs][old_a]
                        )

                    if num_steps > 1:
                        # get rid of oldest reward
                        G = (G - rewards[0]) / (imp_weights[0] * discount)
                        # get rid of oldest importance sample;
                        W /= imp_weights[0]
                        imp_weights.popleft()
                    rewards.popleft()
                    obs_action.popleft()

                # update finished;
                finished = terminated or truncated

                # update obs and action;
                if not finished:
                    obs, action = obstp1, behaviour.sample(obstp1)
                else:
                    # bookkeeping;
                    if shortest_ep > t:
                        shortest_ep = t
                        shortest_ep_idx = K + 1
                    ep_lens.append(t)
                    if num_steps == 1:
                        break
            else:
                # print(f"finished={finished}, t={t}")
                old_obs, old_a = obs_action[0]
                Q[old_obs][old_a] = Q[old_obs][old_a] + lr * (
                    G - Q[old_obs][old_a]
                )
                if num_steps > 1 and imp_weights:
                    # get rid of oldest reward
                    G = (G - rewards[0]) / (imp_weights[0] * discount)
                    # get rid of oldest importance sample;
                    W /= imp_weights[0]
                    imp_weights.popleft()
                rewards.popleft()
                obs_action.popleft()

            # increment time step;
            t += 1

        # decrement num episodes to do;
        K += 1
        if (K + 1) % (num_episodes // 10) == 0:
            print(f"{(K+1) // (num_episodes // 10)}0% episodes done!")
    print(f"shortest episode: {shortest_ep}, ep_idx: {shortest_ep_idx}")
    return policy, ep_lens


experiment_config = dict(
    num_steps=10,
    num_episodes=100,
    discount=0.95,
    lr=0.1,
    eps=0.1,
    seed=4,
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
            else:
                print(f"{'-'*30}\n{k} no a valid experiment arg\n{'-'}*30")

    print(experiment_config, end="\n\n")

    seed = experiment_config["seed"]

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = GridWorld2d()
    Q = {
        (row, col): [0] * 4
        for row, col in product(range(env.HEIGHT), range(env.WIDTH))
        if (row, col) not in env.wall
    }
    n = experiment_config["num_steps"]

    behaviour = TabularEpsGreedy(
        Q, env.action_space, eps=experiment_config["eps"]
    )
    greedy_policy, ep_lens = qlearning(
        num_steps=n,
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

    this_logs = logs_path / f"{n}_step_Qlearning_logs_{time.time()}"
    if not this_logs.exists():
        this_logs.mkdir()

    with open(this_logs / f"{n}_step_Qlearning_greedy_policy.pkl", "wb") as f:
        pickle.dump(greedy_policy.Q, f)

    with open(this_logs / f"{n}_step_Qlearning_policy_vis.pkl", "wb") as f:
        pickle.dump(greedy_policy_vis, f)

    with open(
        this_logs / f"{n}_step_Qlearning_greedy_policy_vis.pkl", "wb"
    ) as f:
        pickle.dump(greedy_policy_vis, f)

    with open(this_logs / f"{n}_step_Qlearning_ep_lens.pkl", "wb") as f:
        pickle.dump(ep_lens, f)

    plt.plot(range(len(ep_lens)), ep_lens)
    plt.yscale("log")
    plt.xlabel("num episodes")
    plt.ylabel("steps per episode")
    plt.tight_layout()
    plt.savefig(this_logs / f"{n}_step_Qlearning_ep_lens.png")
