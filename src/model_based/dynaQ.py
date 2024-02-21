from pathlib import Path

p = Path(__file__).absolute().parent.parent
import sys

sys.path.append(str(p))
from n_step.policies import TabularGreedy


def dynaQ(n_planning, num_episodes, env, Q, behaviour, model, discount, lr):
    pi = TabularGreedy(Q)
    K = 0
    ep_lens = []
    shortest_ep = 1e8
    shortest_ep_idx = None
    tenth = num_episodes // 10

    while K < num_episodes:
        obs, _ = env.reset(start=(2, 0))
        action = behaviour.sample(obs)
        t = 1
        finished = False
        while not finished:
            # env step;
            obstp1, R, terminated, truncated, _ = env.step(action)
            # direct RL step;
            action_greedy = pi.sample(obstp1)
            Q[obs][action] = Q[obs][action] + lr * (
                R
                + discount * Q[obstp1][action_greedy] * (1 - terminated)
                - Q[obs][action]
            )
            # model learning step;
            model.update(obs, action, R, obstp1)
            # planning - one step Q planning;
            for _ in range(n_planning):
                obs_sim, action_sim, R_sim, obstp1_sim = model.sample()
                action_greedy = pi.sample(obstp1_sim)
                Q[obs_sim][action_sim] = Q[obs_sim][action_sim] + lr * (
                    R_sim
                    + discount * Q[obstp1_sim][action_greedy]
                    - Q[obs_sim][action_sim]
                )
            # update next state and action;
            obs, action = obstp1, behaviour.sample(obstp1)
            # stopping criterion;
            finished = terminated or truncated
            if finished:
                ep_lens.append(t)
                if shortest_ep > t:
                    shortest_ep = t
                    shortest_ep_idx = K + 1
            t += 1
        if (K + 1) % tenth == 0:
            print(f"{(K+1) // tenth}0% episodes done!")
        K += 1
    print(f"shortest episode: {shortest_ep}, episode idx: {shortest_ep_idx}")
    return pi, ep_lens


experiment_config = dict(
    n_planning=5,
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

    import matplotlib.pyplot as plt

    logs_path = p.parent / "logs"
    if not logs_path.exists():
        logs_path.mkdir()

    import random
    import re

    import numpy as np
    import torch
    from models import HashTableModel

    from envs.grid_world import GridWorld2d
    from n_step import utils
    from n_step.policies import TabularEpsGreedy

    float_pattern = re.compile("[0-9]\.[0-9]*")
    int_pattern = re.compile("[0-9]+")

    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            k, v = arg.split("=")
            if k in experiment_config:
                print(k)
                if re.match(float_pattern, v):
                    experiment_config[k] = float(v)
                elif re.match(int_pattern, v):
                    experiment_config[k] = int(v)

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
    model = HashTableModel()

    behaviour = TabularEpsGreedy(
        Q, env.action_space, eps=experiment_config["eps"]
    )
    n_planning = experiment_config["n_planning"]
    greedy_policy, ep_lens = dynaQ(
        n_planning=n_planning,
        num_episodes=experiment_config["num_episodes"],
        env=env,
        Q=Q,
        behaviour=behaviour,
        model=model,
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

    this_logs = (
        logs_path / f"{n_planning}_planning_step_dynaQ_logs_{time.time()}"
    )
    if not this_logs.exists():
        this_logs.mkdir()

    with open(
        this_logs / f"{n_planning}_planning_step_dynaQ_greedy_policy.pkl", "wb"
    ) as f:
        pickle.dump(greedy_policy.Q, f)

    with open(
        this_logs / f"{n_planning}_planning_step_dynaQ_policy_vis.pkl", "wb"
    ) as f:
        pickle.dump(greedy_policy_vis, f)

    with open(
        this_logs / f"{n_planning}_planning_step_dynaQ_greedy_policy_vis.pkl",
        "wb",
    ) as f:
        pickle.dump(greedy_policy_vis, f)

    with open(
        this_logs / f"{n_planning}_planning_step_dynaQ_ep_lens.pkl", "wb"
    ) as f:
        pickle.dump(ep_lens, f)

    plt.plot(range(len(ep_lens)), ep_lens)
    plt.yscale("log")
    plt.xlabel("num episodes")
    plt.ylabel("steps per episode")
    plt.tight_layout()
    plt.savefig(this_logs / f"{n_planning}_planning_step_dynaQ_ep_lens.png")
