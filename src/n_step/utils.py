import numpy as np


def get_policy_vis(env, policy, action_decoder):
    print(f"\n{'-'*20}\nLearned Policy:\n{'-'*20}\n\n")
    policy_vis = []
    for h in range(env.HEIGHT):
        temp = []
        for w in range(env.WIDTH):
            obs = (h, w)
            if obs in env.wall:
                temp.append("|")
                print("|", end="  ")
            elif obs == env.goal:
                print("G", end="  ")
                temp.append("G")
            else:
                best = np.argmax(policy.Q[obs])
                if policy.Q[obs][best]:
                    d = action_decoder[best]
                    print(d, end="  ")
                    temp.append(d)
                else:
                    print(".", end="  ")
                    temp.append(".")
        print("\n")
        policy_vis.append(temp)
    return policy_vis


def eval_from_start(start, env, policy):
    obs_action = {}
    obs, _ = env.reset(start=start)
    while obs != env.goal:
        action = policy.sample(obs)
        obs_action[obs] = action
        obstp1, _, _, _, _ = env.step(action)
        if obstp1 in obs_action:
            raise AssertionError("state-action loop detected")
        obs = obstp1
    return obs_action


def vis_traj(env, traj, action_decoder):
    print(f"\n{'-'*20}\nGreedy on Q:\n{'-'*20}\n\n")
    policy_vis = []
    for h in range(env.HEIGHT):
        temp = []
        for w in range(env.WIDTH):
            obs = (h, w)
            if obs in env.wall:
                temp.append("|")
                print("|", end="  ")
            elif obs == env.goal:
                print("G", end="  ")
                temp.append("G")
            else:
                if obs in traj:
                    d = action_decoder[traj[obs]]
                    print(d, end="  ")
                    temp.append(d)
                else:
                    print(".", end="  ")
                    temp.append(".")
        print("\n")
        policy_vis.append(temp)
    return policy_vis
