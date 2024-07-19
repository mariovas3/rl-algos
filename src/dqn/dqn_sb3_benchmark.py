if __name__ == "__main__":
    import gymnasium as gym
    import torch
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.dqn import DQN

    import src.general_utils.data as dutils
    import wandb
    from src.actor_critic.utils import eval_loop
    from src.metadata import metadata

    torch.manual_seed(0)

    p = metadata.SAVED_MODELS_PATH
    p.mkdir(parents=True, exist_ok=True)
    NUM_ENVS = 8
    # the sb3 makevec env is some collection of individual
    # envs and accesses them with random access, so api
    # is slightly different from gym;
    # the seed should be the same though;
    env = make_vec_env("LunarLander-v2", n_envs=NUM_ENVS)
    config = dict(
        learning_rate=3e-4,
        buffer_size=10000,
        learning_starts=10000,
        batch_size=128,
        tau=1,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        target_update_interval=500,
        num_iters=62500,
        num_envs=NUM_ENVS,
        exploration_fraction=30000 / 62500,
        exploration_initial_eps=1,
        exploration_final_eps=0.1,
        max_grad_norm=1,
        stats_window_size=100,
        seed=0,
    )
    run = wandb.init(project="rl-algos", name="dqn-sb3-run", config=config)

    class Logger:
        def record(self, name, val, **kwargs):
            wandb.log({name: val})

        def dump(self, *args, **kwargs):
            pass

    logger = Logger()
    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        buffer_size=10000,
        learning_starts=10000,
        batch_size=128,
        tau=1,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        target_update_interval=500,
        exploration_fraction=30000 / 62500,
        exploration_initial_eps=1,
        exploration_final_eps=0.1,
        max_grad_norm=1,
        stats_window_size=100,
        seed=0,
    )
    model.set_logger(logger=logger)
    print(
        f"Training for {config['num_iters'] * config['num_envs']} steps...\n\n"
    )
    model.learn(total_timesteps=config["num_iters"] * config["num_envs"])
    model.save(p / "sb3_dqn_lunar_lander_v2")

    env.close()
    wandb.finish()
    # eval env;
    env = gym.make("LunarLander-v2", render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env,
        p / "videos",
        episode_trigger=lambda x: True,
        name_prefix="sb3_lunarlander_v2",
    )
    model.sample = lambda obs: torch.from_numpy(model.predict(obs)[0])
    model.greedify = lambda: 0
    model.eval = lambda: 0
    ep_len, ep_return, avg_reward = eval_loop(model, env)
    env.close()
    print(
        f"ep_len: {ep_len}\nep_return: {ep_return}\navg_reward: {avg_reward}"
    )

    dutils.save_to_json(
        {
            "episode_len": ep_len,
            "episode_return": ep_return,
            "average_reward": avg_reward,
        },
        p / "eval_metrics.json",
        indent=2,
    )
