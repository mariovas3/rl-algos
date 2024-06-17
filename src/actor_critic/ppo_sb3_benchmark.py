if __name__ == "__main__":
    from pathlib import Path

    import gymnasium as gym
    import torch
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.ppo import PPO

    import wandb
    from src.actor_critic.train_vanilla_ppo import eval_loop

    torch.manual_seed(0)

    p = Path(__file__).absolute().parents[2] / "saved_models"
    p.mkdir(parents=True, exist_ok=True)
    NUM_ENVS = 8
    # the sb3 makevec env is some collection of individual
    # envs and accesses them with random access, so api
    # is slightly different from gym;
    # the seed should be the same though;
    env = make_vec_env("LunarLander-v2", n_envs=NUM_ENVS)
    config = dict(
        num_iters=61,  # 2^7
        epochs_per_iter=10,
        steps_per_iter=2048,
        batch_size=128,
        seed=0,
        discount=0.99,
        lam=0.9,
        eps=0.2,
        lr=3e-4,
        num_envs=NUM_ENVS,
    )
    run = wandb.init(project="rl-algos", name="ppo-sb3-run", config=config)

    class Logger:
        def record(self, name, val, **kwargs):
            wandb.log({name: val})

        def dump(self, *args, **kwargs):
            pass

    logger = Logger()
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=config["lr"],
        n_steps=config["steps_per_iter"],
        batch_size=config["batch_size"],
        n_epochs=config["epochs_per_iter"],
        gamma=config["discount"],
        gae_lambda=config["lam"],
        clip_range=config["eps"],
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0,
        vf_coef=1,
        max_grad_norm=0.5,
        seed=0,
    )
    model.set_logger(logger=logger)
    print(
        f"Training for {config['num_iters'] * config['num_envs'] * config['steps_per_iter']} steps...\n\n"
    )
    model.learn(
        total_timesteps=config["num_iters"]
        * config["steps_per_iter"]
        * config["num_envs"]
    )
    model.save(p / "sb3_ppo_lunar_lander_v2")

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
