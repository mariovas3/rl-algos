# Repo for misc RL algorithms:

## Maintainer notes:
* I currently have `gymnasium==0.29.1`. 
    * In the `v1.0` version, `gym.vector.make` will be replaced by `gym.make_vec`. 
    * The current version is still widely supported though, as per gymnasium's docs, so I'll stick to it.

List of algos:
* <a href="./docs/bandits.md">Bandits</a>.
* <a href="./src/n_step/README.md">N-step Q-learning</a>.
* <a href="./src/model_based/README.md">Dyna-Q</a>.
* <a href="./src/actor_critic/">PPO</a>.