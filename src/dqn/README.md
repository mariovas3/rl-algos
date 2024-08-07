# DQN based on the <a href="https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf">paper</a> by Mnih et. al.

## Installation details:
* I used a miniconda Python 3.11.9 environment.
* First install `torch` cpu version:

    ```
    pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cpu
    ```

* Next install `ffmpeg`

    ```
    sudo apt-get install ffmpeg
    ```

* Clean the `requirements.txt` file by removing the `torch` and `box` entries and save in `requirements2.txt`:

    ```
    sed 's/(.*torch.*|.*box.*)//' -E requirements.txt > requirements2.txt
    ```

* Install the requirements:

    ```
    pip install -r requirements2.txt
    ```

* Install the `gymnasium[box2d]` stuff:

    ```
    pip install gymnasium[box2d]
    ```
    
    the above command works for `bash`, `zsh` might need some quotation marks. I use `bash`.

* If you don't execute the above instructions successfully, you might get some weird errors.
    * I used to get some weird errors on lightning studio:

        ```
        ...
        
        File "/home/zeus/miniconda3/envs/cloudspace/lib/python3.11/site-packages/moviepy/video/io/ffmpeg_writer.py", line 213, in ffmpeg_write_video
            with FFMPEG_VideoWriter(filename, clip.size, fps, codec = codec,
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        File "/home/zeus/miniconda3/envs/cloudspace/lib/python3.11/site-packages/moviepy/video/io/ffmpeg_writer.py", line 88, in __init__
            '-r', '%.02f' % fps,
                ~~~~~~~~^~~~~
            TypeError: must be real number, not NoneType
        ```

        It's some bs that has to do with `moviepy` and `ffmpeg`. I fixed it by
        
        ```
        pip uninstall moviepy decorator
        pip install moviepy
        ```

## Benchmarks:

* The original paper tests their code on Atari games. This repo focuses more on the RL algos themselves, so will avoid envs that require a lot of tricks in processing. For more details, read the tricks the original paper has used.

I will benchmark my implementation on the `LunarLander-v2` env because I am familiar with it from previous projects, and `CartPole-v1` because CleanRL also use it.

You can also run the `stablebaselines3` DQN by:

```bash
python src/dqn/dqn_sb3_benchmark.py
```

## My results:
* I tuned hyperparameters based on the `LunarLander-v2` environment, since I am familiar with it from my implementation of PPO.
* `CartPole-v1` greedy policy:
    * Episode length: `500`
    * Episode return: `500`
    * Num env steps of training: `5x10^5`
    * Command:
        ```bash
        python src/dqn/train_dqn.py dqn_config.env_name=CartPole-v1 dqn_config.buffer_capacity=10000 dqn_config.uniform_experience=5000 dqn_config.num_iters=62500
        ```
    * More info: CleanRL also uses `5x10^5` env steps and `10^4` buffer capacity.
    * Wandb results - 100-episode moving average for episode lengths, episode returns and average returns during training:

        <img src="../../assets/imgs/dqn/dqn_rollouts_ma_CartPole-v1.png">

* `LunarLander-v2` greedy policy:
    * Episode length: `312`
    * Episode return: `280.79`
    * Num env steps of training: `10^6`
    * Command:
        ```bash
        python src/dqn/train_dqn.py
        ```
    * Wandb results - 100-episode moving average for episode lengths, episode returns and average returns during training:

        <img src="../../assets/imgs/dqn/dqn_rollouts_ma_LunarLander-v2.png"/>
    * The `sb3` dqn with the same config achieved `254` episodic return and `343` episode length with the same config (although not greedy eval since not sure how to access it in `sb3`).
    * It is worth noting that while my implementation ran in `7m 43s`, the `sb3` implementation ran in `29m 48s` on the same hardware. I also observed `sb3` ran slower in my earlier PPO implementation, but this looks like a much larger difference here.
    * While my model achieved 100-episodic-return-moving-average of `237.539` during training, the `sb3` baseline achieved `175.606`.

## Running the code;
* Install the stuff from `requirements.txt` to a Python 3.11.9 env (e.g., conda).
* Go to the root of the repo and export the pythonpath environment variable:

    ```bash
    export PYTHONPATH=.
    ```
* Export the `wandb` environment variable to avoid strange warnings:

    ```bash
    export WANDB_START_METHOD="thread"
    ```
* Run the DQN training script:

    ```bash
    python src/dqn/train_dqn.py
    ```

    this will pick up the config from `ROOT/conf`. To modify the config, just extend the above command by adding e.g., `dqn_config.buffer_capacity=20000`. Similarly you can modify other config options.

* The `do_abs_loss` will use `SmoothL1Loss` as described <a href="https://pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html#smoothl1loss">here</a>. Otherwise, `MSELoss` is used.

* Some learnings:
    * In my experience, it is somewhat difficult to tune the replay buffer capacity. Too much capacity, and you may be retaining bad experience for too long. Too little capacity, and you may not have diverse enough experience - requiring greater exploration rate of the algorithm. 
    * There is also an interplay between buffer capacity and gradient step frequency. Holding grad step frequency constant, you want to quickly reduce your prob of sampling bad experience from the buffer, so want a small buffer capacity.
    * Compared to PPO where at each iter you sample experience according to the old policy and then train for several epochs on that data (relatively fresh data), I found tuning buffer capacity more difficult since it can possibly include experience from way back in time when the agent was still weak.
    * SAC also uses a replay buffer and is also not easy to tune the buffer capacity. That's why PPO in my experience has proved relatively robust compared to replay-buffer-based techniques. The data in PPO are always sampled anew at each iter, making them relatively fresh.

### Bug in `env.action_space.sample` from `gymnasium==0.29.1` and `box2d-py==2.3.5`:

Seems it is not controlled by the seed of the env or `torch.manual_seed` or `np.random.seed` or `random.seed`. can be checked by the following:

```python
import gymnasium as gym
import torch
import numpy as np
import random


def check_action_space_samples(num_envs):
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    env = gym.vector.make('LunarLander-v2', num_envs=num_envs)
    obs_t, info = env.reset(seed=0)
    action = env.action_space.sample()
    env.close()
    return obs_t, action


if __name__ == "__main__":
    for i in range(100):
        _, a1 = check_action_space_samples(2)
        _, a2 = check_action_space_samples(2)
        assert np.allclose(a1, a2), f"run {i}, a1: {a1}, a2: {a2}"
```

It even messes up differently:

```
Traceback (most recent call last):
  File "/home/focal/coding/rl-algos/gymnasium_action_space_bug.py", line 22, in <module>
    assert np.allclose(a1, a2), f"run {i}, a1: {a1}, a2: {a2}"
           ^^^^^^^^^^^^^^^^^^^
AssertionError: run 0, a1: [2 0], a2: [0 0]
```

```
Traceback (most recent call last):
  File "/home/focal/coding/rl-algos/gymnasium_action_space_bug.py", line 22, in <module>
    assert np.allclose(a1, a2), f"run {i}, a1: {a1}, a2: {a2}"
           ^^^^^^^^^^^^^^^^^^^
AssertionError: run 0, a1: [2 3], a2: [2 0]
```

despite having fixed the seeds for `torch`, `numpy`, `random` and the `env` itself.

* This prevented me from being able to reproduce my algorithm. To get around this bs, I coded my own uniform agent:

```python
class DiscreteUniformAgent:
    def __init__(self, out_shape, low, high):
        self.out_shape = out_shape
        self.low, self.high = low, high
    
    def sample(self):
        return np.random.randint(
            low=self.low, 
            high=self.high, 
            size=self.out_shape
        )
```

and substituted `action = env.action_space.sample()` for `action = uniform_agent.sample()`. That fixed everything.