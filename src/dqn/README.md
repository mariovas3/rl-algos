# DQN based on the <a href="https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf">paper</a> by Mnih et. al.

## Install details:
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
* I will be benchmarking my implementation vs the results from <a href="https://docs.cleanrl.dev/rl-algorithms/dqn/#experiment-results_1">CleanRL</a>.

```
Avg episodic returns in CleanRL:
CartPole-v1	488.69 ± 16.11
Acrobot-v1	-91.54 ± 7.20
MountainCar-v0	-194.95 ± 8.48
```

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