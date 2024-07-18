# DQN based on the <a href="https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf">paper</a> by Mnih et. al.

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