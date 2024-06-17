## PPO Tips and Tricks:
### Reference papers and blog posts:
* <a href="https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/">37 tricks from CleanRL</a>
* Engstrom, Ilyas, et. al., (2020).
* Andrychowicz, et al., (2021).


I implemented my own version of PPO, without some excessive tricks such as value func clipping. I use standard mse loss on the approximated returns computed by adding the value func estimate to the advantage estimates. These returns should correspond to $TD(\lambda)$ returns under the old policy. So we are essentially doing policy eval for the old policy when training the value func. I suppose this should work well if the new policy after the grad steps is not that different from the policy setting before the grad optimisation.

Since in the lunar lander env, the truncation is at step 1000; I noticed that in some failed experiments the policy learns to make long bad episodes, so I set the number of sampling steps to 2048 and this had a great improvement in performance compared to 1024 sampling steps - all else equal. The reason is probably because I handle truncation and not termination by adding `discount * v_tp1` to the reward on truncated examples. This is because when calculating the $TD(0)$ errors for the advantage computation, no bootstrap will be added in the above value loss itself since `done` is true. In the beginning, when the value func is not trained enough, this may be an optimistic estimate so might trick the policy to prefer long useless episodes to shorter ones.

Another interesting thing was that I increased performance greatly by changing the lambda parameter from 0.9 to 0.99, which makes the advantages more far-sighted. In particular, I set $\lambda = 0.99$ and $\gamma = 0.99$ giving a horizon of $1 / (1 + \lambda \gamma)\approx 50$ i.e., roughly take into consideration the first 50 `TD(0)` errors when computing the advantages.

I tested my version of vanilla PPO against PPO from stable baselines3. After training for `61 * 2048 * 8 = 999424` env steps I got **301** episodic return for my version of vanilla PPO and 286 for sb3 PPO (with stochastic policy on lunar lander - single seed single run). The 61 is the number of iterations, the 2048 are the number of steps sampled in each iter and 8 is the number of parallel envs (implemented as `gymnasium.vector.VectorEnv`).

Finally, I also computed advantages in batches after the experience is collected. This resulted in a speedup since there's less overhead compared to computing values one at a time in an online fashion while sampling experience. This is something that is not done in the cleanRL implementation. 

It looks like sb3 also don't do this in batches and as a result the above experiments with sb3 took 18m and 30s on an intel i5 machine while my implementation was done in **15m and 25s**. Both experiments had the exact same configs.

Below is a video of my trained model:

<video width="600" controls>
  <source src="https://drive.google.com/file/d/1EQ6Rr8PS7zvMb3A0PAk6drRjR2htCoPH/view?usp=sharing" type="video/mp4">
  Your browser does not support the video tag.
</video>

### My suggestions:
* There could be better hyper-param settings. I tuned my hyperparams based on gut feeling. In summary, it seems the model is quite sensitive to the number of env steps made in each iteration as well as the horizon of the advantages. Which makes sense since we want to trade off greedyness for long-term "planning" behaviour. Longer horizons (around 50) and more timesteps (2048 about double the default 1000 step truncation length) seemed to work well for `LunarLander-v2`. 

Below I also summarise some of the extra implementation tricks I used which look to be common based on the 37 tricks blog post.

### (subset of) 13 base ppo tricks from the 37 tricks blog post:

> I followed some of the 13 base ppo tricks from the 37 tricks blog post. Some points are skipped since they are mentioned in the paper and not ad-hoc, therefore.

* Orthogonal init of weights `torch.nn.init.orthogonal_` and const init of biases to `0`. Seems to perform better than xavier. Scaling for hidden layer weights is `sqrt(2)`. Policy output layer weights have scale `0.01`; value func output layer weights have scale `1`.
* Adam's eps param set to `1e-5` instead of `1e-8`.
* I used a const lr of `3e-4` for both value func and policy. The blog post, however, suggests Adam's lr set to decay from `3e-4` to `0` in mujoco (relatively small gains).
* Returns are computed as `advantage_t + v_t`, where we aim for TD($\lambda$) estimation.
    * **Note**: Using mse loss on that target results in estimating the value func of `pi_old` since that's where we sample our experience from.
* I used shuffling of the `num_envs * num_time_steps` sampled steps when doing mini-batch learning for the policy. Initially I had shuffling only across the time steps dimension, but I saw a small gain from doing their method.
* Standardisation of advantages inside minibatches: `adv = (adv - adv.mean()) / (adv.std() + 1e-8)`.
* I didn't use value func objective clipping. Based on the two papers given above, there shouldn't be great impact anyways.

* I maintained separate policy mlp and vfunc mlp so I could optimise these independently. One could also use param tying and only separate output layers for the vfunc and policy, however, based on the experiments in the 37 tricks blog post this is worse.

* I follow the advice to do grad clipping to `0.5` over the params of policy concatted with params of vfunc. Didn't seem too important although there was some slight gain.

* Debug metrics:
    * I tracked policy loss, value loss, grad norm, episodic return, episode lengths, avg reward in episode, mean of policy ratio as well as its std, and that's probably it.
    * The policy loss almost always went close to zero very quickly, so that was not very informative for debugging.
    * The value loss and the episodic returns, however, fluctuated quite a bit and gave me ideas for what needs to be changed.