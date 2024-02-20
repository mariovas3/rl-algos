# Implementation of N-step methods.

## N-step Q-learning:
It's an off-policy method, since behaves with exploring policy and evaluates the greedy policy. Below I discuss the per-decision importance sampling version of the algorithm since it should be with lesser variance compared to the vanilla off-policy per-episode importance sampling.

The update:

$$
\begin{equation}
    Q(S_t, A_t) = Q(S_t, A_t) + \alpha (\tilde{G}_{t+1:t+n-1} - Q(S_t, A_t))
\end{equation}
$$


<img src="../../assets/imgs/gh_cant_render_latex.png"/>


$$
\begin{equation}
	\rho_{i:j}=\prod_{k=i}^j \frac{\pi(A_k\mid S_k)}{\mu(A_k\mid S_k)}.
\end{equation}
$$


In the above, if $S_{t+n}$ is terminal, then we zero out the last term.

Since $\pi$ is the greedy policy, the importance weights can be $0$ if the action selected by the behaviour, $\mu$, is not the greedy action under $\pi$. This can lead to ignoring the entire episode from some point onwards.


to avoid division by zero errors in the incremental implementation of the algorithm. This also has the interpretation that these trajectories are still pretty unlikely under $\pi$.

Based on the update above, the first $n-1$ steps we don't make updates since we wait to get $n$ rewards. These updates are actually compensated for after the episode terminates, and for each consecutive update we pop the earliest reward based on the previous update in FIFO fashion (queue). So if we want to implement n-step algorithms and assuming the termination of the algo, at time $T$, does not occurr before the $n$ time step, we make a total of $T+n-1$ steps (sampling and/or updating) per episode.

As $n$ grows, in principle we learn faster since the reward signal from the terminal state is used to update $n$ state-action pairs. This is consistent with my experiments.

### Double learning:

Finally, since in Q-learning policy evaluation overlaps with policy improvement, it is possible to get maximisation bias. This can be fixed by doing double learning - using two value functions. Then we alternate updates in which we move the value at a state-action pair of one function towards the value of the other function at that same pair. The action selection is done according to the first function. 

Intuitively, you are updating your beliefs about your favourite action, $A_t$, at $S_t$ to be more similar to what your friend thinks of selecting $A_t$ at $S_t$. Intuitively, this mimics the information transfer when talking to friends - e.g., when debating which sushi place is the best in your area.

Mathematically, given two value functions $Q_1$ and $Q_2$, the $n=1$ update for $Q_1$ is:

$$
\begin{equation}
    Q_1(S_t, A_t) = Q_1(S_t, A_t) + \alpha [R_{t+1} + \gamma Q_2(S_{t+1}, \arg \max_a Q_1(S_{t+1}, a)) - Q_1(S_t, A_t)]
\end{equation}
$$

and then you alternate this and the analogous version for $Q_2$.