# Notes on bandits:



## Nonassociative case:
> TLDR: 
> * Episodes of length one, always starting in the same state **(only for nonassociative case)**. 
> * Find a behaviour/policy that maximises expected reward over some time horizon, $T$. 
> * The behaviour is usually implicit and based on estimates of usefullness/values of actions. 
> * You have to figure out how to trade of exploitation and exploration so you don't miss out on good actions that have been underexplored. Incorporating uncertainty in your decision-making might prove useful, therefore.
> * Many bandit algorithms resmble value-based RL algorithms.

### Objective:
Find $\pi^*$ such that:

$$
\begin{equation}
    \pi^*=\arg \max_{\pi} \sum_{t=1}^T \mathbb{E}_{A_t\sim \pi}\{\mathbb{E}[R_t \mid A_t=a_t]\}.
\end{equation}
$$

If I knew $\mathbb{E}[R_t|A_t=a_t]$ for all $a_t\in \mathcal{A}$ and for all $t$, I can just pick the policy $\pi^*$ that implements a greedy operation:

$$
\begin{equation}
    a_t^* = \arg \max_a \mathbb{E}[R_t|A_t=a].
\end{equation}
$$

Since the expectations are unknown, we have to estimate them and potentially use extra heuristics to pick an action.

### Action Value methods:
These are methods only based on estimates of $\mathbb{E}[R_t|A_t=a_t]$ for all $a_t\in \mathcal{A}$. 

No uncertainty is taken into account. To balance exploitation and exploration we forcefully make a uniformly random action selection with probability $\epsilon$. This is known as $\epsilon$-greedy strategy.

If we have $|\mathcal{A}|$ actions, we pick greedy action/exploit with probability $1-\epsilon + \epsilon / |\mathcal{A}|$. The probability of non-greedy action is $\epsilon \times (|\mathcal{A}| - 1)/ |\mathcal{A}|$,  assuming there is only one greedy action at a time.

### Ten-arm testbed reproduced:
Here I have reproduced the 10-arm testbed from the Sutton and Barto book. I get the same results, see image below. The code is in <a href="../src/bandits/epsgr_vs_gr.py">here</a>.

<img alt="Reproduced plot from 10 armed testbed from the Sutton and Barto book." src="../assets/imgs/ten-arm-testbed.png"/>
