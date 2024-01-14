def get_lambda_return(lam, discount, rewards, values):
    """
    Returns lambda return.

    Assumes len(rewards) == len(values).
    So values[-1] = 0 since value at terminal
    state is 0.
    """
    assert len(rewards) == len(values)
    G = 0
    for i in range(len(rewards) - 1, -1, -1):
        G = rewards[i] + discount * ((1 - lam) * values[i] + lam * G)
    return G


def get_lambda_return_from_td(lam, discount, rewards, values):
    """
    Returns lambda-return.

    The calculation is based on the weighted sum of
    TD(0) errors. The weights are according to
    powers of lambda * discount. All TD(0) errors
    are from now until the end of the episode.

    Assumes len(rewards) = len(values) - 1.
    values[0] = v(s_t)
    values[-1] = 0 -> value at terminal state;
    rewards[0] = R_{t+1};
    """
    assert len(rewards) == len(values) - 1
    G = 0
    for i in range(len(rewards)):
        G += (lam * discount) ** i * (
            rewards[i] + discount * values[i + 1] - values[i]
        )
    return G + values[0]


def value_func_update_elig_trace(
    step_size, lam, discount, rewards, values, grad_vals
):
    """
    Computes parameter update for state-value func
    using eligibility traces.

    Note:
        Updates for the entire episode are accumulated in ep_update
            and can then be applied to update the parameters.
        This style of computation mimics online learning and
            only requires maintaining a single eligibility trace
            vector, e.
        Here the rewards and values and grad_vals are given for
            testing purposes. In practice, we would prefer to
            process rewards and states as we sample the episode.
        The key difference with this and the brute force approach
            are the memory and compute savings - don't need to
            save vector of rewards and/or states during episode.
            Instead, each reward and state contribute with the
            eligibility vector and TD(0) error.
        Applying these updates online, however, will not be correct
            and will not correspond to lambda-return episodic learning.

    """
    assert len(rewards) == len(values) - 1 == len(grad_vals)
    ep_update = 0
    e = 0
    for i in range(len(rewards)):
        e = lam * discount * e + grad_vals[i]
        td0 = rewards[i] + discount * values[i + 1] - values[i]
        ep_update += step_size * td0 * e
    return ep_update


def brute_force_value_func_update(
    step_size, lam, discount, rewards, values, grad_vals
):
    """
    Returns the episodic update to the params of the val func
    based on step_size * sum((G_t - v(s_t)) * grad_vals(s_t).
    """
    assert len(rewards) == len(values) - 1 == len(grad_vals)
    ep_update = 0
    G = 0
    for i in range(len(rewards) - 1, -1, -1):
        G = rewards[i] + discount * ((1 - lam) * values[i + 1] + lam * G)
        ep_update += step_size * (G - values[i]) * grad_vals[i]
    return ep_update
