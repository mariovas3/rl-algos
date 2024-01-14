import numpy as np

from src.eligibility_traces.eligibility_traces import *

np.random.seed(0)

rewards = np.random.normal(size=(5,))
values = list(np.random.normal(size=(4,))) + [0]
lam, discount = 0.5, 0.9
step_size = 0.1
grad_vals = np.random.normal(size=rewards.shape)


def test_get_lambda_return():
    G = get_lambda_return(lam, discount, rewards, values)
    print(G)
    G2 = get_lambda_return_from_td(lam, discount, rewards, [3.14] + values)
    print(G2)
    assert np.allclose(G, G2)


def test_value_func_update_elig_trace():
    ep_update1 = value_func_update_elig_trace(
        step_size, lam, discount, rewards, [3.14] + values, grad_vals
    )
    ep_update2 = brute_force_value_func_update(
        step_size, lam, discount, rewards, [3.14] + values, grad_vals
    )
    assert np.allclose(ep_update1, ep_update2)
    print(ep_update1, ep_update2)
