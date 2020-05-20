import numpy as np
import torch
import scipy.signal

rewards = np.array([ 0.72270377,  1.33093549,  0.36678044,  1.49516228,  2.73720541,
        2.86800807, -0.09644525,  1.25642886,  2.36643214, -0.07095623,
        1.148693  ,  1.61670289,  3.1770148 ,  1.38303684,  1.84109085,
        1.72989288,  1.43273288,  2.43602551,  2.62099409,  2.05260896, -0.45219362])
values = np.array([-0.06875905, -0.07922465, -0.04931327, -0.07570779, -0.05026164,
        0.00947789,  0.28454363,  0.2837139 ,  0.32003614,  0.25362056,
       -0.06848691, -0.04192663, -0.08895249, -0.06932063, -0.07609748,
       -0.02815246,  0.04277497,  0.1234284 ,  0.21208242,  0.17454746,
        0.0933223 ])
dones = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 1.])
last_val = 0
gamma = 0.99
lam = 0.97

def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input: 
        vector x, 
        [x0, 
         x1, 
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

def spinningup_gae(rewards, values):
    rews = np.append(rewards, last_val)
    vals = np.append(values, last_val)
    # the next two lines implement GAE-Lambda advantage calculation
    deltas = rews[:-1] + gamma * vals[1:] - vals[:-1]
    print(deltas)
    adv_buf = discount_cumsum(deltas, gamma * lam)

    # the next line computes rews-to-go, to be targets for the value function
    ret_buf = discount_cumsum(rews, gamma)[:-1]

    return adv_buf, ret_buf


# returns = np.zeros((20,))
# advantages = np.zeros((20,))
# deltas = np.zeros((20,))
# prev_return = 0
# prev_value = 0
# prev_advantage = 0
# returns[-1] = rewards[-1]
# for i in reversed(range(0, len(returns)-1)):
#     returns[i] = rewards[i] + gamma * prev_return * (1 - dones[i])
#     deltas[i] = rewards[i] + gamma * prev_value * (1 - dones[i]) - values[i]
#     # ref: https://arxiv.org/pdf/1506.02438.pdf (generalization advantage estimate)
#     advantages[i] = deltas[i] + gamma * lam * prev_advantage * (1 - dones[i])
#     prev_return = returns[i]
#     prev_value = values[i]
#     prev_advantage = advantages[i]
    
    
# returns = np.zeros((20,))
# returns[-1] = rewards[-1]
# for t in reversed(range(0, len(returns)-1)):
#     returns[t] = rewards[t] + gamma * returns[t+1] * (1-dones[t])
# returns = returns[:-1]