import numpy as np 
from dynamics_rot import *
from plot_misc import *
from scipy.integrate import odeint

# set initial conditions 
q0 = np.array([0.3, 0.2, 0.2, 0])
q0[-1] = np.sqrt(1 - np.sum(q0[0:3]**2))
w0 = np.array([0.5, 1, 0.2])   # rad/s
qw0 = np.concatenate((q0, w0))

J = np.diag([1, 2, 3])  # inertia tensor

# set force (now toque-free)
n_time = 1000
t = np.linspace(0, 100, n_time)
action = np.zeros((3, n_time)) 

# propagate dynaimcs 
qw = get_rot_discrete_control(qw0, action, t, J)
rtn = np.zeros((3, qw.shape[1]))

# plotting (animation)
animate_attitude(rtn, qw)

