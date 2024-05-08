"""
2D version of the detumbling problem. 
Assumption: 
- translational motion of the servicer and target is fixed (ignored)
- rotational motion of the servicer is fixed, reaction of the thrust is ignored 
- the target is a circular object (spherical) in a 2D plane, so attitude does not exist 
- only variable is angular velocity of the target
"""

import numpy as np
from scipy.integrate import odeint
import pygame

import gymnasium as gym
from gymnasium import spaces
from rpo_funcs import *
from dynamics.dynamics_rot import * 


class RPO_ENV(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):
        
        self.t = 0  
        self.dt = 10  # sec
        
        # target info
        self.oe0  = np.array([7000e3, 0.001, 0, 0, 0, 0])  # [m]
        self.oe   = self.oe0.copy()
        self.J_targ = np.diag([100,100,100])
        self.n    = np.sqrt(398600e9/self.oe[0]**3)
        
        self.d = 10 # distance between the target and the satellite (currently constant)
        self.r = 1  # radius of the target 
        
        # state = [q, w, ROE]
        self.state = np.zeros((13))
        
        # action = [u_mag, thetea]
        self.action = np.empty((2))  
        self.action_space = spaces.Box(np.array([0,1]),np.array([-np.pi/4,np.pi/4]),dtype=np.float32)        
        self.umax = 1   # max output of the thrustr [unit?]

        # threshold of detumbling (TBD) rad/s
        self.w_tol = 1e-1
        
    # assuminng perfect observation as of now 
    def _get_obs(self):
        return self.state


    def _get_info(self):
        return {}
        
    def reset(self, seed=None, options=None):
        return 
    
    
    def step(self, action):
        
        qw = self.state[0:7]        
        T = a2t_laser_2d(action, self.d, self.r) 
        qw = odeint(ode_qw, qw, [0, self.dt], args=(T,))[1]
        self.state[3:6] = qw[4:7]   # only update angular velocity now... 
        
        self.t += self.dt 
        self.oe += np.array([0,0,0,0,0,self.n*self.dt])
        
        reward = -action[0]
        
        if (abs(self.state[3:6]) < self.w_tol).all():
            terminated = True
        else:
            terminated = False    
        
        info = {}
        
        return self.state, reward, terminated, False, info