"""
2D version of the detumbling problem. 
Assumption: 
- translational motion of the servicer and target is fixed (ignored)
- rotational motion of the servicer is fixed, reaction of the thrust is ignored 
- the target is a circular object (spherical) in a 2D plane, so attitude does not exist 
- only variable is angular velocity of the target
"""

import numpy as np
import numpy.linalg as la
from scipy.integrate import odeint

import gymnasium as gym
from gym import spaces


class RPO_Detumble2DEnv(gym.Env):
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
        # self.action = np.empty((2))  
        # self.action_space = spaces.Box(np.array([0,1]),np.array([-np.pi/4,np.pi/4]),dtype=np.float32)        
        high = np.array([1,1,1,1, 1,1,1, 200, 200, 200, 200, 200, 200])
        self.observation_space = spaces.Box(-high, high, dtype=np.float64)
        high = np.array([1,np.pi/2])
        low = np.array([0, -np.pi/2])
        self.action_space = spaces.Box(low, high, dtype = np.float64)
        self.umax = 1   # max output of the thrustr [unit?]

        # threshold of detumbling (TBD) rad/s
        self.w_tol = 1e-1
        
    # assuminng perfect observation as of now 
    def _get_obs(self):
        return self.state


    def _get_info(self):
        return {}
        
    def reset(self, seed=None, options=None):
        # Set the seed if provided
        if seed is not None:
            np.random.seed(seed)

        # Reset environment state
        self.state = np.zeros((13))

        # Return initial observation
        # return self._get_obs
        return self.state, {}
    
    
    def step(self, action):

        qw = self.state[0:7]        
        T = a2t_laser_2d(action, self.d, self.r) 
        J = 1e-3
        # qw = odeint(ode_qw, qw, [0, self.dt], args=(T,))[1]
        qw = odeint(ode_qw, qw, [0, self.dt], args=(J, T))[1]
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
    
def a2t_laser_2d(a, d, r):
    """
    Convert action to torque
    Assuming the target is circular (spherical)
    Args:
        a: action [u_mag, theta]
        d: distance between the target and the satellite
        r: radius of the target 
    Returns: 
        torque: [tau_z]   
    """
    
    u_mag, θ = a 
    fvec = np.array([u_mag*np.cos(θ), u_mag*np.sin(θ)])

    ϕ = np.arcsin(r/d*np.sin(abs(θ)))

    if abs(ϕ) < np.pi/2:
        ϕ = np.pi - ϕ
    
    λ = np.pi - θ - ϕ
    
    if θ > 0: 
        rvec = np.array([r*np.cos(λ), r*np.sin(λ)])
    else: 
        rvec = np.array([r*np.cos(λ), -r*np.sin(λ)])
    
    torque = np.cross(rvec, fvec)
    
    return torque 
    
def ode_qw(qw,t,J,T):
    q = qw[0:4]
    w = qw[4:7]
    return np.concatenate((q_kin(q, w),  euler_dyn(w, J, T)))

def q_kin(q, omega):
    # qdot = 0.5 * q * omega
    return 0.5 * q_mul(q, np.array([0, omega[0], omega[1], omega[2]]))

def euler_dyn(w, I, tau):
    # return la.inv(I).dot(tau - np.cross(w, I.dot(w)))
    return (tau - w * I * w) / I

### Quaternion setup 
# q = [q0, q1, q2, q3] = [scalar, vector]

def q_mul(q0, q1):
    return np.array([
        q0[0]*q1[0] - q0[1]*q1[1] - q0[2]*q1[2] - q0[3]*q1[3],
        q0[0]*q1[1] + q0[1]*q1[0] + q0[2]*q1[3] - q0[3]*q1[2],
        q0[0]*q1[2] - q0[1]*q1[3] + q0[2]*q1[0] + q0[3]*q1[1],
        q0[0]*q1[3] + q0[1]*q1[2] - q0[2]*q1[1] + q0[3]*q1[0]
    ])   