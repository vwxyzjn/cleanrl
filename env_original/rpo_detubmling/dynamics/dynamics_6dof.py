# Orbit dynamics
import numpy as np
import numpy.linalg as la
from scipy.integrate import odeint

from .dyn_misc import * 
from .dynamics_trans import *
from .dynamics_rot import *  
from .oe_conversion import *


### linear dynamics

def linearize_6dof(oe, qw, time, J):
    """
    linearize the nonlinear dynamics of 6DoF relative motion 
    input: 
        oe: OE (chief);        6 x (n_time)
        qw: attitude (deputy); 7 x (n_time)
        time: time;            1 x (n_time)
        J: inertia matrix
    return:
        mats: dictionary containing the linearized matrices
            stm: state transition matrix;                   6 x 6 x n_time
            stm_qw: state transition matrix for attitude;   7 x 7 x n_time
            cim: control input matrix;                      6 x 3 x n_time
            cim_qw: control input matrix for attitude;      7 x 3 x n_time
            psi: ROE -> RTN map;                            6 x 6 x n_time
            rs: residual for translational motion;          6 x n_time
            rqw: residual for rotational motion;            7 x n_time
    """
    
    n_time = len(time) 
    dt = time[1]-time[0]

    psi      = np.empty(shape=(6, 6, n_time), dtype=float)   # ROE -> RTN map 
    stm      = np.empty(shape=(6, 6, n_time),   dtype=float)
    stm_qw   = np.empty(shape=(7, 7, n_time),   dtype=float)
    cim      = np.empty(shape=(6, 3, n_time),   dtype=float)    
    cim_qw   = np.empty(shape=(7, 3, n_time),   dtype=float)
    rs       = np.empty(shape=(6, n_time),      dtype=float)
    rqw      = np.empty(shape=(7, n_time),      dtype=float)

    # Time Loop
    for i in range(n_time):
        
        # ROE to RTN map (TODO: check only translational motion matters (i think so?))
        psi[:,:,i] = map_mtx_roe_to_rtn(oe[:,i]) # N.B.: has dimension [-,-,-,1/s,1/s,1/s]
        
        if i < n_time:
            stm[:,:,i]    = state_transition_roe(oe[:,i],dt)   # N.B.: is a-dimentional
            cim[:,:,i]    = control_input_matrix_roe(oe[:,i])  # N.B.: has dimension [s]
            stm_qw[:,:,i] = get_stm_qw(qw[:,i],dt,J) 
            cim_qw[:,:,i] = get_cim_qw(qw[:,i],J) 
            
        # update residuals 
        # TODO: currently trapezoidal approximation. We can do better (Malyuta's rendezvous paper has 3rd order approx.)    
        if i > 0: 
            Aqq_i,   Aqw_i,   Aww_i   = dyn_qw_lin(qw[:,i-1], J)
            Aqq_ip1, Aqw_ip1, Aww_ip1 = dyn_qw_lin(qw[:,i], J)
            
            dq_i   = q_kin(qw[0:4, i-1], qw[4:7, i-1])  # _{i}
            dq_ip1 = q_kin(qw[0:4, i],   qw[4:7, i]  )   # _{i+1}
            dw_i   = euler_dyn(qw[4:7, i-1], J, np.array([0,0,0]))
            dw_ip1 = euler_dyn(qw[4:7, i],   J, np.array([0,0,0]))
            
            dqw_i   = np.concatenate((dq_i, dw_i))
            dqw_ip1 = np.concatenate((dq_ip1, dw_ip1))
            
            A_i   = np.block([[Aqq_i,   Aqw_i  ],[np.zeros((3,4)), Aww_i  ]])
            # A_ip1 = np.block([[Aqq_ip1, Aqw_ip1],[np.zeros((3,4)), Aww_ip1]])
            
            rs[:,i-1]  = np.zeros((6,1)).reshape((6,))
            # rqw[:,i-1] = (stm_qw[:,:,i-1] @ (dqw_i - A_i @ qw[:,i-1]) + \
            #             dqw_ip1 - A_ip1 @ qw[:,i]) * dt/2
            rqw[:,i-1] = stm_qw[:,:,i-1] @ (dqw_i - A_i @ qw[:,i-1]) * dt  
        
    mats = {"stm_r": stm, "stm_qw": stm_qw, "cim_r": cim, "cim_qw": cim_qw, "psi": psi, "rs": rs, "rqw": rqw}

    return mats


### nonlinear dynamics 

def get_6dof_discrete_control(x_ref, action, koe_c, mu_E, t, J, j2=True, drag=False):
    
    x_0 = x_ref[:,0]
    
    if koe_c.shape[1] != len(t):
        raise ValueError("koe_c and t must have the same length of timestep. Re-check your input...")
    
    roe_0 = x_0[0:6] / koe_c[0,0]   # non-dimensionalize
    qw_0  = x_0[6:13]
    
    koe_c_0 = koe_c[:,0]       
    rc, vc  = koe_to_rv(koe_c_0, mu_E)   
    koe_d_0 = qnsroe_to_koe(roe_0, koe_c_0) 

    koe_c_list  = np.empty(shape=(6,len(t)), dtype=float)
    koe_d_list  = np.empty(shape=(6,len(t)), dtype=float)
    qw_list  = np.empty(shape=(7,len(t)), dtype=float)
    roe_list = np.empty(shape=(6,len(t)), dtype=float)
    
    # oe_list[:,0] = koe_d.reshape((6,))
    koe_c_list[:,0] = koe_c_0.reshape((6,))
    koe_d_list[:,0] = koe_d_0.reshape((6,))
    qw_list[:,0]    = qw_0.reshape((7,))
    roe_list[:,0]   = roe_0.reshape((6,))  # non-dimensionalized
    
    for i in range(len(t)-1):
        dv = action[0:3,i]   # in the chief's RTN frame 
        dm = action[3:6,i]   # in the deputy's body frame
        rc, vc = koe_to_rv(koe_c_list[:,i], mu_E)  
        rd, vd = koe_to_rv(koe_d_list[:,i], mu_E)   
        urtn_c = unit_rtn(rc, vc)
        urtn_d = unit_rtn(rd, vd)
        dv = transform_coordinates(urtn_c,urtn_d,dv)  # chief's RTN -> deputy's RTN
        
        F = dv / (t[i+1] - t[i])   # in deputy's RTN frame 
        T = dm / (t[i+1] - t[i])   # in deputy's body frame
        
        time = np.array([0, t[i+1]-t[i]])
        x0   = np.concatenate((roe_list[:,i], koe_c[:,i]))
        roe_ = odeint(gve_qnsroe_full, x0, time, args=(mu_E, F, j2, drag))
        qw_  = odeint(ode_qw, qw_list[:,i].flatten(), time, args=(J, T))
        
        roe_list[:,i+1]    = roe_[-1,0:6].reshape((6,)) 
        qw_list[:,i+1]     = qw_[-1,:].reshape((7,))
        koe_c_list[:,i+1]  = roe_[-1, 6:12].reshape((6,))  # chief
        koe_d_list[:,i+1]  = qnsroe_to_koe(roe_list[:, i+1], koe_c_list[:,i+1]).reshape((6,))  # deputy
    
    # post-processing: dimensionalize 
    for i in range(len(t)):
        roe_list[:,i] = roe_list[:,i] * koe_c_list[0,i] 

    return koe_c_list, qw_list, roe_list,