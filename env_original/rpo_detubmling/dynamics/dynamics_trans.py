########### File containing relevant dynamics models

# Orbit dynamics
import numpy as np
import numpy.linalg as la
from dyn_misc import *
from oe_conversion import *

# Constants
J2 = 0.001082635819197
R_E = 6.3781363e+06    # Earth radius [m]
mu_E = 3.986004415e+14 # Earth gravitational parameter [m^3/s^2]

# Functions
def map_mtx_roe_to_rtn(oe, t=0):

    a = oe.item(0)
    u = oe.item(4) + oe.item(5)
    n = np.sqrt(mu_E/a**3)
    
    map_1 = np.array([1, 0, -np.cos(u), -np.sin(u), 0, 0]).reshape((1,6))
    map_2 = np.array([-(3/2)*n*t, 1, 2*np.sin(u), -2*np.cos(u), 0, 0]).reshape((1,6))
    map_3 = np.array([0, 0, 0, 0, np.sin(u), -np.cos(u)]).reshape((1,6))
    map_4 = np.array([0, 0, np.sin(u)*n, -np.cos(u)*n, 0, 0]).reshape((1,6))
    map_5 = np.array([-(3/2)*n, 0, 2*np.cos(u)*n, 2*np.sin(u)*n, 0, 0]).reshape((1,6))
    map_6 = np.array([0, 0, 0, 0, np.cos(u)*n, np.sin(u)*n]).reshape((1,6))

    map = np.concatenate((map_1, map_2, map_3, map_4, map_5, map_6), axis=0)

    return map

def map_roe_to_rtn(roe, oe, t=0):

    map = map_mtx_roe_to_rtn(oe, t)
    
    return map.dot(roe)

def map_rtn_to_roe(rtn, oe, t=0):

    map = map_mtx_roe_to_rtn(oe, t)
        
    return la.solve(map, rtn)

def state_transition_roe(oe, t):

    # From : Koenig A.W., Guffanti T., D'Amico S.; 
    # New State Transition Matrices for Spacecraft Relative Motion in Perturbed Orbits; 
    # Journal of Guidance, Control, and Dynamics, Vol. 40, No. 7, pp. 1749-1768 (September 2017).

    a = oe.item(0)
    e = oe.item(1)
    i = oe.item(2)
    w = oe.item(4)

    n = np.sqrt(mu_E/a**3)
    eta=np.sqrt(1-e**2)
    k=3/4*J2*R_E**2*np.sqrt(mu_E)/(a**(7/2)*eta**4)
    E=1+eta
    F=4+3*eta
    G=1/eta**2
    P=3*np.cos(i)**2-1
    Q=5*np.cos(i)**2-1
    S=np.sin(2*i)
    T=np.sin(i)**2

    w_dot=k*Q
    w_f=w+w_dot*t
    e_xi=e*np.cos(w)
    e_yi=e*np.sin(w)
    e_xf=e*np.cos(w_f)
    e_yf=e*np.sin(w_f)

    Phi_11=1
    Phi_12=0
    Phi_13=0
    Phi_14=0 
    Phi_15=0 
    Phi_16=0
    Phi_1 = np.array([Phi_11,Phi_12,Phi_13,Phi_14,Phi_15,Phi_16]).reshape((1,6))
    Phi_21= -(7/2*k*E*P+3/2*n)*t
    Phi_22=1
    Phi_23=k*e_xi*F*G*P*t
    Phi_24=k*e_yi*F*G*P*t
    Phi_25=-k*F*S*t
    Phi_26=0
    Phi_2 = np.array([Phi_21,Phi_22,Phi_23,Phi_24,Phi_25,Phi_26]).reshape((1,6))
    Phi_31=7/2*k*e_yf*Q*t
    Phi_32=0
    Phi_33=np.cos(w_dot*t)-4*k*e_xi*e_yf*G*Q*t
    Phi_34=-np.sin(w_dot*t)-4*k*e_yi*e_yf*G*Q*t
    Phi_35=5*k*e_yf*S*t
    Phi_36=0
    Phi_3 = np.array([Phi_31,Phi_32,Phi_33,Phi_34,Phi_35,Phi_36]).reshape((1,6))
    Phi_41=-7/2*k*e_xf*Q*t
    Phi_42=0
    Phi_43=np.sin(w_dot*t)+4*k*e_xi*e_xf*G*Q*t
    Phi_44=np.cos(w_dot*t)+4*k*e_yi*e_xf*G*Q*t
    Phi_45=-5*k*e_xf*S*t
    Phi_46=0
    Phi_4 = np.array([Phi_41,Phi_42,Phi_43,Phi_44,Phi_45,Phi_46]).reshape((1,6))
    Phi_51=0
    Phi_52=0
    Phi_53=0
    Phi_54=0
    Phi_55=1
    Phi_56=0
    Phi_5 = np.array([Phi_51,Phi_52,Phi_53,Phi_54,Phi_55,Phi_56]).reshape((1,6))
    Phi_61=7/2*k*S*t
    Phi_62=0
    Phi_63=-4*k*e_xi*G*S*t
    Phi_64=-4*k*e_yi*G*S*t
    Phi_65=2*k*T*t
    Phi_66=1
    Phi_6 = np.array([Phi_61,Phi_62,Phi_63,Phi_64,Phi_65,Phi_66]).reshape((1,6))

    return np.concatenate((Phi_1, Phi_2, Phi_3, Phi_4, Phi_5, Phi_6), axis=0)

def control_input_matrix_roe(oe):

    a = oe.item(0)
    u = oe.item(4) + oe.item(5)
    n = np.sqrt(mu_E/a**3)

    b_1 = np.array([0, 2/n, 0]).reshape((1,3))
    b_2 = np.array([-2/n, 0, 0]).reshape((1,3))
    b_3 = np.array([np.sin(u)/n, 2*np.cos(u)/n, 0]).reshape((1,3))
    b_4 = np.array([-np.cos(u)/n, 2*np.sin(u)/n, 0]).reshape((1,3))
    b_5 = np.array([0, 0, np.cos(u)/n]).reshape((1,3))
    b_6 = np.array([0, 0, np.sin(u)/n]).reshape((1,3))

    return np.concatenate((b_1, b_2, b_3, b_4, b_5, b_6), axis=0)

def dynamics_roe(state, action, oe, t):

    stm = state_transition_roe(oe, t)
    cim = control_input_matrix_roe(oe)

    new_state = stm.dot(state + cim.dot(action))

    return new_state

def kepler_dyanmics_oe(oe, dt):

    a = oe.item(0)
    n = np.sqrt(mu_E/a**3)
    new_oe = np.array([oe.item(0), oe.item(1), oe.item(2), oe.item(3), oe.item(4), oe.item(5) + n*dt]).reshape((6,))

    return new_oe

def dynamics_roe_optimization(oe_0, t_0, horizon, n_time):

    a = oe_0.item(0)
    n = np.sqrt(mu_E/a**3)
    period = 2*np.pi/n

    # Time discretization (given the number of samples defined in rpod_scenario)
    if n_time-1 != 0:
        dt = horizon*period/(n_time-1)
    else:
        dt = 0.

    stm = np.empty(shape=(6, 6, n_time-1), dtype=float)
    cim = np.empty(shape=(6, 3, n_time), dtype=float)
    psi = np.empty(shape=(6, 6, n_time), dtype=float)

    time = np.empty(shape=(n_time,), dtype=float)
    oe = np.empty(shape=(6, n_time), dtype=float)

    # Time 0
    time[0] = t_0
    oe[:,0] = oe_0

    cim[:,:,0] = control_input_matrix_roe(oe[:,0])
    psi[:,:,0] = map_mtx_roe_to_rtn(oe[:,0])

    # Time Loop
    for iter in range(n_time-1):
        
        # Precompute the STM
        stm[:,:,iter] = state_transition_roe(oe[:,iter],dt) # N.B.: is a-dimentional

        # Propagate reference orbit (this assumes keplerian dynamics on the OE, it is an approximation)
        time[iter+1] = time[iter] + dt
        oe[:,iter+1] = np.array([oe_0.item(0), oe_0.item(1), oe_0.item(2), oe_0.item(3), oe_0.item(4), oe_0.item(5) + n*(time.item(iter+1)-t_0)]).reshape((6,))

        # Control input matrix
        cim[:,:,iter+1] = control_input_matrix_roe(oe[:,iter+1]) # N.B.: has dimension [s]
        
        # ROE to RTN map
        psi[:,:,iter+1] = map_mtx_roe_to_rtn(oe[:,iter+1]) # N.B.: has dimension [-,-,-,1/s,1/s,1/s]

    return stm, cim, psi, oe, time, dt

def roe_to_rtn_horizon(roe, oe, n_time):
    
    rtn = np.empty(shape=(6, n_time), dtype=float)
    for i in range(n_time):
        rtn[:,i] = map_roe_to_rtn(roe[:,i], oe[:,i])

    return rtn

def state_transition_hcw(oe, t):

    a = oe.item(0)
    n = np.sqrt(mu_E/a**3)

    Phi_11=4-3*np.cos(n*t)
    Phi_12=6*(np.sin(n*t)-n*t)
    Phi_13=0
    Phi_14=3*n*np.sin(n*t) 
    Phi_15=6*n*(np.cos(n*t)-1)
    Phi_16=0
    Phi_1 = np.array([Phi_11,Phi_12,Phi_13,Phi_14,Phi_15,Phi_16]).reshape((6,1)) # first column
    Phi_21=0
    Phi_22=1
    Phi_23=0
    Phi_24=0
    Phi_25=0
    Phi_26=0
    Phi_2 = np.array([Phi_21,Phi_22,Phi_23,Phi_24,Phi_25,Phi_26]).reshape((6,1))
    Phi_31=0
    Phi_32=0
    Phi_33=np.cos(n*t)
    Phi_34=0
    Phi_35=0
    Phi_36=-n*np.sin(n*t)
    Phi_3 = np.array([Phi_31,Phi_32,Phi_33,Phi_34,Phi_35,Phi_36]).reshape((6,1))
    Phi_41=(1/n)*np.sin(n*t)
    Phi_42=(2/n)*(np.cos(n*t)-1)
    Phi_43=0
    Phi_44=np.cos(n*t)
    Phi_45=-2*np.sin(n*t)
    Phi_46=0
    Phi_4 = np.array([Phi_41,Phi_42,Phi_43,Phi_44,Phi_45,Phi_46]).reshape((6,1))
    Phi_51=(2/n)*(1-np.cos(n*t))
    Phi_52=(1/n)*(4*np.sin(n*t)-3*n*t)
    Phi_53=0
    Phi_54=2*np.sin(n*t)
    Phi_55=4*np.cos(n*t)-3
    Phi_56=0
    Phi_5 = np.array([Phi_51,Phi_52,Phi_53,Phi_54,Phi_55,Phi_56]).reshape((6,1))
    Phi_61=0
    Phi_62=0
    Phi_63=(1/n)*np.sin(n*t)
    Phi_64=0
    Phi_65=0
    Phi_66=np.cos(n*t)
    Phi_6 = np.array([Phi_61,Phi_62,Phi_63,Phi_64,Phi_65,Phi_66]).reshape((6,1))

    return np.concatenate((Phi_1, Phi_2, Phi_3, Phi_4, Phi_5, Phi_6), axis=1)

def state_transition_hcw_jac_dt(oe, t):

    a = oe.item(0)
    n = np.sqrt(mu_E/a**3)

    Phi_11=3*n*np.sin(n*t) 
    Phi_12=6*n*(np.cos(n*t)-1)
    Phi_13=0
    Phi_14=3*(n**2)*np.cos(n*t) 
    Phi_15=-6*(n**2)*np.sin(n*t)
    Phi_16=0
    Phi_1 = np.array([Phi_11,Phi_12,Phi_13,Phi_14,Phi_15,Phi_16]).reshape((6,1)) # first column
    Phi_21=0
    Phi_22=0
    Phi_23=0
    Phi_24=0
    Phi_25=0
    Phi_26=0
    Phi_2 = np.array([Phi_21,Phi_22,Phi_23,Phi_24,Phi_25,Phi_26]).reshape((6,1))
    Phi_31=0
    Phi_32=0
    Phi_33=-n*np.sin(n*t)
    Phi_34=0
    Phi_35=0
    Phi_36=-(n**2)*np.cos(n*t)
    Phi_3 = np.array([Phi_31,Phi_32,Phi_33,Phi_34,Phi_35,Phi_36]).reshape((6,1))
    Phi_41=np.cos(n*t)
    Phi_42=-2*np.sin(n*t)
    Phi_43=0
    Phi_44=-n*np.sin(n*t)
    Phi_45=-2*n*np.cos(n*t)
    Phi_46=0
    Phi_4 = np.array([Phi_41,Phi_42,Phi_43,Phi_44,Phi_45,Phi_46]).reshape((6,1))
    Phi_51=2*np.sin(n*t)
    Phi_52=4*np.cos(n*t)-3
    Phi_53=0
    Phi_54=2*n*np.cos(n*t)
    Phi_55=-4*n*np.sin(n*t)
    Phi_56=0
    Phi_5 = np.array([Phi_51,Phi_52,Phi_53,Phi_54,Phi_55,Phi_56]).reshape((6,1))
    Phi_61=0
    Phi_62=0
    Phi_63=np.cos(n*t)
    Phi_64=0
    Phi_65=0
    Phi_66=-n*np.sin(n*t)
    Phi_6 = np.array([Phi_61,Phi_62,Phi_63,Phi_64,Phi_65,Phi_66]).reshape((6,1))

    return np.concatenate((Phi_1, Phi_2, Phi_3, Phi_4, Phi_5, Phi_6), axis=1)

def dynamics_hcw_optimization_dt_const(oe_0, t_0, horizon, n_time):

    a = oe_0.item(0)
    n = np.sqrt(mu_E/a**3)
    period = 2*np.pi/n

    # Time discretization (given the number of samples defined in rpod_scenario)
    if n_time-1 != 0:
        dt = horizon*period/(n_time-1)
    else:
        dt = 0.

    stm = np.empty(shape=(6, 6, n_time-1), dtype=float)
    stm_jac = np.empty(shape=(6, 6, n_time-1), dtype=float)

    time = np.empty(shape=(n_time,), dtype=float)
    oe = np.empty(shape=(6, n_time), dtype=float)
    dt_vect = np.empty(shape=(n_time-1,), dtype=float)

    # Time 0
    time[0] = t_0
    oe[:,0] = oe_0

    # Time Loop
    for iter in range(n_time-1):
        
        # Precompute the STM
        stm[:,:,iter] = state_transition_hcw(oe[:,iter],dt)
        stm_jac[:,:,iter] = state_transition_hcw_jac_dt(oe[:,iter],dt)
        dt_vect[iter] = dt

        # Propagate reference orbit (this assumes keplerian dynamics on the OE, it is an approximation)
        time[iter+1] = time[iter] + dt
        oe[:,iter+1] = np.array([oe_0.item(0), oe_0.item(1), oe_0.item(2), oe_0.item(3), oe_0.item(4), oe_0.item(5) + n*(time.item(iter+1)-t_0)]).reshape((6,))

    return stm, stm_jac, oe, time, dt_vect

def dynamics_hcw_optimization_dt_opt(oe_0, t_0, dt_opt):

    n_time = len(dt_opt) + 1

    a = oe_0.item(0)
    n = np.sqrt(mu_E/a**3)

    stm = np.empty(shape=(6, 6, n_time-1), dtype=float)
    stm_jac = np.empty(shape=(6, 6, n_time-1), dtype=float)

    time = np.empty(shape=(n_time,), dtype=float)
    oe = np.empty(shape=(6, n_time), dtype=float)

    # Time 0
    time[0] = t_0
    oe[:,0] = oe_0

    # Time Loop
    for iter in range(n_time-1):
        
        # Precompute the STM
        stm[:,:,iter] = state_transition_hcw(oe[:,iter],dt_opt[iter])
        stm_jac[:,:,iter] = state_transition_hcw_jac_dt(oe[:,iter],dt_opt[iter])

        # Propagate reference orbit (this assumes keplerian dynamics on the OE, it is an approximation)
        time[iter+1] = time[iter] + dt_opt[iter]
        oe[:,iter+1] = np.array([oe_0.item(0), oe_0.item(1), oe_0.item(2), oe_0.item(3), oe_0.item(4), oe_0.item(5) + n*(time.item(iter+1)-t_0)]).reshape((6,))

    return stm, stm_jac, oe, time

def check_ps_cart_hcw(states, oe_vec, n_time, horizon_safe, n_safe, DEED):

    states_ps = np.empty(shape=(6, n_time, n_safe), dtype=float)
    min_sep_ps = np.empty(shape=(n_time, ), dtype=float)

    a = oe_vec[0, 0]
    n = np.sqrt(mu_E/a**3)
    period = 2*np.pi/n
    # Time discretization (given the number of samples defined in rpod_scenario)
    if n_safe-1 != 0:
        dt_safe = horizon_safe*period/(n_safe-1)
    else:
        dt_safe = 0.

    for k in range(n_time):
        sep_k = np.empty(shape=(1, n_safe), dtype=float)
        oe_j = oe_vec[:, k]
        state_j = states[:, k]
        sep_k[:,0] = np.sqrt(np.transpose(state_j).dot(DEED.dot(state_j)))
        states_ps[:,k,0] = state_j
        for j in range(n_safe-1):
            stm_j = state_transition_hcw(oe_j, dt_safe)
            state_j = stm_j.dot(state_j)
            states_ps[:,k,j+1] = state_j
            sep_k[:,j+1] = np.sqrt(np.transpose(state_j).dot(DEED.dot(state_j)))
            oe_j = np.array([oe_j.item(0), oe_j.item(1), oe_j.item(2), oe_j.item(3), oe_j.item(4), oe_j.item(5) + n*dt_safe]).reshape((6,))      
        t_star_j = np.argmin(sep_k)
        min_sep_ps[k] = sep_k[:,t_star_j]
    
    return states_ps, min_sep_ps


### nonlinear dynamics 

# GVE for KOE [a, e, i, Omega, omega, M]
# F = [F_R, F_T, F_N]
def gve_koe(oe,t,mu,F, j2=True, drag=False):
    """
    GVE for KOE [a, e, i, Omega, omega, M]
    inputs:
        oe : orbital elements [a, e, i, Omega, omega, M]
        t  : time
        mu : gravitational parameter
        F  : non-two-body forces [F_R, F_T, F_N]
        j2 : boolean for J2 perturbation
    returns:
        d_oe : time derivative of the KOE
    """
    
    a = oe.item(0)
    e = oe.item(1)
    i = oe.item(2)
    Omega = oe.item(3)
    w = oe.item(4)
    M = oe.item(5)
    
    E = solve_kepler(M, e)
    nu = 2*np.arctan(np.sqrt((1+e)/(1-e))*np.tan(E/2))

    if np.isnan(nu):
        print("e = ", e)
        print("nu = ", nu)
        print(" X = ", (1+e)/(1-e))
    
    h = np.sqrt(mu*a*(1-e**2))
    n = np.sqrt(mu/a**3)
    p = a*(1-e**2)
    r = a*(1-e**2)/(1+e*np.cos(nu))
    L = Omega + w + nu

    if j2:
        J2 = 1082.64e-6
        Re = 6378.1363e3
        h_ = np.tan(i/2)*np.cos(Omega)
        k_ = np.tan(i/2)*np.sin(Omega)
        Delta_J2r = (-3  * mu * J2 * Re**2 / (2 * r**4)) * (1 - 12 * (h_ * np.sin(L) - k_ * np.cos(L))**2 / (1 + h_**2 + k_**2)**2)
        Delta_J2t = (-12 * mu * J2 * Re**2 / r**4) * ((h_ * np.sin(L) - k_ * np.cos(L)) * (h_ * np.cos(L) + k_ * np.sin(L)) / (1 + h_**2 + k_**2)**2)
        Delta_J2n = (-6  * mu * J2 * Re**2 / r**4) * ((1 - h_**2 - k_**2) * (h_ * np.sin(L) - k_ * np.cos(L)) / (1 + h_**2 + k_**2)**2)
        F = np.array([F[0] + Delta_J2r, F[1] + Delta_J2t, F[2] + Delta_J2n]).reshape((3,1))      
    
    if drag:
        rho = 0  # atmospheric density [kg/m^3] 
        S   = 0  # surface area [m^2]
        CD  = 0  # drag coefficient [-]
        # velocity in ECI (I know there should be a easier way to do this... later work)
        f = e * np.cos(w + Omega)
        g = e * np.sin(w + Omega)
        
        v = np.sqrt(mu * (2/r - 1/a))
        vr = np.sqrt(mu/p)*(f*np.sin(L) - g*np.cos(L))
        vt = np.sqrt(mu/p)*(1 + f*np.cos(L) + g*np.sin(L))
        
        Delta_Dr = - 0.5 * rho * S * CD * v * vr 
        Delta_Dt = - 0.5 * rho * S * CD * v * vt
        
        F = np.array([F[0] + Delta_Dr, F[1] + Delta_Dt, F[2]]).reshape((3,1))            
    
    # a
    A11 = (2 * a**2 * e * np.sin(nu)) / h
    A12 = (2 * a**2 * p) / (r * h)
    A13 = 0
    # e
    A21 = np.sqrt(1-e**2) / (n*a) * np.sin(nu)
    A22 = np.sqrt(1-e**2) / (n*a) * ( np.cos(nu) + (e+np.cos(nu))/(1+e*np.cos(nu)) )
    A23 = 0
    # i
    A31 = 0
    A32 = 0
    A33 = (r * np.cos(nu + w)) / h
    # Omega
    A41 = 0
    A42 = 0
    A43 = (r * np.sin(nu + w)) / (h * np.sin(i))
    # omega
    A51 = np.sqrt(1-e**2)/(n*a*e) * (- np.cos(nu))
    A52 = np.sqrt(1-e**2)/(n*a*e) * np.sin(nu) * (1 + r/p)
    A53 = - (r * np.sin(nu + w) * np.cos(i)) / (h * np.sin(i))
    # M
    A61 = h/mu * np.sqrt(1-e**2)/e * (np.cos(nu) - 2*e/(1-e**2)*r/a)
    A62 = - h/mu * np.sqrt(1-e**2)/e * (1+1/(1-e**2)*r/a)*np.sin(nu)
    A63 = 0

    row1 = np.array([A11, A12, A13]).reshape(1, 3)
    row2 = np.array([A21, A22, A23]).reshape(1, 3)
    row3 = np.array([A31, A32, A33]).reshape(1, 3)
    row4 = np.array([A41, A42, A43]).reshape(1, 3)
    row5 = np.array([A51, A52, A53]).reshape(1, 3)
    row6 = np.array([A61, A62, A63]).reshape(1, 3)

    A = np.concatenate((row1, row2, row3, row4, row5, row6), axis=0)
    B = np.array([0, 0, 0, 0, 0, n])   #.reshape(6,1)
    return (A.dot(F.flatten()) + B).flatten()


def gve_qnsroe_full(x, t, mu_E, F, j2=True, drag=False):
    """
    nonlinear GVE for the QNSROE dynamics
    input:
        x    : [qnsroe_d(t), koe_c(t)] (12 x 1)
        t    : time
        mu_E : gravitational parameter of the Earth
        F    : control input
        j2   : boolean for J2 perturbation
        drag : boolean for drag perturbation    
    """

    ac = x.item(6)
    ec = x.item(7)
    ic = x.item(8)
    Omegac = x.item(9)
    omegac = x.item(10)
    Mc = x.item(11)
        
    # re-scale the sma term 
    qnsroe = x[0:6]
    koe_d = qnsroe_to_koe(qnsroe, x[6:12]).flatten()
    
    ad = koe_d.item(0)
    ed = koe_d.item(1)
    id = koe_d.item(2)
    Omegad = koe_d.item(3)
    omegad = koe_d.item(4)
    Md = koe_d.item(5)
    
    F0 = np.array([0,0,0])
    
    dkoe_c = gve_koe(x[6:12], t, mu_E, F0, j2, drag)
    dkoe_d = gve_koe(koe_d, t, mu_E, F, j2, drag)

    dad     = dkoe_d.item(0)
    ded     = dkoe_d.item(1)
    did     = dkoe_d.item(2)
    dOmegad = dkoe_d.item(3)
    domegad = dkoe_d.item(4)
    dMd     = dkoe_d.item(5)
    
    dac     = dkoe_c.item(0)
    dec     = dkoe_c.item(1)
    dic     = dkoe_c.item(2)
    dOmegac = dkoe_c.item(3)
    domegac = dkoe_c.item(4)
    dMc     = dkoe_c.item(5)
    
    droe1 = ((dad - dac)*ac - qnsroe[0]*dac) / ac**2
    droe2 = (dMd + domegad) - (dMc + domegac) + (dOmegad - dOmegac) * np.cos(ic) - dic * (Omegad - Omegac) * np.sin(ic) 
    droe3 = ded * np.cos(omegad) - ed * domegad * np.sin(omegad) \
          - dec * np.cos(omegac) + ec * domegac * np.sin(omegac)
    droe4 = ded * np.sin(omegad) + ed * domegad * np.cos(omegad) \
          - dec * np.sin(omegac) - ec * domegac * np.cos(omegac)
    droe5 = did - dic  
    droe6 = (dOmegad - dOmegac) * np.sin(ic) + dic * (Omegad - Omegac) * np.cos(ic)
    
    # print(np.array([ droe1, droe2, droe3, droe4, droe5, droe6 ]).flatten())
    return np.array([ droe1, droe2, droe3, droe4, droe5, droe6, dac, dec, dic, dOmegac, domegac, dMc ]).flatten()        



### Linearization model 

def linearize_trans(oe, time):
    """
    linearize the nonlinear dynamics of 6DoF relative motion 
    input: 
        oe: OE (chief);        6 x (n_time+1)
        qw: attitude (deputy); 7 x (n_time+1)
        time: time;            1 x (n_time+1)
        J: inertia matrix
    return:
        mats: dictionary containing the linearized matrices
            stm: state transition matrix;                   6 x 6 x n_time
            stm_qw: state transition matrix for attitude;   7 x 7 x n_time
            cim: control input matrix;                      6 x 3 x n_time
            cim_qw: control input matrix for attitude;      7 x 3 x n_time
            psi: ROE -> RTN map;                            6 x 6 x (n_time+1)
            rs: residual for translational motion;          6 x n_time
            rqw: residual for rotational motion;            7 x n_time
    """
    
    n_time = len(time) - 1 
    dt = time[1]-time[0]

    psi      = np.empty(shape=(6, 6, n_time+1), dtype=float)   # ROE -> RTN map 
    stm      = np.empty(shape=(6, 6, n_time),   dtype=float)
    cim      = np.empty(shape=(6, 3, n_time),   dtype=float)    
    rs       = np.empty(shape=(6, n_time),      dtype=float)

    # Time Loop
    for i in range(n_time+1):
        
        # ROE to RTN map (TODO: check only translational motion matters (i think so?))
        psi[:,:,i] = map_mtx_roe_to_rtn(oe[:,i]) # N.B.: has dimension [-,-,-,1/s,1/s,1/s]
        
        if i < n_time:
            stm[:,:,i]    = state_transition_roe(oe[:,i],dt)  # N.B.: is a-dimentional
            cim[:,:,i]    = control_input_matrix_roe(oe[:,i])  # N.B.: has dimension [s]

        # no residual computation for now 
        
    mats = {"stm_r": stm, "cim_r": cim, "psi": psi, "rs": rs}

    return mats

