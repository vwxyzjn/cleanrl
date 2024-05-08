import numpy as np 


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
    
    ϕ = np.arcsin(d/r*np.sin(abs(θ)))
    if abs(ϕ) < np.pi/2:
        ϕ = np.pi - ϕ
    
    λ = np.pi - θ - ϕ
    
    if θ > 0: 
        rvec = np.array([r*np.cos(λ), r*np.sin(λ)])
    else: 
        rvec = np.array([r*np.cos(λ), -r*np.sin(λ)])
    
    torque = np.cross(rvec, fvec)
    
    return torque 