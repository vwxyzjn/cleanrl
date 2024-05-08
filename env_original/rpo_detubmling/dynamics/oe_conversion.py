"""
    Conversions of orbital elemnts: 
    - koe_to_rv
    - rv_to_koe
    - qnsroe_to_koe
    - cart2rtn    
"""

import numpy as np
import numpy.linalg as la
from dyn_misc import *  

def koe_to_rv(koe, mu):
    """
        inputs: 
            koe: [a,e,i,Omega,omega,M] (Keplarian orbital elements)
            mu: gravitational parameter of primary
        output: 
            rv: 6-element [r,v] in Cartesian coordinate
    """
    
    a,e,i,Omega,omega,M = koe
    
    E = solve_kepler(M, e)
    nu = 2*np.arctan(np.sqrt((1+e)/(1-e))*np.tan(E/2))

    p = a*(1-e**2)
    r = p/(1+e*np.cos(nu))    
    
    # perifocal frame 
    rp = r*np.array([np.cos(nu), np.sin(nu), 0]).reshape((3,1))
    vp = np.sqrt(mu/p)*np.array([-np.sin(nu), e+np.cos(nu), 0]).reshape((3,1))
    
    # rotation matrix from perifocal to ECI
    r_cart = Rz(-Omega).dot(Rx(-i)).dot(Rz(-omega)).dot(rp)
    v_cart = Rz(-Omega).dot(Rx(-i)).dot(Rz(-omega)).dot(vp)
    
    # return np.concatenate((r_cart, v_cart), axis=0) 
    return r_cart.flatten(), v_cart.flatten()


def rv_to_koe(rv, mu):
    """
        inputs: 
            rv: 6-element [r,v] in Cartesian coordinate
            mu: gravitational parameter of primary
        output: 
            oe: [a,e,i,Omega,omega,M] (Keplarian orbital elements)
    """
    rvec = rv[0:3]
    vvec = rv[3:6]
    hvec = np.cross(rvec, vvec)
    
    r = la.norm(rvec)
    v = la.norm(vvec)
    h = la.norm(hvec)
        
    nvec = np.cross(np.array([0,0,1]), hvec) / np.linalg.norm(np.cross(np.array([0,0,1]), hvec))
    
    ene = 1/2*v**2 - mu/r
    a = -mu/(2*ene)
    evec = (np.cross(vvec, hvec) - mu*rvec/r) / mu
    e = la.norm(evec)
    i = np.arccos(hvec[2]/h)
    Omega = np.arctan2(nvec[1], nvec[0])
    omega = np.arccos(np.dot(evec, nvec)/e)
    if np.dot(evec, np.array([0,0,1])) < 0:
        omega = -omega 
        
    nu = np.arccos(np.dot(evec, rvec)/(e*r))
    if np.dot(rvec, vvec) < 0:
        nu = -nu
    
    if e > 1 - 1e-5:
        raise ValueError('Hyperbolic orbit')
    
    E = 2 * np.arctan(np.sqrt((1-e)/(1+e))*np.tan(nu/2))
    M = E - e*np.sin(E)
    
    return np.array([a, e, i, Omega, omega, M])


def qnsroe_to_koe(qnsroe, koe_c):
    """
    return deputy's KOE based on the chief's KOE and the deputy's ROE 
    inputs: 
        qnsroe: [da, dl, dex, dey, dix, diy]
        koe_c: [a,e,i,Omega,omega,M]
    returns:
        koe_d: [ad, ed, id, Omegad, omegad, Md]
    """
    
    da  = qnsroe.item(0)
    dl  = qnsroe.item(1)
    dex = qnsroe.item(2)
    dey = qnsroe.item(3)
    dix = qnsroe.item(4)
    diy = qnsroe.item(5)
    
    a     = koe_c.item(0)
    e     = koe_c.item(1)
    i     = koe_c.item(2)
    Omega = koe_c.item(3)
    omega = koe_c.item(4)
    M     = koe_c.item(5)
    
    ad  = da * a + a
    exd = dex + e*np.cos(omega)
    eyd = dey + e*np.sin(omega)
    ed  = np.sqrt(exd**2 + eyd**2)
    omegad = np.arctan2(eyd, exd)
    id  = dix + i 
    Omegad = diy / np.sin(i) + Omega
    Md  = dl - (Omegad - Omega) * np.cos(i) + (M + omega) - omegad
    
    return np.array([ ad, ed, id, Omegad, omegad, Md ]).reshape((6,1))


# returns the RTN coordinate frame of the deputy w.r.t. the chief's RTN
def cart2rtn(cart_d, cart_c):

    r_d = cart_d[0:3]
    v_d = cart_d[3:6]
    r_c = cart_c[0:3]
    v_c = cart_c[3:6]
    
    r_rel = r_d - r_c 
    v_rel = v_d - v_c
    
    # radial vector
    ur = r_d/la.norm(r_c)
    un = np.cross(r_c,v_c)/la.norm(np.cross(r_c,v_c))
    ut = np.cross(un,ur)

    # relative position vector in RTN
    r_r = np.dot(r_rel,ur)
    r_t = np.dot(r_rel,ut)
    r_n = np.dot(r_rel,un)
    v_r = np.dot(v_rel,ur)
    v_t = np.dot(v_rel,ut)
    v_n = np.dot(v_rel,un)

    return np.array([r_r, r_t, r_n, v_r, v_t, v_n])