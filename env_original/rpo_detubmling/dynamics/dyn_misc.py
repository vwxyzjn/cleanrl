### miscellaneous functions for dynamics ### 

import numpy as np
import numpy.linalg as la


# skew matrix 
def skw(a):
    return np.array([[0, -a[2], a[1]],
                     [a[2], 0, -a[0]],
                     [-a[1], a[0], 0]])
    

def unit_rtn(r,v):
    # radial vector
    ur = r/la.norm(r)
    un = np.cross(r,v)/la.norm(np.cross(r,v))
    ut = np.cross(un,ur)
    
    return np.vstack((ur, ut, un))

# Rotation matrix 
def Rx(x):
    mat = np.array([[1, 0, 0],
                    [0, np.cos(x), np.sin(x)],
                    [0, -np.sin(x), np.cos(x)]])
    return mat

def Ry(x):
    mat = np.array([[np.cos(x), 0, -np.sin(x)],
                    [0, 1, 0],
                    [np.sin(x), 0, np.cos(x)]])
    return mat

def Rz(x):
    mat = np.array([[np.cos(x), np.sin(x), 0],
                    [-np.sin(x), np.cos(x), 0],
                    [0, 0, 1]])
    return mat


def transform_coordinates(u_vectors, v_vectors, point_u):
    """
    Transform coordinates from one system to another.
    inputs:    
        u_vectors: A 3x3 numpy array representing the unit vectors of the original coordinate system
        v_vectors: A 3x3 numpy array representing the unit vectors of the new coordinate system
        point_u: The coordinates of the point in the original system
    return:
        The coordinates of the point in the new system
    """
    # Construct the transformation matrix
    T = np.array([[np.dot(v, u) for u in u_vectors] for v in v_vectors])

    return np.dot(T, point_u)


# Kepler equation solver 
def solve_kepler(M, e):
    E = np.pi
    for _ in range(100):
        E_next = E - (E - e*np.sin(E) - M)/(1 - e*np.cos(E))

        if abs(E_next - E) < 1e-5:
            break

    return E_next