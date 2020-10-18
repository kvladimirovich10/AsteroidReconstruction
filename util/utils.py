import numpy as np
from scipy.spatial.transform import Rotation as R
from vpython import *
from math import *
from numpy import *


def generate_inertia_body_matrix(a, b, c, mass):
    I_a = pow(b, 2) + pow(c, 2)
    I_b = pow(c, 2) + pow(a, 2)
    I_c = pow(a, 2) + pow(b, 2)
    coef = mass / 5
    
    return coef * np.array([[I_a, 0, 0], [0, I_b, 0], [0, 0, I_c]])


def generate_quaternion_by_angels(alpha, beta, gamma):
    r_matrix = R.from_euler('xyz', [alpha, beta, gamma], degrees=True)
    return r_matrix.as_quat()


def quaternion_to_matrix(q):
    return R.from_quat(q, normalized=True).as_matrix()


def solver(el, y, time_step):
    k1 = el.get_dy_dt_array(y)
    k2 = el.get_dy_dt_array(y + time_step * k1 / 2)
    k3 = el.get_dy_dt_array(y + time_step * k2 / 2)
    k4 = el.get_dy_dt_array(y + time_step * k3)
    
    return y + time_step * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def vector_from_array(array):
    return vector(array[0], array[1], array[2])


def array_from_vector(vector):
    return np.array(vector.value)


def rotate(body, ell):
    body.rotate(angle=ell.omega[0], axis=vector(1, 0, 0))
    body.rotate(angle=ell.omega[1], axis=vector(0, 1, 0))
    body.rotate(angle=ell.omega[2], axis=vector(0, 0, 1))


def translate(body, ell):
    body.pos = body.pos + vector(ell.v[0], ell.v[1], ell.v[2])


def R_from_2vec(vector_orig, vector_fin):
    R = np.zeros((3, 3))
    
    vector_orig = vector_orig / np.linalg.norm(vector_orig)
    vector_fin = vector_fin / np.linalg.norm(vector_fin)
    
    axis = cross(vector_orig, vector_fin)
    axis_len = np.linalg.norm(axis)
    
    try:
        axis = axis / axis_len
        
        x = axis[0]
        y = axis[1]
        z = axis[2]
        
        angle = acos(np.dot(vector_orig, vector_fin))
        
        ca = cos(angle)
        sa = sin(angle)
        
        R[0][0] = 1.0 + (1.0 - ca) * (pow(x, 2) - 1.0)
        R[0][1] = -z * sa + (1.0 - ca) * x * y
        R[0][2] = y * sa + (1.0 - ca) * x * z
        R[1][0] = z * sa + (1.0 - ca) * x * y
        R[1][1] = 1.0 + (1.0 - ca) * (pow(y, 2) - 1.0)
        R[1][2] = -x * sa + (1.0 - ca) * y * z
        R[2][0] = -y * sa + (1.0 - ca) * x * z
        R[2][1] = x * sa + (1.0 - ca) * y * z
        R[2][2] = 1.0 + (1.0 - ca) * (pow(z, 2) - 1.0)
        
        return R
    
    except:
        print('axis_len = 0')
