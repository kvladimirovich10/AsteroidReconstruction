import numpy as np
from scipy.spatial.transform import Rotation as R


def generate_inertia_body_matrices(a, b, c, mass):
    I_body = generate_inertia_body_matrix(a, b, c, mass)
    
    I_body_inv = np.linalg.inv(I_body)
    
    return I_body, I_body_inv


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


def matrix_to_matrix(matrix):
    return R.from_matrix(matrix).as_quat()

def ode(body):
    pass


