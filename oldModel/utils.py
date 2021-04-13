import numpy as np
from scipy.spatial.transform import Rotation as R
from itertools import chain


def generate_inertia_body_matrix(a, b, c, mass):
    I_a = pow(b, 2) + pow(c, 2)
    I_b = pow(c, 2) + pow(a, 2)
    I_c = pow(a, 2) + pow(b, 2)
    coef = mass / 5
    
    return coef * np.array([[I_a, 0, 0], [0, I_b, 0], [0, 0, I_c]])


def star_omega(omega):
    return np.array([[0, -omega[2], omega[1]],
                     [omega[2], 0, -omega[0]],
                     [-omega[1], omega[0], 0]])


def matrix_to_array(matrix):
    return list(chain.from_iterable(matrix))


def generate_quaternion_by_angels(alpha, beta, gamma):
    r_matrix = R.from_euler('xyz', [alpha, beta, gamma], degrees=True)
    return r_matrix.as_quat()


def rot_matrix_by_angles(alpha, beta, gamma):
    r_matrix = R.from_euler('xyz', [alpha, beta, gamma], degrees=True)
    return r_matrix.as_matrix()


def quaternion_to_matrix(q):
    return R.from_quat(q, normalized=True).as_matrix()


def solver(el, y, time_step):
    k1 = el.dy_dt_to_array(y)
    k2 = el.dy_dt_to_array(y + time_step * k1 / 2)
    k3 = el.dy_dt_to_array(y + time_step * k2 / 2)
    k4 = el.dy_dt_to_array(y + time_step * k3)
    
    y_new = y + time_step * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return y_new
