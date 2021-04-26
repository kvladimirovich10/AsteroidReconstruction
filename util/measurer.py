from __future__ import division, print_function
import numpy as np
import model.ellipsoid as ell
import math as m
import numpy.linalg as lg
from model.ellipsoid import Ellipsoid


def _get_xyz_from_polar_rotated(ell, phi, theta):
    a, b, c = ell.a, ell.b, ell.c
    R = ell.rotation_matrix
    R_inv = np.linalg.inv(R)
    
    x_origin = np.array([a * m.cos(phi) * m.cos(theta),
                         b * m.cos(phi) * m.sin(theta),
                         c * m.sin(phi)])
    
    return np.dot(R_inv, x_origin) - ell.x


def _get_f_w_rotated(point, ell, phi, theta):
    a, b, c = ell.a, ell.b, ell.c
    R = ell.rotation_matrix
    
    x_origin = np.array([a * m.cos(phi) * m.cos(theta),
                         b * m.cos(phi) * m.sin(theta),
                         c * m.sin(phi)])
    
    x_origin_d_theta = np.array([-a * m.cos(phi) * m.sin(theta),
                                 b * m.cos(phi) * m.cos(theta),
                                 0])
    
    x_origin_d_phi = np.array([-a * m.sin(phi) * m.cos(theta),
                               -b * m.sin(phi) * m.sin(theta),
                               c * m.cos(phi)])
    
    x_origin_d2_theta = np.array([-a * m.cos(phi) * m.cos(theta),
                                  -b * m.cos(phi) * m.sin(theta),
                                  0])
    
    x_origin_d2_phi = np.array([-a * m.cos(phi) * m.cos(theta),
                                -b * m.cos(phi) * m.sin(theta),
                                -c * m.sin(phi)])
    
    x_origin_d2_phi_theta = np.array([a * m.sin(phi) * m.sin(theta),
                                      -b * m.sin(phi) * m.cos(theta),
                                      0])
    
    x_rotated = ell.x + np.matmul(R, x_origin)
    
    x_rotated_d_theta = np.matmul(R, x_origin_d_theta)
    
    x_rotated_d_phi = np.matmul(R, x_origin_d_phi)
    
    x_rotated_d2_theta = np.matmul(R, x_origin_d2_theta)
    
    x_rotated_d2_phi = np.matmul(R, x_origin_d2_phi)
    
    x_rotated_d2_phi_theta = np.matmul(R, x_origin_d2_phi_theta)
    
    # ============================================
    
    f_phi = np.dot(point - x_rotated, x_rotated_d_phi)
    
    f_theta = np.dot(point - x_rotated, x_rotated_d_theta)
    
    f = np.array([f_phi, f_theta])
    
    # ============================================
    
    a11 = -1 * np.dot(x_rotated_d_phi, x_rotated_d_phi) + np.dot(point - x_rotated, x_rotated_d2_phi)
    a12 = -1 * np.dot(x_rotated_d_theta, x_rotated_d_phi) + np.dot(point - x_rotated, x_rotated_d2_phi_theta)
    
    a21 = -1 * np.dot(x_rotated_d_phi, x_rotated_d_theta) + np.dot(point - x_rotated, x_rotated_d2_phi_theta)
    a22 = -1 * np.dot(x_rotated_d_theta, x_rotated_d_theta) + np.dot(point - x_rotated, x_rotated_d2_theta)
    
    w = np.array([[a11, a12], [a21, a22]])
    
    return f, w


def _get_initial_solution_rotated(point, ell):
    R_inv = np.linalg.inv(ell.rotation_matrix)
    point_ell_system = np.dot(R_inv, point - ell.x)
    
    a, b, c = ell.a, ell.b, ell.c
    x, y, z = point_ell_system[0], point_ell_system[1], point_ell_system[2]
    
    phi_0 = np.arctan2(z, c * m.sqrt((x / a) ** 2 + (y / b) ** 2))
    theta_0 = np.arctan2(a * y, b * x)
    
    return np.array([phi_0, theta_0])


def _get_tan_eq_vec(point, ell, phi, theta):
    a, b, c = ell.a, ell.b, ell.c
    x, y, z = point[0], point[1], point[2]
    
    f_theta = (a ** 2 - b ** 2) * m.cos(theta) * m.sin(theta) * m.cos(phi) \
              - x * a * m.sin(theta) + y * b * m.cos(theta)
    
    f_phi = ((a ** 2) * (m.cos(theta) ** 2) + (b ** 2) * (m.sin(theta) ** 2) - c ** 2) * m.sin(phi) * m.cos(phi) \
            - x * a * m.sin(phi) * m.cos(theta) \
            - y * b * m.sin(phi) * m.sin(theta) \
            + z * c * m.cos(phi)
    
    return np.array([f_phi, f_theta])


def _get_w_matrix(point, ell, phi, theta):
    a, b, c = ell.a, ell.b, ell.c
    x, y, z = point[0], point[1], point[2]
    
    a11 = (a ** 2 - b ** 2) * ((m.cos(theta) ** 2) - (m.sin(theta) ** 2)) * m.cos(phi) \
          - x * a * m.cos(theta) \
          - y * b * m.sin(theta)
    
    a12 = -(a ** 2 - b ** 2) * m.cos(theta) * m.sin(theta) * m.sin(phi)
    
    a21 = -2 * (a ** 2 - b ** 2) * m.cos(theta) * m.sin(theta) * m.sin(phi) * m.cos(phi) \
          + x * a * m.sin(phi) * m.sin(theta) \
          - y * b * m.sin(phi) * m.cos(theta)
    
    a22 = ((a ** 2) * (m.cos(theta) ** 2) + (b ** 2) * (m.sin(theta) ** 2) - c ** 2) * (
            (m.cos(phi) ** 2) - (m.sin(phi) ** 2)) \
          - x * a * m.cos(phi) * m.cos(theta) \
          - y * b * m.cos(phi) * m.sin(theta) \
          - z * c * m.sin(phi)
    
    return np.array([[a11, a12], [a21, a22]])


def _get_closest_dist(ell, point, eps=0.0001):
    k = 0
    np.set_printoptions(precision=3, suppress=True)
    
    x_old = _get_initial_solution(point, ell)
    print(f'init angles {180 * x_old / m.pi}')
    x_new = np.ones(2)
    
    while metrics(ell, x_old, x_new) > eps and k < 10000:
        f = _get_tan_eq_vec(point, ell, x_old[0], x_old[1])
        w = _get_w_matrix(point, ell, x_old[0], x_old[1])
        
        x_new = x_old - np.matmul(lg.inv(w), f)
        x_old = x_new
        k += 1
    
    print(k, metrics(ell, x_old, x_new))
    point_projection = _get_xyz_from_polar(ell, *x_new)
    return lg.norm(point - point_projection), point_projection


def _get_xyz_from_polar(ell, phi, theta):
    return np.array([ell.a * m.cos(phi) * m.cos(theta),
                     ell.b * m.cos(phi) * m.sin(theta),
                     ell.c * m.sin(phi)])


def _get_f_w(point, ell, phi, theta):
    a, b, c = ell.a, ell.b, ell.c
    
    x = np.array([a * m.cos(phi) * m.cos(theta),
                  b * m.cos(phi) * m.sin(theta),
                  c * m.sin(phi)])
    
    x_d_theta = np.array([-a * m.cos(phi) * m.sin(theta),
                          b * m.cos(phi) * m.cos(theta),
                          0])
    
    x_d_phi = np.array([-a * m.sin(phi) * m.cos(theta),
                        -b * m.sin(phi) * m.sin(theta),
                        c * m.cos(phi)])
    
    x_d2_theta = np.array([-a * m.cos(phi) * m.cos(theta),
                           -b * m.cos(phi) * m.sin(theta),
                           0])
    
    x_d2_phi = np.array([-a * m.cos(phi) * m.cos(theta),
                         -b * m.cos(phi) * m.sin(theta),
                         -c * m.sin(phi)])
    
    x_d2_phi_theta = np.array([a * m.sin(phi) * m.sin(theta),
                               -b * m.sin(phi) * m.cos(theta),
                               0])
    
    # ============================================
    
    f_phi = np.dot(point - x, x_d_phi)
    
    f_theta = np.dot(point - x, x_d_theta)
    
    f = np.array([f_phi, f_theta])
    
    # ============================================
    
    a11 = -1 * np.dot(x_d_phi, x_d_phi) + np.dot(point - x, x_d2_phi)
    a12 = -1 * np.dot(x_d_theta, x_d_phi) + np.dot(point - x, x_d2_phi_theta)
    
    a21 = -1 * np.dot(x_d_phi, x_d_theta) + np.dot(point - x, x_d2_phi_theta)
    a22 = -1 * np.dot(x_d_theta, x_d_theta) + np.dot(point - x, x_d2_theta)
    
    w = np.array([[a11, a12], [a21, a22]])
    
    return f, w


def _get_initial_solution(point, ell):
    a, b, c = ell.a, ell.b, ell.c
    x, y, z = point[0], point[1], point[2]
    
    phi_0 = np.arctan2(z, c * m.sqrt((x / a) ** 2 + (y / b) ** 2))
    theta_0 = np.arctan2(a * y, b * x)
    
    return np.array([phi_0, theta_0])


def metrics(ell, x_old, x_new):
    return lg.norm(_get_xyz_from_polar(ell, *x_old) - _get_xyz_from_polar(ell, *x_new))


def get_closest_dist_rotated(ell: Ellipsoid, point, eps=0.001):
    R_inv = np.linalg.inv(ell.rot_matrix)
    point_in_ell_system = np.dot(R_inv, point - ell.x)
    k = 0
    np.set_printoptions(precision=3, suppress=False)
    
    x_old = _get_initial_solution(point_in_ell_system, ell)
    x_new = np.ones(2)
    
    while metrics(ell, x_old, x_new) > eps and k < 100:
        f_r, w_r = _get_f_w(point_in_ell_system, ell, x_old[0], x_old[1])
        
        x_new = x_old - np.matmul(lg.inv(w_r), f_r)
        x_old = x_new
        k += 1
    
    nonrotated_point_projection = _get_xyz_from_polar(ell, *x_new)
    
    return lg.norm(point_in_ell_system - nonrotated_point_projection), nonrotated_point_projection
