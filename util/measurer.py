from __future__ import division, print_function
import numpy as np
from model import Ellipsoid
from math import *
from numpy.linalg import *
import matplotlib.pyplot as plt


def init_ell():
    mass = 100
    x = np.array([0, 0, 0])
    semi_axes = {'a': 2, 'b': 4, 'c': 2}
    euler_angles = {'alpha': 30, 'beta': 0, 'gamma': 0}
    
    P = np.array([1, 1, 1])
    L = np.array([2.5, -3, 1])
    
    force = np.array([0, 0, 0])
    torque = np.array([0, 0, 0])
    
    return Ellipsoid(semi_axes, x, mass, euler_angles, P, L, force, torque)


def get_xyz_from_polar_rotated(ell, phi, theta):
    a, b, c = ell.a, ell.b, ell.c
    R = ell.rotation_matrix
    R_inv = np.linalg.inv(R)
    
    x_origin = np.array([a * cos(phi) * cos(theta),
                         b * cos(phi) * sin(theta),
                         c * sin(phi)])
    
    return np.dot(R_inv, x_origin) - ell.x


def get_f_w_rotated(point, ell, phi, theta):
    a, b, c = ell.a, ell.b, ell.c
    R = ell.rotation_matrix
    
    x_origin = np.array([a * cos(phi) * cos(theta),
                         b * cos(phi) * sin(theta),
                         c * sin(phi)])
    
    x_origin_d_theta = np.array([-a * cos(phi) * sin(theta),
                                 b * cos(phi) * cos(theta),
                                 0])
    
    x_origin_d_phi = np.array([-a * sin(phi) * cos(theta),
                               -b * sin(phi) * sin(theta),
                               c * cos(phi)])
    
    x_origin_d2_theta = np.array([-a * cos(phi) * cos(theta),
                                  -b * cos(phi) * sin(theta),
                                  0])
    
    x_origin_d2_phi = np.array([-a * cos(phi) * cos(theta),
                                -b * cos(phi) * sin(theta),
                                -c * sin(phi)])
    
    x_origin_d2_phi_theta = np.array([a * sin(phi) * sin(theta),
                                      -b * sin(phi) * cos(theta),
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


def get_initial_solution_rotated(point, ell):
    R_inv = np.linalg.inv(ell.rotation_matrix)
    point_ell_system = np.dot(R_inv, point - ell.x)
    
    a, b, c = ell.a, ell.b, ell.c
    x, y, z = point_ell_system[0], point_ell_system[1], point_ell_system[2]
    
    phi_0 = np.arctan2(z, c * sqrt((x / a) ** 2 + (y / b) ** 2))
    theta_0 = np.arctan2(a * y, b * x)
    
    return np.array([phi_0, theta_0])


def get_xyz_from_polar(ell, phi, theta):
    return np.array([ell.a * cos(phi) * cos(theta),
                     ell.b * cos(phi) * sin(theta),
                     ell.c * sin(phi)])


def get_tan_eq_vec(point, ell, phi, theta):
    a, b, c = ell.a, ell.b, ell.c
    x, y, z = point[0], point[1], point[2]
    
    f_theta = (a ** 2 - b ** 2) * cos(theta) * sin(theta) * cos(phi) \
              - x * a * sin(theta) + y * b * cos(theta)
    
    f_phi = ((a ** 2) * (cos(theta) ** 2) + (b ** 2) * (sin(theta) ** 2) - c ** 2) * sin(phi) * cos(phi) \
            - x * a * sin(phi) * cos(theta) \
            - y * b * sin(phi) * sin(theta) \
            + z * c * cos(phi)
    
    return np.array([f_phi, f_theta])


def get_w_matrix(point, ell, phi, theta):
    a, b, c = ell.a, ell.b, ell.c
    x, y, z = point[0], point[1], point[2]
    
    a11 = (a ** 2 - b ** 2) * ((cos(theta) ** 2) - (sin(theta) ** 2)) * cos(phi) \
          - x * a * cos(theta) \
          - y * b * sin(theta)
    
    a12 = -(a ** 2 - b ** 2) * cos(theta) * sin(theta) * sin(phi)
    
    a21 = -2 * (a ** 2 - b ** 2) * cos(theta) * sin(theta) * sin(phi) * cos(phi) \
          + x * a * sin(phi) * sin(theta) \
          - y * b * sin(phi) * cos(theta)
    
    a22 = ((a ** 2) * (cos(theta) ** 2) + (b ** 2) * (sin(theta) ** 2) - c ** 2) * (
            (cos(phi) ** 2) - (sin(phi) ** 2)) \
          - x * a * cos(phi) * cos(theta) \
          - y * b * cos(phi) * sin(theta) \
          - z * c * sin(phi)
    
    return np.array([[a11, a12], [a21, a22]])


def get_initial_solution(point, ell):
    a, b, c = ell.a, ell.b, ell.c
    x, y, z = point[0], point[1], point[2]
    
    phi_0 = np.arctan2(z, c * sqrt((x / a) ** 2 + (y / b) ** 2))
    theta_0 = np.arctan2(a * y, b * x)
    
    return np.array([phi_0, theta_0])


def metrics(ell, x_old, x_new):
    return norm(get_xyz_from_polar_rotated(ell, *x_old) - get_xyz_from_polar_rotated(ell, *x_new))


def get_closest_dist(ell, point, eps=0.0001):
    k = 0
    np.set_printoptions(precision=3, suppress=True)
    
    x_old = get_initial_solution(point, ell)
    print(f'init angles {180 * x_old / pi}')
    x_new = np.ones(2)
    
    while metrics(ell, x_old, x_new) > eps and k < 10000:
        f = get_tan_eq_vec(point, ell, x_old[0], x_old[1])
        w = get_w_matrix(point, ell, x_old[0], x_old[1])
        
        x_new = x_old - np.matmul(inv(w), f)
        x_old = x_new
        k += 1
    
    print(k, metrics(ell, x_old, x_new))
    point_projection = get_xyz_from_polar_rotated(ell, *x_new)
    return norm(point - point_projection), point_projection


def get_closest_dist_rotated(ell, point, eps=0.0001):
    k = 0
    np.set_printoptions(precision=3, suppress=False)
    
    x_old = get_initial_solution_rotated(point, ell)
    print(f'init angles rotated {180 * x_old / pi}')
    x_new = np.ones(2)
    
    while metrics(ell, x_old, x_new) > eps and k < 10000:
        f_r, w_r = get_f_w_rotated(point, ell, x_old[0], x_old[1])
        
        x_new = x_old - np.matmul(inv(w_r), f_r)
        x_old = x_new
        k += 1
    
    print(x_new, metrics(ell, x_old, x_new))
    point_projection = get_xyz_from_polar_rotated(ell, *x_new)
    return norm(point - point_projection), point_projection


def scatter_point(point, eps, ell, ax):
    distance_r, point_projection_r = get_closest_dist_rotated(ell, point, eps)
    
    print(f'distance_r = {distance_r}\n'
          f'point_projection_r = {point_projection_r}')
    
    ax.scatter(point[0], point[1], point[2], s=2, marker='*')
    ax.scatter(point_projection_r[0], point_projection_r[1], point_projection_r[2], s=2, marker='o')


def test():
    # -----------------------------
    
    ell = init_ell()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d', proj_type='ortho')
    # -----------------------------
    
    eps = 0.001
    m = 100
    
    for i in range(m + 1):
        scatter_point(np.array([-m / 2 + i, 0, 5]), eps, ell, ax)
        #
        # print('')
        #
        # scatter_point(np.array([-m / 2 + i, 5, 0]), eps, ell, ax)
        #
        # print('')
        #
        # scatter_point(np.array([-m / 2 + i, -5, 0]), eps, ell, ax)
        #
        # print('')
        #
        # scatter_point(np.array([5, 0, -m / 2 + i]), eps, ell, ax)
        #
        # print('')
        #
        # scatter_point(np.array([-5, 0, -m / 2 + i]), eps, ell, ax)
        #
        # print('')
        #
        # scatter_point(np.array([5, -m / 2 + i, 0]), eps, ell, ax)
        #
        # print('')
        #
        # scatter_point(np.array([-5, -m / 2 + i, 0]), eps, ell, ax)
        #
        # print('')
        #
        # scatter_point(np.array([0, -5, -m / 2 + i]), eps, ell, ax)
        #
    for axis in 'xyz':
        getattr(ax, 'set_{}lim'.format(axis))((-5, 5))
    
    plt.show()
    
    # -----------------------------


test()
