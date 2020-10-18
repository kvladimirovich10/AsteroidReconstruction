from __future__ import division, print_function
import numpy as np
from model import Ellipsoid
from math import *
from numpy.linalg import *
import matplotlib.pyplot as plt


def init_ell():
    mass = 100
    x = np.array([0, 0, 0])
    semi_axes = {'a': 2, 'b': 1.5, 'c': 1.3}
    euler_angles = {'alpha': 0, 'beta': 0, 'gamma': 0}
    
    P = np.array([1, 1, 1])
    L = np.array([2.5, -3, 1])
    
    force = np.array([0, 0, 0])
    torque = np.array([0, 0, 0])
    
    return Ellipsoid(semi_axes, x, mass, euler_angles, P, L, force, torque)


def get_xyz_from_polar(ell, phi, theta):
    return np.array([ell.a * cos(phi) * cos(theta), ell.b * cos(phi) * sin(theta), ell.c * sin(phi)])


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
    return norm(get_xyz_from_polar(ell, *x_old) - get_xyz_from_polar(ell, *x_new))


def get_closest_dist(ell, point, eps=0.00001):
    k = 0
    
    x_old = get_initial_solution(point, ell)
    x_new = np.ones(2)
    
    while metrics(ell, x_old, x_new) > eps and k < 100:
        w = get_w_matrix(point, ell, x_old[0], x_old[1])
        f = get_tan_eq_vec(point, ell, x_old[0], x_old[1])
        
        x_new = x_old - np.matmul(inv(w), f)
        x_old = x_new
        k += 1
    
    point_projection = get_xyz_from_polar(ell, x_new[0], x_new[1])
    return norm(point - point_projection), point_projection


def test():
    # -----------------------------
    
    ell = init_ell()
    
    # -----------------------------
    
    point = np.array([4, 3, 2])
    
    # -----------------------------
    
    eps = 0.001
    x_old = get_initial_solution(point, ell)
    x_new = np.ones(2)
    k = 0
    while metrics(x_old, x_new) > eps and k < 100:
        print(metrics(x_old, x_new))
        w = get_w_matrix(point, ell, x_old[0], x_old[1])
        f = get_tan_eq_vec(point, ell, x_old[0], x_old[1])
        
        x_new = x_old - np.matmul(inv(w), f)
        x_old = x_new
        k += 1
    
    point_projection = get_xyz_from_polar(ell, x_new[0], x_new[1])
    print(f'distance = {norm(point - point_projection)}')
    
    # -----------------------------
    
    n = 100
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d', proj_type='ortho')
    
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    
    x = ell.a * np.outer(cos(u), sin(v))
    y = ell.b * np.outer(sin(u), sin(v))
    z = ell.c * np.outer(np.ones_like(u), cos(v))
    
    ax.plot_surface(x, y, z, rstride=5, cstride=5, color='b')
    
    ax.scatter(point[0], point[1], point[2], s=30, marker='*', c='r')
    ax.scatter(point_projection[0], point_projection[1], point_projection[2], s=30, marker='^', c='r')
    
    for axis in 'xyz':
        getattr(ax, 'set_{}lim'.format(axis))((-5, 5))
    
    # ax.view_init(elev=0, azim=-90)
    
    plt.show()
    
    # -----------------------------
