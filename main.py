import numpy as np
from model.ellipsoid import Ellipsoid
import math
import matplotlib.pyplot as plt
from util import measurer


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


def get_dist(ell, point):
    return math.sqrt(pow((point[0] - ell.x[0]) * ell.b * ell.c, 2) + \
                     pow((point[1] - ell.x[1]) * ell.a * ell.c, 2) + \
                     pow((point[2] - ell.x[2]) * ell.a * ell.b, 2)) - \
           ell.a * ell.b * ell.c


def get_point_in_ell_system(self, point):
    return np.matmul(self.rotation_matrix, point)


def visualisation_test(ell):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d', proj_type='ortho')
    
    n = 10
    for i in range(n):
        point = [4, -n / 2 + i, 0]
        distance, point_projection = measurer._get_closest_dist(ell, point)
        print(distance)
        ax.scatter(point[0], point[1], point[2], s=2, marker='*')
        ax.scatter(point_projection[0], point_projection[1], point_projection[2], s=2, marker='o')

        point = [-n / 2 + i, 4, 0]
        distance, point_projection = measurer._get_closest_dist(ell, point)
        print(distance)
        ax.scatter(point[0], point[1], point[2], s=2, marker='*')
        ax.scatter(point_projection[0], point_projection[1], point_projection[2], s=2, marker='o')

        point = [0, 4, -n / 2 + i]
        distance, point_projection = measurer._get_closest_dist(ell, point)
        print(distance)
        ax.scatter(point[0], point[1], point[2], s=2, marker='*')
        ax.scatter(point_projection[0], point_projection[1], point_projection[2], s=2, marker='o')
    
    ax.view_init(elev=90, azim=-90)
    ax.axis('auto')
    plt.show()


def main():
    ellipsoid = init_ell()
    
    visualisation_test(ellipsoid)
    
    # rate = 25
    # time_step = 1/rate
    # ell.motion_visualisation(time_step, rate)


main()
