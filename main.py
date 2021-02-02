import sys

import numpy as np
from model.ellipsoid import Ellipsoid
import math
import matplotlib.pyplot as plt
from util import measurer
from util import RayMarching as rm


def init_ell():
    mass = 1000
    x = np.array([10000, 0, 0])
    semi_axes = {'a': 1, 'b': 2, 'c': 1.3}
    euler_angles = {'alpha': 0, 'beta': 0, 'gamma': 0}
    
    P = np.array([0, 0, 0])
    L = np.array([-5, 3, 10])
    # L = np.array([0, 0, 0])
    
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


def main():
    ellipsoid = init_ell()
    
    seconds = 80
    rate = 50
    time_step = 1 / rate
    
    print('\nMOTION MODELING PART')
    y = ellipsoid.body_to_array()
    
    for i in range(seconds * rate):
        sys.stdout.write(f'\r{i}/{seconds * rate}')
        sys.stdout.flush()
        y = ellipsoid.update_position(y, time_step)
    
    print('\nRAY MARCHING PART')
    observation_point = [0, 0, 0]
    grid_side_point_number = 50
    rm.ellipsoid_ray_marching(ellipsoid, observation_point, grid_side_point_number,
                              make_ray_marching_image=False, make_radio_image=True)

    # rate = 25
    # time_step = 1/rate
    # ellipsoid.motion_visualisation(time_step, rate)


main()
