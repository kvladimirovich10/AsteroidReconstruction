from __future__ import division, print_function
import numpy as np
from model import Ellipsoid


def main():
    mass = 1000
    x = np.array([0, 0, 0])
    semi_axes = {'a': 2.5, 'b': 1.5, 'c': 1.3}
    euler_angles = {'alpha': 0, 'beta': 0, 'gamma': 0}
    
    P = np.array([1, 1, 1])
    L = np.array([2.5, -3, 1])
    
    force = np.array([0, 0, 0])
    torque = np.array([0, 0, 0])
    
    ell = Ellipsoid(semi_axes, x, mass, euler_angles, P, L, force, torque)
    
    ell.motion_visualisation()


main()
