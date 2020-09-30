from __future__ import division, print_function

import numpy as np
from model import Ellipsoid
from vpython import *
from pyquaternion import Quaternion


def rotate(body, ell):
    body.rotate(angle=ell.omega[0], axis=vector(1, 0, 0))
    body.rotate(angle=ell.omega[1], axis=vector(0, 1, 0))
    body.rotate(angle=ell.omega[2], axis=vector(0, 0, 1))


def translate(body, ell):
    body.pos = body.pos + vector(ell.v[0], ell.v[1], ell.v[2])


def main():
    mass = 1000
    x = np.array([0, 0, 0])
    semi_axes = {'a': 2.5, 'b': 1.5, 'c': 1}
    euler_angles = {'alpha': 0, 'beta': 0, 'gamma': 0}
    
    P = np.array([0, 1, 1])
    L = np.array([1, 1, -1])
    
    force = np.array([0, 0, 0])
    torque = np.array([0, 0, 0])
    
    ell = Ellipsoid(semi_axes, x, mass, euler_angles, P, L, force, torque)
    
    y = ell.body_to_array()
    time_step = 0.001
    
    q0 = Quaternion(ell.q)
    print(q0)
    ell_shape = ellipsoid(pos=vector(x[0], x[1], x[2]),
                          length=semi_axes.get('a'), height=semi_axes.get('b'), width=semi_axes.get('c'))
    
    initial_pos = vector(0, 0, 0)
    arrow(pos=initial_pos, axis=vector(3, 0, 0), shaftwidth=0.01, color=vector(255, 0, 0))
    arrow(pos=initial_pos, axis=vector(0, 3, 0), shaftwidth=0.01, color=vector(0, 255, 0))
    arrow(pos=initial_pos, axis=vector(0, 0, 3), shaftwidth=0.01, color=vector(0, 0, 255))
    
    x_v_b = arrow(pos=initial_pos, axis=vector(3, 0, 0), shaftwidth=0.01, color=vector(255, 0, 0))
    y_v_b = arrow(pos=initial_pos, axis=vector(0, 3, 0), shaftwidth=0.01, color=vector(0, 255, 0))
    z_v_b = arrow(pos=initial_pos, axis=vector(0, 0, 3), shaftwidth=0.01, color=vector(0, 0, 255))
    
    while True:
        y = ell.update_position(y, time_step)
        
        translate(ell_shape, ell)
        translate(x_v_b, ell)
        translate(y_v_b, ell)
        translate(z_v_b, ell)
        
        rotate(ell_shape, ell)
        rotate(x_v_b, ell)
        rotate(y_v_b, ell)
        rotate(z_v_b, ell)


main()
