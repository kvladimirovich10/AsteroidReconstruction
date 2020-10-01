from __future__ import division, print_function

import numpy as np
from model import Ellipsoid
from vpython import *


def rotate(body, ell):
    body.rotate(angle=ell.omega[0], axis=vector(1, 0, 0))
    body.rotate(angle=ell.omega[1], axis=vector(0, 1, 0))
    body.rotate(angle=ell.omega[2], axis=vector(0, 0, 1))


def translate(body, ell):
    body.pos = body.pos + vector(ell.v[0], ell.v[1], ell.v[2])


def main():
    s = canvas(title='Rigid ellipsoid free motion',
               width=1920, height=1080,
               background=color.gray(0.3))
    
    mass = 1000
    x = np.array([0, 0, 0])
    semi_axes = {'a': 2.5, 'b': 1.5, 'c': 1.3}
    euler_angles = {'alpha': np.pi/3, 'beta': np.pi/7, 'gamma': 0}
    
    P = np.array([0, 0, 0])
    L = np.array([4, -2, 1.5])
    
    force = np.array([0, 0, 0])
    torque = np.array([0, 0, 0])
    
    ell = Ellipsoid(semi_axes, x, mass, euler_angles, P, L, force, torque)
    
    y = ell.body_to_array()
    time_step = 0.001
    
    ell_shape = ellipsoid(pos=vector(x[0], x[1], x[2]),
                          axis=vector(1, 0, 0),
                          length=semi_axes.get('a'), height=semi_axes.get('b'), width=semi_axes.get('c'))
    
    initial_pos = vector(0, 0, 0)
    
    arrow(pos=initial_pos, axis=vector(3, 0, 0), shaftwidth=0.01, color=vector(255, 0, 0))
    arrow(pos=initial_pos, axis=vector(0, 3, 0), shaftwidth=0.01, color=vector(0, 255, 0))
    arrow(pos=initial_pos, axis=vector(0, 0, 3), shaftwidth=0.01, color=vector(0, 0, 255))
    
    x_v_b = arrow(pos=initial_pos, axis=vector(3, 0, 0), shaftwidth=0.01, color=vector(255, 0, 0))
    y_v_b = arrow(pos=initial_pos, axis=vector(0, 3, 0), shaftwidth=0.01, color=vector(0, 255, 0))
    z_v_b = arrow(pos=initial_pos, axis=vector(0, 0, 3), shaftwidth=0.01, color=vector(0, 0, 255))
    
    v_v = arrow(pos=initial_pos, axis=initial_pos, shaftwidth=0.05, color=vector(255, 255, 0))
    omega_v = arrow(pos=initial_pos, axis=initial_pos, shaftwidth=0.05, color=vector(255, 0, 255))
    
    rotate(ell_shape, ell)
    rotate(x_v_b, ell)
    rotate(y_v_b, ell)
    rotate(z_v_b, ell)
    
    while True:
        y = ell.update_position(y, time_step)
        
        translate(ell_shape, ell)
        translate(x_v_b, ell)
        translate(y_v_b, ell)
        translate(z_v_b, ell)
        translate(v_v, ell)
        translate(omega_v, ell)
        
        v_v.axis = 1000 * vector(ell.v[0], ell.v[1], ell.v[2])
        omega_v.axis = 1000 * vector(ell.omega[0], ell.omega[1], ell.omega[2])
        
        rotate(ell_shape, ell)
        rotate(x_v_b, ell)
        rotate(y_v_b, ell)
        rotate(z_v_b, ell)
        
        s.center = vector(ell_shape.pos.value[0], ell_shape.pos.value[1], ell_shape.pos.value[2])


main()
