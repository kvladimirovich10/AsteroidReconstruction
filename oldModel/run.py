from __future__ import division, print_function

import numpy as np
from oldModel.model import Ellipsoid
from vpython import *
import time
import math


def rotate(body, ell):
    body.rotate(angle=ell.omega[0], axis=vector(1, 0, 0))
    body.rotate(angle=ell.omega[1], axis=vector(0, 1, 0))
    body.rotate(angle=ell.omega[2], axis=vector(0, 0, 1))


def initial_rotation(body, ell: Ellipsoid):
    body.rotate(angle=math.radians(ell.euler_angles.get('alpha')), axis=vector(1, 0, 0))
    body.rotate(angle=math.radians(ell.euler_angles.get('beta')), axis=vector(0, 1, 0))
    body.rotate(angle=math.radians(ell.euler_angles.get('gamma')), axis=vector(0, 0, 1))


def translate(body, ell):
    body.pos = body.pos + vector(ell.v[0], ell.v[1], ell.v[2])


def main():
    np.set_printoptions(precision=3)
    s = canvas(title='Rigid ellipsoid free motion',
               width=500, height=500,
               background=color.white)
    
    mass = 1
    x = np.array([0, 0, 0])
    a = 5
    semi_axes = {'a': a, 'b': 2, 'c': 1}
    euler_angles = {'alpha': 0, 'beta': 0, 'gamma': 0}
    
    P = np.array([0, 0, 0])
    L = np.array([10, 10, 10])
    
    force = np.array([0, 0, 0])
    torque = np.array([0, 0, 0])
    
    ell = Ellipsoid(semi_axes, x, mass, euler_angles, P, L, force, torque)
    
    y = ell.body_to_array()
    time_step = 0.01
    
    initial_pos = vector(0, 0, 0)
    
    ell_shape = ellipsoid(pos=vector(x[0], x[1], x[2]),
                          length=ell.a, width=ell.c, height=ell.b)
    
    v = [a / 2, 0, 0]
    trail_vector = arrow(pos=initial_pos, axis=vector_from_array(v), shaftwidth=0.01, color=vector(255, 255, 255))
    point_to_trail = sphere(pos=vector_from_array(v), radius=0.01, color=color.red)
    
    attach_trail(point_to_trail, color=color.black, radius=0.01)
    
    arrow(pos=initial_pos, axis=vector(3, 0, 0), shaftwidth=0.01, color=vector(255, 0, 0))
    arrow(pos=initial_pos, axis=vector(0, 3, 0), shaftwidth=0.01, color=vector(0, 255, 0))
    arrow(pos=initial_pos, axis=vector(0, 0, 3), shaftwidth=0.01, color=vector(0, 0, 255))
    
    omega_v = arrow(pos=initial_pos, axis=initial_pos, shaftwidth=0.05, color=vector(255, 0, 255))
    
    l_v = arrow(pos=initial_pos, axis=initial_pos, shaftwidth=0.05, color=vector(255, 255, 0))
    
    x_v_b = arrow(pos=initial_pos, axis=vector(3, 0, 0), shaftwidth=0.01, color=vector(255, 0, 0))
    y_v_b = arrow(pos=initial_pos, axis=vector(0, 3, 0), shaftwidth=0.01, color=vector(0, 255, 0))
    z_v_b = arrow(pos=initial_pos, axis=vector(0, 0, 3), shaftwidth=0.01, color=vector(0, 0, 255))
    
    initial_rotation(ell_shape, ell)
    initial_rotation(trail_vector, ell)
    initial_rotation(x_v_b, ell)
    initial_rotation(y_v_b, ell)
    initial_rotation(z_v_b, ell)
    
    while True:
        point_to_trail.pos = trail_vector.axis + trail_vector.pos
        y = ell.update_position(y, time_step)
        
        translate(ell_shape, ell)
        translate(omega_v, ell)
        translate(l_v, ell)
        translate(x_v_b, ell)
        translate(y_v_b, ell)
        translate(z_v_b, ell)
        translate(trail_vector, ell)
        
        omega_v.axis = 3 * vector_from_array(ell.omega).norm()
        l_v.axis = 3 * vector_from_array(ell.L).norm()
        
        rotate(ell_shape, ell)
        rotate(trail_vector, ell)
        rotate(x_v_b, ell)
        rotate(y_v_b, ell)
        rotate(z_v_b, ell)
        time.sleep(time_step)


def vector_from_array(array):
    return vector(array[0], array[1], array[2])


main()
