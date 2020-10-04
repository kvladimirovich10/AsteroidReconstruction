from __future__ import division, print_function

import numpy as np
from model import Ellipsoid
from vpython import *
import vpython as vp


def vector_from_array(array):
    return vector(array[0], array[1], array[2])


def array_from_vector(vector):
    return np.array(vector.value)


def rotate(body, ell):
    body.rotate(angle=ell.omega[0], axis=vector(1, 0, 0))
    body.rotate(angle=ell.omega[1], axis=vector(0, 1, 0))
    body.rotate(angle=ell.omega[2], axis=vector(0, 0, 1))


def translate(body, ell):
    body.pos = body.pos + vector(ell.v[0], ell.v[1], ell.v[2])


def main():
    s = canvas(title='Rigid ellipsoid free motion',
               width=1920, height=1080,
               background=color.gray(0.2))
    
    mass = 1000
    x = np.array([0, 0, 0])
    semi_axes = {'a': 2.5, 'b': 1.5, 'c': 1.3}
    euler_angles = {'alpha': 0, 'beta': 0, 'gamma': 0}
    
    P = np.array([1, 1, 1])
    L = np.array([2.5, -3, 1])
    
    force = np.array([0, 0, 0])
    torque = np.array([0, 0, 0])
    
    ell = Ellipsoid(semi_axes, x, mass, euler_angles, P, L, force, torque)
    
    y = ell.body_to_array()
    time_step = 0.001
    
    ell_shape = ellipsoid(pos=vector_from_array(x),
                          axis=vector(1, 0, 0),
                          length=semi_axes.get('a'), height=semi_axes.get('b'), width=semi_axes.get('c'),
                          texture=vp.textures.rough)
        
    initial_pos = vector(0, 0, 0)
    
    arrow(pos=initial_pos, axis=vector(3, 0, 0), shaftwidth=0.01, color=vector(255, 0, 0))
    arrow(pos=initial_pos, axis=vector(0, 3, 0), shaftwidth=0.01, color=vector(0, 255, 0))
    arrow(pos=initial_pos, axis=vector(0, 0, 3), shaftwidth=0.01, color=vector(0, 0, 255))
    
    x_v_b = arrow(pos=initial_pos, axis=vector(3, 0, 0), shaftwidth=0.01, color=vector(255, 0, 0))
    y_v_b = arrow(pos=initial_pos, axis=vector(0, 3, 0), shaftwidth=0.01, color=vector(0, 255, 0))
    z_v_b = arrow(pos=initial_pos, axis=vector(0, 0, 3), shaftwidth=0.01, color=vector(0, 0, 255))
    
    v_v = arrow(pos=initial_pos, axis=initial_pos, shaftwidth=0.05, color=vector(255, 220, 0))
    omega_v = arrow(pos=initial_pos, axis=initial_pos, shaftwidth=0.05, color=vector(255, 0, 255))
    
    r = arrow(pos=initial_pos, axis=vector(0, semi_axes.get('b') / 2, 0), shaftwidth=0.01,
              color=vector(255, 255, 255))
    
    r_p = arrow(pos=ell_shape.pos, axis=vector(0, semi_axes.get('b') / 2, 0), shaftwidth=0.01,
                color=vector(255, 255, 255))
    
    v_c = arrow(pos=ell_shape.pos, axis=vector(0, semi_axes.get('b') / 2, 0), shaftwidth=0.01,
                color=vector(255, 255, 255))
    
    rotate(ell_shape, ell)
    rotate(x_v_b, ell)
    rotate(y_v_b, ell)
    rotate(z_v_b, ell)
    
    const = 5
    while True:
        y = ell.update_position(y, time_step)
        
        translate(ell_shape, ell)
        translate(x_v_b, ell)
        translate(y_v_b, ell)
        translate(z_v_b, ell)
        translate(v_v, ell)
        translate(omega_v, ell)
        translate(r, ell)
        
        v_v.axis = const * vector_from_array(ell.v).norm()
        omega_v.axis = const * vector_from_array(ell.omega).norm()
        
        rotate(ell_shape, ell)
        rotate(x_v_b, ell)
        rotate(y_v_b, ell)
        rotate(z_v_b, ell)
        rotate(r, ell)
        
        # для отрисовки вектора обшей скорости точки на поверхности
        r_projection = (np.dot(array_from_vector(omega_v.axis), array_from_vector(r.axis)) / pow(
            np.linalg.norm(array_from_vector(omega_v.axis)), 2)) * omega_v.axis
        
        r_p_axis = r.axis - r_projection
        
        r_p.pos = r_projection + ell_shape.pos
        r_p.axis = r_p_axis
        
        v_c.axis = vector_from_array(np.cross(ell.omega, array_from_vector(r_p.axis)) + ell.v).norm()
        v_c.pos = ell_shape.pos + r.axis
        
        s.center = vector_from_array(ell_shape.pos.value)


main()
