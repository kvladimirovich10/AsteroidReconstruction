import util.utilMethods as utils
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as lg
from pyquaternion import Quaternion
from vpython import *
import vpython as vp


class Ellipsoid:
    
    def __init__(self, semi_axes: dict, x, mass, euler_angles: dict, P, L, force, torque):
        self.a = semi_axes.get('a')
        self.b = semi_axes.get('b')
        self.c = semi_axes.get('c')
        
        self.mass = mass
        
        self.I_body = utils.generate_inertia_body_matrix(self.a, self.b, self.c, self.mass)
        self.I_body_inv = np.linalg.inv(self.I_body)
        
        self.x = x
        self.q = utils.generate_quaternion_by_angels(euler_angles.get('alpha'),
                                                     euler_angles.get('beta'),
                                                     euler_angles.get('gamma'))
        self.P = P
        self.L = L
        
        self.rotation_matrix = utils.quaternion_to_matrix(self.q)
        
        self.I_inv = None
        self.v = None
        self.omega = np.array([euler_angles.get('alpha'),
                               euler_angles.get('beta'),
                               euler_angles.get('gamma')])
        
        self.force = force
        self.torque = torque
    
    def body_to_array(self):
        return np.concatenate((self.x, self.q, self.P, self.L), axis=None)
    
    def get_normal_vector_in_point(self, point):
        normal = np.array([2 * (point[0]) / pow(self.a, 2),
                           2 * (point[1]) / pow(self.b, 2),
                           2 * (point[2]) / pow(self.c, 2)])
        
        rotated_normal = np.matmul(self.rotation_matrix, normal)
        
        return rotated_normal / np.linalg.norm(rotated_normal)
    
    def get_full_velocity_in_point(self, point):
        v_linear = self.v
        omega = np.array(self.omega)
        r = point - self.x
        
        v_angular = np.zeros(3)
        
        if np.count_nonzero(omega) > 0:
            r_proj_on_omega = omega * np.dot(omega, r) / pow(lg.norm(omega), 2)
            r_perp = r - r_proj_on_omega
            v_angular = np.cross(omega, r_perp)
        
        return v_angular + v_linear
    
    def array_to_body(self, y):
        self.x = np.array(y[0:3])
        self.q = np.array(y[3:7])
        self.P = np.array(y[7:10])
        self.L = np.array(y[10:13])
        
        self.rotation_matrix = utils.quaternion_to_matrix(self.q)
        
        self.v = np.true_divide(self.P, self.mass)
        self.I_inv = np.matmul(np.matmul(self.rotation_matrix, self.I_body_inv), np.transpose(self.rotation_matrix))
        self.omega = np.matmul(self.I_inv, self.L)
    
    def update_position(self, y, time_step):
        utils.solver(self, y, time_step)
        return self.body_to_array()
    
    def get_dy_dt_array(self, y):
        self.array_to_body(y)
        
        omega_as_quat = Quaternion(scalar=0, vector=self.omega)
        dq_dt = 0.5 * omega_as_quat * self.q
        dq_dt_v = np.insert(dq_dt.imaginary, 0, dq_dt.scalar)
        
        return np.concatenate((self.v, dq_dt_v, self.force, self.torque), axis=None)
    
    def motion_visualisation(self, time_step, frame_rate):
        s = canvas(title='Rigid ellipsoid free motion',
                   width=1920, height=1080,
                   background=color.gray(0.2))
        
        y = self.body_to_array()
        
        ell_shape = ellipsoid(pos=utils
                              .vector_from_array(self.x),
                              axis=vector(1, 0, 0),
                              length=2 * self.a, height=2 * self.b, width=2 * self.c,
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
        
        r = arrow(pos=initial_pos, axis=vector(0, self.b / 2, 0), shaftwidth=0.01,
                  color=vector(255, 255, 255))
        
        r_p = arrow(pos=ell_shape.pos, axis=vector(0, self.b / 2, 0), shaftwidth=0.01,
                    color=vector(255, 255, 255))
        
        v_c = arrow(pos=ell_shape.pos, axis=vector(0, self.b / 2, 0), shaftwidth=0.01,
                    color=vector(255, 255, 255))
        
        # utils.rotate(ell_shape, self)
        utils.rotate(x_v_b, self)
        utils.rotate(y_v_b, self)
        utils.rotate(z_v_b, self)
        
        const = 5
        while True:
            rate(frame_rate)
            y = self.update_position(y, time_step)
            # utils.translate(ell_shape, self)
            utils.translate(x_v_b, self)
            utils.translate(y_v_b, self)
            utils.translate(z_v_b, self)
            utils.translate(v_v, self)
            utils.translate(omega_v, self)
            utils.translate(r, self)
            
            v_v.axis = const * utils.vector_from_array(self.v).norm()
            omega_v.axis = const * utils.vector_from_array(self.omega).norm()
            
            utils.rotate(ell_shape, self)
            utils.rotate(x_v_b, self)
            utils.rotate(y_v_b, self)
            utils.rotate(z_v_b, self)
            utils.rotate(r, self)
            
            # для отрисовки вектора общей скорости точки на поверхности
            r_projection = (np.dot(utils.array_from_vector(omega_v.axis), utils.array_from_vector(r.axis)) /
                            pow(np.linalg.norm(utils.array_from_vector(omega_v.axis)), 2)) * omega_v.axis
            
            r_p_axis = r.axis - r_projection
            
            r_p.pos = r_projection + ell_shape.pos
            r_p.axis = r_p_axis
            
            v_c.axis = utils.vector_from_array(np.cross(self.omega, utils
                                                        .array_from_vector(r_p.axis)) + self.v).norm()
            v_c.pos = ell_shape.pos + r.axis
            
            s.center = utils.vector_from_array(ell_shape.pos.value)
    
    def ray_marching_visualisation(self):
        canvas(title='Rigid ellipsoid free motion',
               width=1920, height=1080,
               background=color.gray(0.2))
        
        initial_pos = np.array([0, 0, 0])
        v_initial_pos = utils.vector_from_array(initial_pos)
        
        ellipsoid(pos=utils
                  .vector_from_array(self.x),
                  axis=vector(1, 0, 0),
                  length=2 * self.a, height=2 * self.b, width=2 * self.c,
                  texture=vp.textures.rough)
        
        arrow(pos=v_initial_pos, axis=vector(3, 0, 0), shaftwidth=0.01, color=vector(255, 0, 0))
        arrow(pos=v_initial_pos, axis=vector(0, 3, 0), shaftwidth=0.01, color=vector(0, 255, 0))
        arrow(pos=v_initial_pos, axis=vector(0, 0, 3), shaftwidth=0.01, color=vector(0, 0, 255))
        
        points(pos=[v_initial_pos], radius=5, color=color.white)
        
        arrow(pos=v_initial_pos, axis=utils
              .vector_from_array(self.x).norm(), shaftwidth=0.01,
              color=vector(255, 255, 255))
        
        rotated_point = self.get_point_in_ell_system(initial_pos)
        print(rotated_point)
        
        dist = utils.get_dist(self, rotated_point)
        print(dist)
    
    def get_point_in_ell_system(self, point):
        return np.matmul(self.rotation_matrix, point)
