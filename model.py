import utils
import matplotlib.pyplot as plt
import numpy as np
from pyquaternion import Quaternion
from vpython import *
import vpython as vp


class Ellipsoid:
    
    def __init__(self, semi_axes, x, mass, euler_angles, P, L, force, torque):
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
        
        self.I_inv = None
        self.v = None
        self.omega = np.array([euler_angles.get('alpha'),
                               euler_angles.get('beta'),
                               euler_angles.get('gamma')])
        
        self.force = force
        self.torque = torque
    
    def body_to_array(self):
        return np.concatenate((self.x, self.q, self.P, self.L), axis=None)
    
    def array_to_body(self, y):
        self.x = np.array(y[0:3])
        self.q = np.array(y[3:7])
        self.P = np.array(y[7:10])
        self.L = np.array(y[10:13])
        
        rotation_matrix = utils.quaternion_to_matrix(self.q)
        
        self.v = np.true_divide(self.P, self.mass)
        self.I_inv = np.matmul(np.matmul(rotation_matrix, self.I_body_inv), np.transpose(rotation_matrix))
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
    
    def draw(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        
        x = self.a * np.outer(np.cos(u), np.sin(v))
        y = self.b * np.outer(np.sin(u), np.sin(v))
        z = self.c * np.outer(np.ones_like(u), np.cos(v))
        
        ax.plot_surface(x, y, z, rstride=10, cstride=10, color='r')
        max_radius = max(self.a, self.b, self.c)
        for axis in 'xyz':
            getattr(ax, 'set_{}lim'.format(axis))((-max_radius, max_radius))
        
        plt.show()

    def motion_visualisation(self):
        s = canvas(title='Rigid selfipsoid free motion',
                   width=1920, height=1080,
                   background=color.gray(0.2))
    
        y = self.body_to_array()
    
        time_step = 0.001
    
        self_shape = ellipsoid(pos=utils.vector_from_array(self.x),
                              axis=vector(1, 0, 0),
                              length=self.a, height=self.b, width=self.c,
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
    
        r_p = arrow(pos=self_shape.pos, axis=vector(0, self.b / 2, 0), shaftwidth=0.01,
                    color=vector(255, 255, 255))
    
        v_c = arrow(pos=self_shape.pos, axis=vector(0, self.b / 2, 0), shaftwidth=0.01,
                    color=vector(255, 255, 255))
    
        utils.rotate(self_shape, self)
        utils.rotate(x_v_b, self)
        utils.rotate(y_v_b, self)
        utils.rotate(z_v_b, self)
    
        const = 5
        while True:
            y = self.update_position(y, time_step)
        
            utils.translate(self_shape, self)
            utils.translate(x_v_b, self)
            utils.translate(y_v_b, self)
            utils.translate(z_v_b, self)
            utils.translate(v_v, self)
            utils.translate(omega_v, self)
            utils.translate(r, self)
        
            v_v.axis = const * utils.vector_from_array(self.v).norm()
            omega_v.axis = const * utils.vector_from_array(self.omega).norm()
        
            utils.rotate(self_shape, self)
            utils.rotate(x_v_b, self)
            utils.rotate(y_v_b, self)
            utils.rotate(z_v_b, self)
            utils.rotate(r, self)
        
            # для отрисовки вектора общей скорости точки на поверхности
            r_projection = (np.dot(utils.array_from_vector(omega_v.axis), utils.array_from_vector(r.axis)) / pow(
                np.linalg.norm(utils.array_from_vector(omega_v.axis)), 2)) * omega_v.axis
        
            r_p_axis = r.axis - r_projection
        
            r_p.pos = r_projection + self_shape.pos
            r_p.axis = r_p_axis
        
            v_c.axis = utils.vector_from_array(np.cross(self.omega, utils.array_from_vector(r_p.axis)) + self.v).norm()
            v_c.pos = self_shape.pos + r.axis
        
            s.center = utils.vector_from_array(self_shape.pos.value)
