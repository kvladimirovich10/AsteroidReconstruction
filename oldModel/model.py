import oldModel.utils as utils
import matplotlib.pyplot as plt
import numpy as np
from pyquaternion import Quaternion


class Ellipsoid:
    
    def __init__(self, semi_axes, x, mass, euler_angles, P, L, force=None, torque=None):
        self.a = semi_axes.get('a')
        self.b = semi_axes.get('b')
        self.c = semi_axes.get('c')
        
        self.mass = mass

    
        self.x = x
        
        self.rot_matrix = utils.rot_matrix_by_angles(euler_angles.get('alpha'),
                                                     euler_angles.get('beta'),
                                                     euler_angles.get('gamma'))
        
        self.rot_matrix_flatten = utils.matrix_to_array(self.rot_matrix)

        I_body_orig = utils.generate_inertia_body_matrix(self.a, self.b, self.c, self.mass)
        self.I_body = np.matmul(np.matmul(self.rot_matrix, I_body_orig), np.transpose(self.rot_matrix))
        self.I_body_inv = np.linalg.inv(self.I_body)


        self.P = P
        self.L = L
        
        self.I_inv = None
        self.v = None
        self.omega = None
        
        self.force = force
        self.torque = torque
    
    def body_to_array(self):
        return np.concatenate((self.x, self.rot_matrix_flatten, self.P, self.L), axis=None)
    
    def array_to_body(self, y):
        self.x = np.array(y[0:3])
        self.rot_matrix_flatten = np.array(y[3:12])
        self.P = np.array(y[12:15])
        self.L = np.array(y[15:18])
        
        self.rot_matrix = np.reshape(self.rot_matrix_flatten, (-1, 3))

        self.v = np.true_divide(self.P, self.mass)
        self.I_inv = np.matmul(np.matmul(self.rot_matrix, self.I_body_inv), np.transpose(self.rot_matrix))
        self.omega = np.matmul(self.I_inv, self.L)
    
    def update_position(self, y0_state, time_step):
        np.set_printoptions(precision=3)
        y_state = utils.solver(self, y0_state, time_step)
        print(f'new y = {y_state}')
        self.array_to_body(y_state)
        return self.body_to_array()
    
    def dy_dt_to_array(self, y_state):
        self.array_to_body(y_state)
    
        rot_dt = np.matmul(utils.star_omega(self.omega), self.rot_matrix)
        rot_dt_flatten = utils.matrix_to_array(rot_dt)
        res = np.concatenate((self.v, rot_dt_flatten, self.force, self.torque), axis=None)
        return res
    
    def draw(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        
        print(self.b)
        x = self.a * np.outer(np.cos(u), np.sin(v))
        y = self.b * np.outer(np.sin(u), np.sin(v))
        z = self.c * np.outer(np.ones_like(u), np.cos(v))
        
        ax.plot_surface(x, y, z, rstride=10, cstride=10, color='r')
        max_radius = max(self.a, self.b, self.c)
        for axis in 'xyz':
            getattr(ax, 'set_{}lim'.format(axis))((-max_radius, max_radius))
        
        plt.show()
