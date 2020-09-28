import utils
import matplotlib.pyplot as plt
import numpy as np
from pyquaternion import Quaternion


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
        self.omega = None
        
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
        
        self.force = np.array([0, 0, 0])
        self.torque = np.array([0, 0, 0])
    
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
        
        print(self.b)
        x = self.a * np.outer(np.cos(u), np.sin(v))
        y = self.b * np.outer(np.sin(u), np.sin(v))
        z = self.c * np.outer(np.ones_like(u), np.cos(v))
        
        ax.plot_surface(x, y, z, rstride=10, cstride=10, color='r')
        max_radius = max(self.a, self.b, self.c)
        for axis in 'xyz':
            getattr(ax, 'set_{}lim'.format(axis))((-max_radius, max_radius))
        
        plt.show()
