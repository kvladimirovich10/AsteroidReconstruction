import utils
import matplotlib.pyplot as plt
import numpy as np


class Ellipsoid:
    
    def __init__(self, a, b, c, x, mass=None, alpha=None, beta=None, gamma=None, P=None, L=None):
        self.a = a
        self.b = b
        self.c = c
        self.mass = mass
        self.I_body, self.I_body_inv = utils.generate_inertia_body_matrices(a, b, c, mass)
        
        self.x = x
        self.q = utils.generate_quaternion_by_angels(alpha, beta, gamma)
        self.P = P
        self.L = L
        
        self.I_inv = None
        self.v = None
        self.omega = None
        
        self.force = None
        self.torque = None
    
    def body_to_array(self):
        y = self.x
        y.extend(self.q)
        y.extend(self.P)
        y.extend(self.L)
        
        return y
    
    def array_to_body(self, y):
        self.x = np.array(y[0:3])
        self.q = np.array(y[3:7])
        self.P = np.array(y[7:10])
        self.L = np.array(y[10:13])
        
        R = utils.quaternion_to_matrix(self.q)
        
        self.v = np.true_divide(self.P, self.mass)
        self.I_inv = np.matmul(np.matmul(R, self.I_body_inv), np.transpose(R))
        self.omega = np.matmul(self.I_inv, self.L)
        
        self.force = np.array([0, 0, 0])
        self.torque = np.array([0, 0, 0])
    
    def get_ddt_array(self, y):
        dydt = []
    
    def update_position(self, y):
        self.array_to_body(y)
        #     smth
        self.body_to_array()
        
    def ode(self, y):
        pass
    
    def draw(self):
        fig = plt.figure(figsize=plt.figaspect(1))
        ax = fig.add_subplot(111, projection='3d')
        
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        
        x = self.a * np.cos(u) * np.sin(v)
        y = self.b * np.sin(u) * np.sin(v)
        
        z = self.c * np.ones_like(u) * np.cos(v)
        
        R = utils.quaternion_to_matrix(self.q)
        
        XYZ = np.transpose(np.array([x, y, z]))
        x, y, z = np.transpose(np.dot(XYZ, R))
        ax.plot_surface(x, y, z, rstride=4, cstride=4, color='b')
        #
        # max_radius = max(self.a, self.b, self.c)
        # for axis in 'xyz':
        #     getattr(ax, 'set_{}lim'.format(axis))((-max_radius, max_radius))
        #
        plt.show()
