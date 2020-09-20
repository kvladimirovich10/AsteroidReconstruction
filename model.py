import utils
import matplotlib.pyplot as plt
import numpy as np


class Ellipsoid:
    
    def __init__(self, a, b, c, x, mass=None, q=None, p=None, l=None):
        self.a = a
        self.b = b
        self.c = c
        self.mass = mass
        self.i_body, self.i_body_inv = utils.generate_inertia_body_matrices(a, b, c, mass)
        
        self.x = x
        self.q = q
        self.p = p
        self.l = l
        
        self.i_inv = None
        self.v = None
        self.omega = None
        self.force = None
        self.torque = None
    
    def draw(self):
        fig = plt.figure(figsize=plt.figaspect(1))
        ax = fig.add_subplot(111, projection='3d')
        
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        
        x = self.a * np.outer(np.cos(u), np.sin(v))
        y = self.b * np.outer(np.sin(u), np.sin(v))
        z = self.c * np.outer(np.ones_like(u), np.cos(v))
        
        ax.plot_surface(x, y, z, rstride=4, cstride=4, color='b')
        
        max_radius = max(self.a, self.b, self.c)
        for axis in 'xyz':
            getattr(ax, 'set_{}lim'.format(axis))((-max_radius, max_radius))
        
        plt.show()
