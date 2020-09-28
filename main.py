import numpy as np

from model import Ellipsoid

mass = 10
x = np.array([0, 0, 0])
semi_axes = {'a': 1, 'b': 1, 'c': 1}
euler_angles = {'alpha': np.pi / 10, 'beta': np.pi / 2, 'gamma': np.pi / 10}

P = np.array([0, 0, 0])
L = np.array([0, 0, 0])

force = np.array([0, 0, 0])
torque = np.array([0, 0, 0])

ellipsoid = Ellipsoid(semi_axes, x, mass, euler_angles, P, L, force, torque)

y = ellipsoid.body_to_array()
time_step = 0.05

for i in range(100):
    y = ellipsoid.update_position(y, time_step)
