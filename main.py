import numpy as np

from model import Ellipsoid

a = 1
b = 2
c = 3
mass = 10
x = np.array([0, 0, 0])

alpha = np.pi / 10
beta = np.pi / 2
gamma = np.pi / 10

P = np.array([0, 0, 0])
L = np.array([0, 0, 0])

ellipsoid = Ellipsoid(a, b, c, x, mass, alpha, beta, gamma, P, L)
ellipsoid.draw()
# y = ellipsoid.body_to_array()
#
# while 1:
#     y = ellipsoid.update_position(y)




