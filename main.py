import numpy as np

from model import Ellipsoid

a = 1
b = 2
c = 3
mass = 10
x = np.array([0, 0, 0])

alpha = np.pi / 10
beta = np.pi / 10
gamma = np.pi / 10

ellipsoid = Ellipsoid(a, b, c, x)
ellipsoid.draw()
