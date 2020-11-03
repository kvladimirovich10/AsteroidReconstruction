from __future__ import division, print_function
import numpy as np
from model import Ellipsoid
from math import *
from numpy.linalg import *
import matplotlib.pyplot as plt
from util import measurer, utils


def init_ell():
    mass = 100
    x = np.array([1, 0, 0])
    semi_axes = {'a': 2, 'b': 2.5, 'c': 2.3}
    euler_angles = {'alpha': pi/3, 'beta': 0, 'gamma': 0}
    
    P = np.array([1, 1, 1])
    L = np.array([2.5, -3, 1])
    
    force = np.array([0, 0, 0])
    torque = np.array([0, 0, 0])
    
    return Ellipsoid(semi_axes, x, mass, euler_angles, P, L, force, torque)


def generate_grid(n, side):
    grid = np.zeros((n * n, 3))
    
    step = side / (n - 1)
    
    for i in range(n):
        for j in range(n):
            grid[j + i * n][0] = -side / 2 + j * step
            grid[j + i * n][1] = -side / 2 + i * step
            grid[j + i * n][2] = 0
    
    return grid


ell = init_ell()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d', proj_type='ortho')

n = 30

vector = [-3, 0, 0]
shift = [5, 0, 0]
vector = vector / np.linalg.norm(vector)

R = utils.R_from_2vec([0, 0, 1], vector)
initial_grid = generate_grid(n, 2)
grid = np.matmul(initial_grid, R)

for i in range(len(grid)):
    point = grid[i] + shift

    eps = 0.0001

    max_dist = np.linalg.norm(ell.x - point) + max([ell.a, ell.b, ell.c])
    closest_point = ell.x

    dist = np.linalg.norm(ell.x - point)
    
    while eps < dist < max_dist:
        dist, point_projection = measurer.get_closest_dist(ell, point)
        closest_point = point + dist * vector
        point = closest_point
    print(dist)
    ax.scatter(closest_point[0], closest_point[1], closest_point[2], s=2, marker='o')
    
ax.view_init(elev=45, azim=45)
ax.axis('auto')
plt.show()
