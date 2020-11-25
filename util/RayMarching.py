import numpy as np
from model.ellipsoid import Ellipsoid
from model.ray import Ray
from model.radioImage import RadioImage
import matplotlib.pyplot as plt
from util import measurer, utilMethods
import plotly.graph_objects as pgo


def generate_grid(n, side):
    grid = np.zeros((n * n, 3))
    
    step = side / (n - 1)
    c = 0
    for i in range(n):
        for j in range(n):
            if pow(-side / 2 + j * step, 2) + pow(-side / 2 + i * step, 2) < pow(side / 2, 2):
                grid[c][0] = -side / 2 + j * step
                grid[c][1] = -side / 2 + i * step
                grid[c][2] = 0
                c += 1
    
    return np.transpose(grid[:c])


def init_ell(x_init):
    mass = 100
    x = np.array(x_init)
    semi_axes = {'a': 7, 'b': 3, 'c': 5}
    euler_angles = {'alpha': 30, 'beta': 40, 'gamma': 50}
    
    P = np.array([1, 1, 1])
    L = np.array([2.5, -3, 1])
    
    force = np.array([0, 0, 0])
    torque = np.array([0, 0, 0])
    
    return Ellipsoid(semi_axes, x, mass, euler_angles, P, L, force, torque)


def ellipsoid_ray_marching(ell, observation_point, grid_side_point_number):
    np.set_printoptions(precision=3, suppress=False)
    
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d', proj_type='ortho')
    
    fig = pgo.Figure()
    
    radio_image = RadioImage()
    
    vector = ell.x - observation_point
    vector = vector / np.linalg.norm(vector)
    print(vector)
    
    initial_grid = generate_grid(grid_side_point_number, 2 * max([ell.a, ell.b, ell.c]))
    initial_grid_normal = [0, 0, 1]
    
    R = utilMethods.R_from_2vec(initial_grid_normal, vector)
    grid = np.matmul(R, initial_grid)
    
    max_dist = np.linalg.norm(ell.x - observation_point) + max([ell.a, ell.b, ell.c])
    print(max_dist)
    
    eps = 0.001
    k = 0
    for grid_point in np.transpose(grid):
        
        if k % 100 == 0:
            print(f'{k}/{len(np.transpose(grid))}')
        
        point_init = grid_point + observation_point
        
        dist = np.linalg.norm(ell.x - point_init)
        point = point_init
        closest_point = [0, 0, 0]
        
        while eps < dist < max_dist:
            dist, point_projection = measurer.get_closest_dist_rotated(ell, point)
            closest_point = point + 0.9 * dist * vector
            point = closest_point
        
        if dist < max_dist:
            ray = Ray(0, closest_point, [0, 0, 0])
            radio_image.rays.append(ray)
            
            
            fig.add_scatter3d(x=[closest_point[0]],
                              y=[closest_point[1]],
                              z=[closest_point[2]])
            # ax.scatter(closest_point[0], closest_point[1], closest_point[2], s=2, marker='o')
        else:
            fig.add_scatter3d(x=[point_init[0]],
                              y=[point_init[1]],
                              z=[point_init[2]])
            # ax.scatter(point_init[0], point_init[1], point_init[2], s=2, marker='o')
        
        k += 1
    
    fig.add_scatter3d(x=[observation_point[0]],
                      y=[observation_point[1]],
                      z=[observation_point[2]])
    # ax.scatter(observation_point[0],
    #            observation_point[1],
    #            observation_point[2],
    #            s=3, marker='o')
    
    fig.show()
    # plt.show()


def ray_marching_test():
    
    # Helix equation
    t = np.linspace(0, 10, 50)
    x, y, z = np.cos(t), np.sin(t), t
    
    fig = pgo.Figure(data=[pgo.Scatter3d(x=x, y=y, z=z,
                                       mode='markers')])
    fig.show()
    # ellipsoid = init_ell([6, -5, 4])
    # observation_point = [0, 0, 0]
    # grid_side_point_number = 20
    #
    # ellipsoid_ray_marching(ellipsoid, observation_point, grid_side_point_number)


if __name__ == '__main__':
    ray_marching_test()
