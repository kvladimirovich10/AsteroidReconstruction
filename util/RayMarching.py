import numpy as np
from model.ellipsoid import Ellipsoid
from model.ray import Ray
from model.radioImage import RadioImage
from util import measurer, utilMethods
import plotly.graph_objects as pgo


def generate_grid(n, max_semi_axis, x_center_distance):
    grid = np.zeros((n * n, 3))
    
    semi_side = max_semi_axis * (x_center_distance + max_semi_axis) / np.sqrt(
        x_center_distance ** 2 - max_semi_axis ** 2)
    step = 2 * semi_side / (n - 1)
    c = 0
    for i in range(n):
        for j in range(n):
            # if pow(-side / 2 + j * step, 2) + pow(-side / 2 + i * step, 2) < pow(side / 2, 2):
            grid[c][0] = -semi_side + j * step
            grid[c][1] = -semi_side + i * step
            grid[c][2] = 0
            c += 1
    
    return np.transpose(grid[:c])


def init_ell(x_init, semi_axes, euler_angles, P=None, L=None):
    mass = 100
    
    force = np.array([0, 0, 0])
    torque = np.array([0, 0, 0])
    
    return Ellipsoid(semi_axes, x_init, mass, euler_angles, P, L, force, torque)


def ellipsoid_ray_marching(ell, observation_point, grid_side_point_number):
    np.set_printoptions(precision=3, suppress=False)
    
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d', proj_type='ortho')
    
    fig = pgo.Figure()
    
    radio_image = RadioImage()
    
    vector = ell.x - observation_point
    normalized_vector = vector / np.linalg.norm(vector)
    print(normalized_vector)
    
    initial_grid = generate_grid(grid_side_point_number, max([ell.a, ell.b, ell.c]), np.linalg.norm(vector))
    initial_grid_normal = [0, 0, 1]
    
    R = utilMethods.R_from_2vec(initial_grid_normal, normalized_vector)
    grid = np.matmul(R, initial_grid)
    
    max_dist = np.linalg.norm(ell.x - observation_point) + max([ell.a, ell.b, ell.c])
    print(max_dist)
    
    eps = 0.001
    k = 0
    
    scan_grid = []
    coef = (np.linalg.norm(ell.x) + max([ell.a, ell.b, ell.c])) / np.linalg.norm(ell.x)
    grid = np.transpose(grid) + coef * ell.x - observation_point
    for grid_point in grid:
        
        if k % 100 == 0:
            print(f'{k}/{len(grid)}')
        
        point_init = grid_point  # + observation_point
        
        dist = np.linalg.norm(ell.x)
        point = observation_point
        closest_point = [0, 0, 0]
        
        closest_point_projection = None
        
        marching_vector = grid_point - observation_point
        normalized_marching_vector = marching_vector / np.linalg.norm(marching_vector)
        while eps < dist < max_dist:
            dist, nonrotated_point_projection = measurer.get_closest_dist_rotated(ell, point)
            closest_point = point + 0.8 * dist * normalized_marching_vector
            point = closest_point
            closest_point_projection = nonrotated_point_projection
        
        if dist < max_dist:
            scan_grid.append(closest_point)
            
            normal = None  # ell.get_normal_vector_in_point(closest_point_projection)
            velocity = ell.get_full_velocity_in_point(closest_point_projection)
            ray = Ray(normal, closest_point, velocity)
            
            radio_image.rays.append(ray)
        else:
            scan_grid.append(point_init)
            
            ray = Ray(distance_to_point=point_init)
            radio_image.rays.append(ray)
        
        k += 1
    
    ray_points = []
    
    for k in range(len(scan_grid) - grid_side_point_number):
        n = grid_side_point_number
        i = k // n
        j = k % n
        
        a = i * n + j
        b = i * n + j + 1
        c = (i + 1) * n + j
        d = (i + 1) * n + j + 1
        
        comparison = (scan_grid[a] != grid[a]).any() \
                     and (scan_grid[b] != grid[b]).any() \
                     and (scan_grid[c] != grid[c]).any() \
                     and (scan_grid[d] != grid[d]).any()
        if comparison:
            mid_point_in_element = utilMethods.centroid([scan_grid[a], scan_grid[b], scan_grid[c], scan_grid[d]])
            ray_points.append(mid_point_in_element)
    
    # for point in scan_grid:
    #     fig.add_scatter3d(x=[point[0]],
    #                       y=[point[1]],
    #                       z=[point[2]],
    #                       marker=dict(
    #                           size=2,
    #                           color='rgb(0, 0, 0)'
    #                       ))
    
    for point in ray_points:
        fig.add_scatter3d(x=[point[0]],
                          y=[point[1]],
                          z=[point[2]],
                          marker=dict(
                              size=2,
                              color='rgb(255, 0, 0)'
                          ))
        # for ray in radio_image.rays:
        # if ray.velocity_in_point is not None:
        #     fig.add_trace(utilMethods.get_arrow_cone(ray.distance_to_point,
        #                                              ray.velocity_in_point))
        #
        # if ray.normal_in_point is not None:
        #     fig.add_trace(utilMethods.get_arrow_cone(ray.distance_to_point,
        #                                              ray.normal_in_point))
        
        # fig.add_scatter3d(x=[ray.distance_to_point[0]],
        #                   y=[ray.distance_to_point[1]],
        #                   z=[ray.distance_to_point[2]])
    
    # fig.add_scatter3d(x=[observation_point[0]],
    #                   y=[observation_point[1]],
    #                   z=[observation_point[2]],
    #                   marker=dict(
    #                       size=2,
    #                       color='rgb(255, 0, 0)'
    #                   )
    #                   )
    
    fig.show()
    fig.write_html('ellipsoid_ray_marching.html', auto_open=True)


def ray_marching_test():
    x_init = np.array([-10000, 0, 0])
    semi_axes = {'a': 4, 'b': 2, 'c': 2}
    euler_angles = {'alpha': -40, 'beta': -45, 'gamma': 80}
    
    ellipsoid = init_ell(x_init, semi_axes, euler_angles)
    ellipsoid.v = [0, 0, 0]
    ellipsoid.omega = [0, 0, 0]
    
    observation_point = [0, 0, 0]
    grid_side_point_number = 60
    
    ellipsoid_ray_marching(ellipsoid, observation_point, grid_side_point_number)


if __name__ == '__main__':
    ray_marching_test()
