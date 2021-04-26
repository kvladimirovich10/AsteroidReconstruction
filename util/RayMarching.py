import sys

import numpy as np
from model.ellipsoid import Ellipsoid
from model.grid import Grid
from model.ray import Ray
from model.radioImage import RadioImage
from util import measurer, utilMethods
from shapely.geometry.polygon import Polygon
import plotly.graph_objects as pgo
from model.ellipsoid import Ellipsoid


def plot_grid(grid):
    x = grid[0]
    y = grid[1]
    z = grid[2]
    
    radio_scatter = pgo.Figure(data=[pgo.Scatter3d(x=x,
                                                   y=y,
                                                   z=z,
                                                   mode='markers',
                                                   marker=dict(size=1))]
                               )
    
    radio_scatter.show()


def generate_ray_grid(ell, grid_side_point_number):
    
    semi_side, step, initial_grid = generate_initial_grid(ell, grid_side_point_number)
    
    center_unit_vector = ell.x / np.linalg.norm(ell.x)
    
    grid_rotation_mat = utilMethods.R_from_2vec(center_unit_vector)
    untranslated_grid = np.matmul(grid_rotation_mat, initial_grid)
    
    translation_coef = (np.linalg.norm(ell.x) + max([ell.a, ell.b, ell.c])) / np.linalg.norm(ell.x)
    
    final_grid = np.transpose(untranslated_grid) + translation_coef * ell.x
    
    return Grid(semi_side, step, grid_side_point_number, final_grid)


def generate_initial_grid(ell, n):
    x_center_distance = np.linalg.norm(ell.x)
    max_semi_axis = max([ell.a, ell.b, ell.c])
    grid = np.zeros((n * n, 3))
    
    semi_side = max_semi_axis * (x_center_distance + max_semi_axis) / x_center_distance
                # np.sqrt(x_center_distance ** 2 - max_semi_axis ** 2)
    
    step = 2 * semi_side / (n - 1)
    
    for i in range(n):
        for j in range(n):
            x = -semi_side + j * step
            y = -semi_side + i * step
            current_index = j + i * n
            grid[current_index][0] = x
            grid[current_index][1] = y
            grid[current_index][2] = 0
    
    return semi_side, step, np.transpose(grid)


def init_ell(x_init, semi_axes, euler_angles, P=None, L=None):
    mass = 100
    
    force = np.array([0, 0, 0])
    torque = np.array([0, 0, 0])
    
    return Ellipsoid(semi_axes, x_init, mass, euler_angles, P, L, force, torque)


def filter_out_of_circle(grid, k):
    n = grid.grid_side_point_number
    i = k // n
    j = k % n
    x = -grid.semi_side + j * grid.step
    y = -grid.semi_side + i * grid.step
    
    if pow(x, 2) + pow(y, 2) > pow(grid.semi_side, 2):
        return True
    
    return False


def ellipsoid_ray_marching(ell, grid_side_point_number):
    grid = generate_ray_grid(ell, grid_side_point_number)
    
    radio_image = RadioImage(ell)
    
    max_dist = np.linalg.norm(ell.x) + max([ell.a, ell.b, ell.c])
    k = 0
    
    for grid_point in grid.points:
        
        if k % 100 == 0:
            sys.stdout.write(f'\r{k}/{len(grid.points)}')
            sys.stdout.flush()
        
        if filter_out_of_circle(grid, k):
            k += 1
            continue

        eps = 0.0001
        
        dist, closest_point = ray_marching(ell, grid_point, max_dist, eps)
        
        if dist < eps:
            _, not_rotated_point_projection = measurer.get_closest_dist_rotated(ell, closest_point)
            
            normal = ell.get_normal_vector_in_point(not_rotated_point_projection)
            velocity = ell.get_full_velocity_in_point(closest_point)
            ray = Ray(normal_in_point=normal, ray_to_point=closest_point, velocity_in_point=velocity)
            
            radio_image.add_ray(ray)
        
        k += 1
    
    # plot_ray_marching(radio_image)
    
    return radio_image


def ray_marching(ell, grid_point, max_dist, eps):
    
    dist = np.linalg.norm(ell.x)
    point = [0, 0, 0]
    closest_point = [0, 0, 0]
    
    marching_vector = grid_point
    normalized_marching_vector = marching_vector / np.linalg.norm(marching_vector)
    
    while eps < dist and np.linalg.norm(point) < max_dist:
        dist, _ = measurer.get_closest_dist_rotated(ell, point)
        closest_point = point + 0.8 * dist * normalized_marching_vector
        point = closest_point
    
    return dist, closest_point


def plot_ray_marching(radio_image):
    fig = pgo.Figure()
    fig.update_layout(
        scene=dict(
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False),
            zaxis=dict(showticklabels=False),
        )
    )
    
    fig.update_layout(showlegend=False)
    
    for ray in radio_image.rays:
        fig.add_scatter3d(x=[ray.ray_to_point[0]],
                          y=[ray.ray_to_point[1]],
                          z=[ray.ray_to_point[2]],
                          marker=dict(
                              size=2,
                              color='rgb(255, 0, 0)'
                          ))
        
        # if ray.normal_in_point is not None:
        #     fig.add_trace(utilMethods.get_arrow_cone(ray.ray_to_point,
        #                                              ray.normal_in_point))
        
        if ray.velocity_in_point is not None:
            fig.add_trace(utilMethods.get_arrow_cone(ray.ray_to_point,
                                                     ray.velocity_projection))
    
    fig.add_scatter3d(x=[0, 0],
                      y=[0, 0],
                      z=[0, 0.1],
                      marker=dict(
                          size=2,
                          color='rgb(0, 0, 255)'
                      ))
    fig.show()
    fig.write_html('ellipsoid_ray_marching.html', auto_open=True)


"""
for k in range(len(scan_grid) - grid_side_point_number):
        n = grid_side_point_number
        i = k // n
        j = k % n

        a = i * n + j
        b = i * n + j + 1
        c = (i + 1) * n + j
        d = (i + 1) * n + j + 1

        comparison = (scan_grid[a] != grid.points[a]).any() \
                     and (scan_grid[b] != grid.points[b]).any() \
                     and (scan_grid[c] != grid.points[c]).any() \
                     and (scan_grid[d] != grid.points[d]).any()
        if comparison:
            element = [scan_grid[a], scan_grid[b], scan_grid[c], scan_grid[d]]
            _x_list = [v[0] for v in element]
            _y_list = [v[1] for v in element]
            _z_list = [v[2] for v in element]

            poligon = Polygon(list(zip(_x_list, _y_list, _z_list)))

            mid_point_in_element = utilMethods.centroid(element)

            _, nonrotated_point_projection = measurer.get_closest_dist_rotated(ell, mid_point_in_element)

            normal = ell.get_normal_vector_in_point(nonrotated_point_projection)
            velocity = ell.get_full_velocity_in_point(mid_point_in_element)
            ray = Ray(normal, mid_point_in_element, velocity, poligon.area)

            radio_image.add_ray(ray)
"""
