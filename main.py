import sys

import numpy as np
from PIL import Image

import math
from util import RayMarching as rm
import plotly.graph_objects as pgo
from util import utilMethods
import os, errno
import time
from model.ellipsoid import Ellipsoid


def init_ell(semi_axes_params):
    mass = 100
    x = np.array([10000, 0, 0])
    semi_axes = semi_axes_params
    euler_angles = {'alpha': 0, 'beta': 0, 'gamma': 0}
    
    P = np.array([0, 0, 0])
    L = np.array([-5, 3, 10])
    
    force = np.array([0, 0, 0])
    torque = np.array([0, 0, 0])
    
    return Ellipsoid(semi_axes, x, mass, euler_angles, P, L, force, torque)


def get_dist(ell, point):
    return math.sqrt(pow((point[0] - ell.x[0]) * ell.b * ell.c, 2) + \
                     pow((point[1] - ell.x[1]) * ell.a * ell.c, 2) + \
                     pow((point[2] - ell.x[2]) * ell.a * ell.b, 2)) - \
           ell.a * ell.b * ell.c


def get_point_in_ell_system(self, point):
    return np.matmul(self.rotation_matrix, point)


def make_ray_marching_image(radio_image):
    fig = pgo.Figure()
    fig.update_layout(
        scene=dict(
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False),
            zaxis=dict(showticklabels=False),
        )
    )
    
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
    
    fig.show()
    fig.write_html('ellipsoid_ray_marching.html', auto_open=True)


def silent_remove(filename):
    try:
        os.remove(f'{filename}.png')
    except OSError as e:
        if e.errno != errno.ENOENT:
            raise  # re-raise exception if a different error occurred


def sign_generator(i, a):
    return pow(-1, i) * ((i == a) or ((i + 1) == a))


def main_process(a, b, c):
    image_name_c = "radio_image_computed"
    image_name_o = "radio_image_observed"
    
    silent_remove(image_name_c)
    silent_remove(image_name_o)
    
    seconds = 60
    rate = 50
    time_step = 1 / rate
    grid_side_point_number = 100
    
    print('\nMOTION MODELING PART : OBSERVED')
    semi_axis_params_o = {'a': a, 'b': b, 'c': c}
    ellipsoid_o = init_ell(semi_axis_params_o)
    
    y_o = ellipsoid_o.body_to_array()
    
    for i in range(seconds * rate):
        sys.stdout.write(f'\r{i}/{seconds * rate}')
        sys.stdout.flush()
        y_o = ellipsoid_o.update_position(y_o, time_step)
    
    print('\nRAY MARCHING PART : OBSERVED')
    radio_image_o = rm.ellipsoid_ray_marching(ellipsoid_o, grid_side_point_number)

    x_lim, y_lim = utilMethods.get_axis_min_max(radio_image_o, radio_image_o)

    radio_image_o.build_image(image_name_o, x_lim, y_lim)

    return
    # ---------------------------------------------------
    
    eps = 0.1
    old_delta_a = -0.3
    old_delta_b = 0.1
    old_delta_c = -0.2
    
    iteration = 1
    last_best_index = 7
    
    best_norm_global = 100
    while best_norm_global > eps:
        
        best_match = [0] * 6
        
        for j in range(6):
            
            silent_remove(image_name_c)
            silent_remove(image_name_o)
            
            delta_a = old_delta_a + eps * sign_generator(j, 1)
            delta_b = old_delta_b + eps * sign_generator(j, 3)
            delta_c = old_delta_c + eps * sign_generator(j, 5)
            
            semi_axis_params_c = {'a': a + delta_a, 'b': b + delta_b, 'c': c + delta_c}
            ellipsoid_c = init_ell(semi_axis_params_c)
            
            print(
                f'\n{iteration} {j} delta_a = {delta_a} delta_b = {delta_b} delta_c = {delta_c} ---------------------')
            
            print('\nMOTION MODELING PART : COMPUTED')
            y_c = ellipsoid_c.body_to_array()
            
            for i in range(seconds * rate):
                sys.stdout.write(f'\r{i}/{seconds * rate}')
                sys.stdout.flush()
                y_c = ellipsoid_c.update_position(y_c, time_step)
            
            print('\nRAY MARCHING PART : COMPUTED')
            radio_image_c = rm.ellipsoid_ray_marching(ellipsoid_c, grid_side_point_number)
            
            x_lim, y_lim = utilMethods.get_axis_min_max(radio_image_c, radio_image_o)
            
            radio_image_o.build_image(image_name_o, x_lim, y_lim)
            radio_image_c.build_image(image_name_c, x_lim, y_lim)
            
            array_c = np.reshape(np.asarray(Image.open(f'{image_name_c}.png').convert('L')).astype(int), (1, -1)) / 255
            array_o = np.reshape(np.asarray(Image.open(f'{image_name_o}.png').convert('L')).astype(int), (1, -1)) / 255
            
            array_res = np.absolute(array_o - array_c)
            
            norm = np.linalg.norm(array_res)
            print(f'\nSHIFT NORM = {norm}')
            
            best_match[j] = norm
        
        best_norm = min(best_match)
        best_index = best_match.index(best_norm)
        
        if ((best_index % 2 == 0 and best_index + 1 == last_best_index) or
                (best_index % 2 == 1 and best_index - 1 == last_best_index)):
            best_norm = sorted(best_match)[1]
            best_index = best_match.index(best_norm)
        
        print(f'\n{iteration} BEST ACTION NORM = {best_index} {best_norm}')
        best_norm_global = best_norm
        last_best_index = best_index
        
        old_delta_a += eps * sign_generator(best_index, 1)
        old_delta_b += eps * sign_generator(best_index, 3)
        old_delta_c += eps * sign_generator(best_index, 5)
        
        iteration += 1


def motion_visualisation(a, b, c):
    rate = 25
    time_step = 1 / rate
    
    semi_axis_params_o = {'a': a, 'b': b, 'c': c}
    ellipsoid = init_ell(semi_axis_params_o)
    
    ellipsoid.motion_visualisation(time_step, rate)


def main():
    a = 1
    b = 3
    c = 1.3
    
    main_process(a, b, c)
    
    # motion_visualisation(a, b, c)


main()
