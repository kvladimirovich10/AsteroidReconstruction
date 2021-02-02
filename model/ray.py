import math as m
import numpy as np
import numpy.linalg as lg


class Ray:
    def __init__(self, normal_in_point=None, ray_to_point=None, velocity_in_point=None, area=None):
        self.normal_in_point = normal_in_point
        self.area = area
        self.velocity_in_point = np.array(velocity_in_point)
        self.ray_to_point = np.array(ray_to_point)
        
        self.distance_to_point = lg.norm(self.ray_to_point)
        self.angle_to_normal = self.calc_angle_to_normal()
        self.cos_of_angle_to_normal = np.cos(np.deg2rad(self.angle_to_normal))
        self.velocity_projection = self.calc_velocity_projection()
        self.velocity_sign_direction = np.dot(self.ray_to_point, self.velocity_projection)
        self.color = self.angle_to_normal#self.cos_of_angle_to_normal * self.area

    
    def calc_angle_to_normal(self):
        normalized_ray = -1 * np.array(self.ray_to_point / lg.norm(self.ray_to_point))
        return 180*m.acos(np.dot(normalized_ray, self.normal_in_point))/np.pi
    
    def calc_velocity_projection(self):
        return np.dot(self.ray_to_point, self.velocity_in_point) * self.ray_to_point / pow(
            lg.norm(self.ray_to_point), 2)
