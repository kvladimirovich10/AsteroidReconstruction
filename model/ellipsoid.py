import util.utilMethods as utils
import numpy as np
import numpy.linalg as lg
from vpython import *
import vpython as vp
import time


class Ellipsoid:
    
    def __init__(self, semi_axes: dict, x, mass, euler_angles: dict, P, L, force, torque):
        self.a = semi_axes.get('a')
        self.b = semi_axes.get('b')
        self.c = semi_axes.get('c')
        
        self.euler_angles = euler_angles
        self.mass = mass
        
        self.x = x
        
        self.rot_matrix = utils.rot_matrix_by_angles(euler_angles.get('alpha'),
                                                     euler_angles.get('beta'),
                                                     euler_angles.get('gamma'))
        
        self.rot_matrix_flatten = utils.matrix_to_array(self.rot_matrix)
        
        self.I_body = utils.generate_inertia_body_matrix(self.a, self.b, self.c, self.mass)
        self.I_body_inv = np.linalg.inv(self.I_body)
        
        self.P = P
        self.L = L
        
        self.I_inv = None
        self.v = None
        self.omega = None
        
        self.force = force
        self.torque = torque
    
    def body_to_array(self):
        return np.concatenate((self.x, self.rot_matrix_flatten, self.P, self.L), axis=None)
    
    def array_to_body(self, y):
        self.x = np.array(y[0:3])
        self.rot_matrix_flatten = np.array(y[3:12])
        self.P = np.array(y[12:15])
        self.L = np.array(y[15:18])
        
        self.rot_matrix = np.reshape(self.rot_matrix_flatten, (-1, 3))
        
        self.v = np.true_divide(self.P, self.mass)
        self.I_inv = np.matmul(np.matmul(self.rot_matrix, self.I_body_inv), np.transpose(self.rot_matrix))
        self.omega = np.matmul(self.I_inv, self.L)
    
    def update_position(self, y0_state, time_step):
        y_state = utils.solver(self, y0_state, time_step)
        self.array_to_body(y_state)
        return y_state
    
    def dy_dt_to_array(self, y_state):
        self.array_to_body(y_state)
        
        rot_dt = np.matmul(utils.star_omega(self.omega), self.rot_matrix)
        rot_dt_flatten = utils.matrix_to_array(rot_dt)
        res = np.concatenate((self.v, rot_dt_flatten, self.force, self.torque), axis=None)
        return res
    
    def get_normal_vector_in_point(self, point):
        normal = np.array([2 * (point[0]) / pow(self.a, 2),
                           2 * (point[1]) / pow(self.b, 2),
                           2 * (point[2]) / pow(self.c, 2)])
        
        rotated_normal = np.matmul(self.rot_matrix, normal)
        
        return rotated_normal / np.linalg.norm(rotated_normal)
    
    def get_full_velocity_in_point(self, point):
        v_linear = self.v
        omega = np.array(self.omega)
        r = point - self.x
        
        v_angular = np.zeros(3)
        
        if np.count_nonzero(omega) > 0:
            r_proj_on_omega = omega * np.dot(omega, r) / pow(lg.norm(omega), 2)
            r_perp = r - r_proj_on_omega
            v_angular = np.cross(omega, r_perp)
        
        return v_angular + v_linear
    
    def motion_visualisation(self, time_step, frame_rate):
        s = canvas(title='Rigid ellipsoid free motion',
                   width=500, height=500,
                   background=color.gray(0.2))
        
        y = self.body_to_array()
        initial_pos = vector(0, 0, 0)
        
        ell_shape = ellipsoid(pos=utils.vector_from_array(self.x),
                              axis=vector(1, 0, 0),
                              length=2 * self.a, height=2 * self.b, width=2 * self.c,
                              texture=vp.textures.rough)
        
        v = [self.a / 2, 0, 0]
        trail_vector = arrow(pos=initial_pos, axis=utils.vector_from_array(v), shaftwidth=0.01,
                             color=vector(255, 255, 255))
        point_to_trail = sphere(pos=utils.vector_from_array(v), radius=0.01, color=color.red)
        attach_trail(point_to_trail, color=color.black, radius=0.01)
        
        arrow(pos=initial_pos, axis=vector(3, 0, 0), shaftwidth=0.01, color=vector(255, 0, 0))
        arrow(pos=initial_pos, axis=vector(0, 3, 0), shaftwidth=0.01, color=vector(0, 255, 0))
        arrow(pos=initial_pos, axis=vector(0, 0, 3), shaftwidth=0.01, color=vector(0, 0, 255))
        
        x_v_b = arrow(pos=initial_pos, axis=vector(3, 0, 0), shaftwidth=0.01, color=vector(255, 0, 0))
        y_v_b = arrow(pos=initial_pos, axis=vector(0, 3, 0), shaftwidth=0.01, color=vector(0, 255, 0))
        z_v_b = arrow(pos=initial_pos, axis=vector(0, 0, 3), shaftwidth=0.01, color=vector(0, 0, 255))
        
        v_v = arrow(pos=initial_pos, axis=initial_pos, shaftwidth=0.05, color=vector(255, 220, 0))
        omega_v = arrow(pos=initial_pos, axis=initial_pos, shaftwidth=0.05, color=vector(255, 0, 255))
        l_v = arrow(pos=initial_pos, axis=initial_pos, shaftwidth=0.05, color=vector(255, 255, 0))

        r = arrow(pos=initial_pos, axis=vector(0, self.b / 2, 0), shaftwidth=0.01,
                  color=vector(255, 255, 255))
        
        r_p = arrow(pos=ell_shape.pos, axis=vector(0, self.b / 2, 0), shaftwidth=0.01,
                    color=vector(255, 255, 255))
        
        v_c = arrow(pos=ell_shape.pos, axis=vector(0, self.b / 2, 0), shaftwidth=0.01,
                    color=vector(255, 255, 255))
        
        utils.initial_rotation(ell_shape, self)
        utils.initial_rotation(trail_vector, self)
        utils.initial_rotation(x_v_b, self)
        utils.initial_rotation(y_v_b, self)
        utils.initial_rotation(z_v_b, self)
        
        const = 5
        while True:
            point_to_trail.pos = trail_vector.axis + trail_vector.pos
            y = self.update_position(y, time_step)
            
            utils.translate(ell_shape, self)
            utils.translate(x_v_b, self)
            utils.translate(y_v_b, self)
            utils.translate(z_v_b, self)
            utils.translate(v_v, self)
            utils.translate(omega_v, self)
            utils.translate(r, self)
            utils.translate(trail_vector, self)

            utils.rotate(ell_shape, self)
            utils.rotate(trail_vector, self)
            utils.rotate(x_v_b, self)
            utils.rotate(y_v_b, self)
            utils.rotate(z_v_b, self)
            utils.rotate(r, self)

            v_v.axis = const * utils.vector_from_array(self.v).norm()
            omega_v.axis = const * utils.vector_from_array(self.omega).norm()
            l_v.axis = 3 * utils.vector_from_array(self.L).norm()

            # для отрисовки вектора общей скорости точки на поверхности
            r_projection = (np.dot(utils.array_from_vector(omega_v.axis), utils.array_from_vector(r.axis)) /
                            pow(np.linalg.norm(utils.array_from_vector(omega_v.axis)), 2)) * omega_v.axis
            
            r_p_axis = r.axis - r_projection
            
            r_p.pos = r_projection + ell_shape.pos
            r_p.axis = r_p_axis
            
            v_c.axis = utils.vector_from_array(np.cross(self.omega, utils
                                                        .array_from_vector(r_p.axis)) + self.v).norm()
            v_c.pos = ell_shape.pos + r.axis
            
            s.center = utils.vector_from_array(ell_shape.pos.value)
            time.sleep(time_step)
    
    def ray_marching_visualisation(self):
        canvas(title='Rigid ellipsoid free motion',
               width=1920, height=1080,
               background=color.gray(0.2))
        
        initial_pos = np.array([0, 0, 0])
        v_initial_pos = utils.vector_from_array(initial_pos)
        
        ellipsoid(pos=utils
                  .vector_from_array(self.x),
                  axis=vector(1, 0, 0),
                  length=2 * self.a, height=2 * self.b, width=2 * self.c,
                  texture=vp.textures.rough)
        
        arrow(pos=v_initial_pos, axis=vector(3, 0, 0), shaftwidth=0.01, color=vector(255, 0, 0))
        arrow(pos=v_initial_pos, axis=vector(0, 3, 0), shaftwidth=0.01, color=vector(0, 255, 0))
        arrow(pos=v_initial_pos, axis=vector(0, 0, 3), shaftwidth=0.01, color=vector(0, 0, 255))
        
        points(pos=[v_initial_pos], radius=5, color=color.white)
        
        arrow(pos=v_initial_pos, axis=utils
              .vector_from_array(self.x).norm(), shaftwidth=0.01,
              color=vector(255, 255, 255))
        
        rotated_point = self.get_point_in_ell_system(initial_pos)
        print(rotated_point)
        
        dist = utils.get_dist(self, rotated_point)
        print(dist)
    
    def get_point_in_ell_system(self, point):
        return np.matmul(self.rot_matrix, point)
