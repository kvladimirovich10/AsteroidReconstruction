class Ray:
    def __init__(self, normal_in_point=None, distance_to_point=None, velocity_in_point=None):
        self.normal_in_point = normal_in_point
        self.velocity_in_point = velocity_in_point
        self.distance_to_point = distance_to_point
        
        self.angle_to_normal = self.calc_angle_to_normal(normal_in_point)
        self.velocity_projection = self.calc_velocity_projection(velocity_in_point)
    
    def calc_angle_to_normal(self, normal):
        pass
    
    def calc_velocity_projection(self, velocity_in_point):
        pass
