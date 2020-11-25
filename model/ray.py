class Ray:
    def __init__(self, angle_to_point_normal, velocity_projection, distance_to_point):
        self.angle_to_point_normal = angle_to_point_normal
        self.velocity_projection = velocity_projection
        self.distance_to_point = distance_to_point