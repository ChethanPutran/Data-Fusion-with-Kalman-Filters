import numpy as np

class VehicleModel2D():

    def __init__(self):
        self.x_pos = 0
        self.y_pos = 0
        self.vel = 0
        self.yaw = 0

    def initialise(self, vehicle_params):
        self.x_pos = vehicle_params['initial_x_position']
        self.y_pos = vehicle_params['initial_y_position']
        self.yaw = vehicle_params['initial_heading']
        self.vel = vehicle_params['initial_speed']

    def update_vehicle(self, time_step, accel, yaw_rate):
        self.vel = self.vel + accel * time_step
        self.yaw = self.yaw + yaw_rate * time_step
        self.x_pos = self.x_pos + self.vel * np.cos(self.yaw) * time_step
        self.y_pos = self.y_pos + self.vel * np.sin(self.yaw) * time_step
        return

    def get_position(self):
        return [self.x_pos, self.y_pos]

    def get_velocity(self):
        x_vel = self.vel * np.cos(self.yaw)
        y_vel = self.vel * np.sin(self.yaw)
        return [x_vel, y_vel]

    def get_speed(self):
        return self.vel

    def get_heading(self):
        return self.yaw

