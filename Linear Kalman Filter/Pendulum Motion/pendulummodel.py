import numpy as np

class PendulumModel():

    def __init__(self):
        self.length = 1
        self.mass = 1
        self.position = 0
        self.velocity = 0

    def initialise(self, pendulum_params):
        self.position = pendulum_params['initial_position']
        self.velocity = pendulum_params['initial_velocity']
        self.mass = pendulum_params['mass']
        self.length = pendulum_params['length']

    def update(self, time_step, torque):
        accel = -9.81/self.length * np.sin(self.position) #+ 1/(self.mass*self.length*self.length)*torque
        self.velocity = self.velocity + accel * time_step
        self.position = self.position + self.velocity * time_step
        return

    def get_position(self):
        return self.position

    def get_velocity(self):
        return self.velocity 

